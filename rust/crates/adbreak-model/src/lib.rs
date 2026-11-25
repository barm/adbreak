//! adbreak's commercial-break model, from per-second inputs to breaks.
//!
//! Each second of a recording is described by a [`Row`]: an embedding of one picture from that second (from the
//! image model the [`Model`] names; computing it is the caller's job, as are decoding and the other inputs),
//! what the captions were doing, and how much of it was black or silent. With the moments that were both black
//! and silent ([`Inputs::edges`]), [`Model::breaks`] finds the breaks:
//!
//! 1. a logistic regression on the embedding gives each second a break probability on the picture alone;
//! 2. that, the captions and the black/silence inputs, averaged over windows from ±2 s to ±150 s, go through
//!    gradient-boosted trees for the final probability;
//! 3. a threshold, short gaps closed, short runs dropped, and each break's ends snapped to a black-and-silent
//!    moment nearby.
//!
//! The trees and weights come from `model.json`, written by `adbreak-ml export` (adbreak's `ml/`), which also writes
//! test vectors this crate's tests check against, so the Python training and this code compute the same thing.

#![forbid(unsafe_code)]

use serde::Deserialize;

/// The model file's format version this code reads.
pub const FORMAT: u32 = 1;

/// One second of a recording.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    /// The picture embedding (length [`Model::embedding_dim`]); empty if no picture was decoded for this second
    /// (the frame probability is then carried over from the previous second).
    pub embedding: Vec<f32>,
    /// The recording has captions at all.
    pub has_captions: bool,
    /// A caption cue is on screen, it rolled up from the one before, it started within the last second
    /// (`adbreak_captions::at_seconds`).
    pub caption_visible: bool,
    pub caption_roll_up: bool,
    pub caption_new: bool,
    /// Share of the second that was black, and silent (0–1).
    pub black: f32,
    pub silent: f32,
}

/// A recording's inputs: one row per second from 0, and the moments (seconds) that were both black and silent.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub rows: Vec<Row>,
    pub edges: Vec<f64>,
}

/// A break found: seconds from the start of the recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Break {
    pub start: f64,
    pub end: f64,
}

#[derive(Deserialize)]
struct Node {
    /// Feature index, threshold (go left if ≤), children; or a leaf with its value.
    #[serde(default)]
    f: usize,
    #[serde(default)]
    t: f64,
    #[serde(default)]
    l: usize,
    #[serde(default)]
    r: usize,
    #[serde(default)]
    leaf: bool,
    #[serde(default)]
    v: f64,
    /// Where a missing (NaN) value goes.
    #[serde(default)]
    missing_left: bool,
}

#[derive(Deserialize)]
struct Post {
    threshold: f64,
    min_len: f64,
    max_gap: f64,
    snap: f64,
}

#[derive(Deserialize)]
struct Json {
    format: u32,
    image_model: String,
    embedding_dim: usize,
    frame_coef: Vec<f32>,
    frame_intercept: f64,
    windows: Vec<usize>,
    edge_cap: f64,
    features: Vec<String>,
    baseline: f64,
    trees: Vec<Vec<Node>>,
    post: Post,
}

/// A trained model.
pub struct Model {
    m: Json,
}

/// The input columns the trees read, in order; the model file must list the same.
pub fn feature_names(windows: &[usize]) -> Vec<String> {
    let mut v: Vec<String> =
        ["p", "has_captions", "visible", "roll_up", "new_cue", "black", "silent", "edge_dist", "edge_here"].iter().map(|s| s.to_string()).collect();
    for w in windows {
        v.extend([format!("p_w{w}"), format!("visible_w{w}"), format!("roll_up_w{w}"), format!("edges_w{w}")]);
    }
    v.push("p_std15".into());
    v
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Mean of `x` over `i - half ..= i + half`, clipped to the recording.
pub fn window_mean(x: &[f64], half: usize) -> Vec<f64> {
    let mut c = vec![0.0; x.len() + 1];
    for (i, v) in x.iter().enumerate() {
        c[i + 1] = c[i] + v;
    }
    (0..x.len())
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(half), (i + half + 1).min(x.len()));
            (c[hi] - c[lo]) / (hi - lo).max(1) as f64
        })
        .collect()
}

impl Model {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let m: Json = serde_json::from_str(text).map_err(|e| format!("model file: {e}"))?;
        if m.format != FORMAT {
            return Err(format!("model file format {} (this code reads {FORMAT})", m.format));
        }
        if m.frame_coef.len() != m.embedding_dim {
            return Err("frame weights don't match the embedding size".into());
        }
        if m.features != feature_names(&m.windows) {
            return Err("the model's inputs don't match this code's: retrain or update".into());
        }
        Ok(Model { m })
    }

    /// The image model the embeddings must come from, e.g. `mobileclip2-s0`.
    pub fn image_model(&self) -> &str {
        &self.m.image_model
    }

    pub fn embedding_dim(&self) -> usize {
        self.m.embedding_dim
    }

    /// Break probability of each second from its picture alone.
    pub fn frame_probabilities(&self, rows: &[Row]) -> Vec<f64> {
        let mut last = 0.5;
        rows.iter()
            .map(|r| {
                if r.embedding.len() == self.m.embedding_dim {
                    // In f64, as the training code computes it: a second right on a tree threshold must go the same way.
                    let z: f64 = r.embedding.iter().zip(&self.m.frame_coef).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() + self.m.frame_intercept;
                    last = sigmoid(z);
                }
                last
            })
            .collect()
    }

    /// The trees' input table: one row per second, columns as [`feature_names`].
    pub fn table(&self, inputs: &Inputs, p: &[f64]) -> Vec<Vec<f64>> {
        let n = inputs.rows.len();
        let col = |f: &dyn Fn(&Row) -> f64| -> Vec<f64> { inputs.rows.iter().map(f).collect() };
        let has = col(&|r| f64::from(u8::from(r.has_captions)));
        let vis = col(&|r| f64::from(u8::from(r.caption_visible)));
        let roll = col(&|r| f64::from(u8::from(r.caption_roll_up)));
        let new = col(&|r| f64::from(u8::from(r.caption_new)));
        let black = col(&|r| f64::from(r.black));
        let silent = col(&|r| f64::from(r.silent));
        let mut edges = inputs.edges.clone();
        edges.sort_by(f64::total_cmp);
        let (mut dist, mut here) = (vec![0.0; n], vec![0.0; n]);
        for s in 0..n {
            let t = s as f64;
            let d = edges.iter().map(|e| (t - e).abs()).fold(f64::INFINITY, f64::min);
            dist[s] = d.min(self.m.edge_cap) / self.m.edge_cap;
            here[s] = f64::from(u8::from(d < 1.0));
        }
        let mut cols: Vec<Vec<f64>> = vec![p.to_vec(), has, vis.clone(), roll.clone(), new, black, silent, dist, here.clone()];
        for &w in &self.m.windows {
            cols.push(window_mean(p, w));
            cols.push(window_mean(&vis, w));
            cols.push(window_mean(&roll, w));
            cols.push(window_mean(&here, w).into_iter().map(|v| v * (2 * w + 1) as f64).collect());
        }
        let p2: Vec<f64> = p.iter().map(|v| v * v).collect();
        let (m1, m2) = (window_mean(p, 15), window_mean(&p2, 15));
        cols.push(m1.iter().zip(&m2).map(|(a, b)| (b - a * a).max(0.0).sqrt()).collect());
        // Rounded to f32, as in training: Rust and NumPy can differ in the last bit of f64 arithmetic, which would send
        // a value sitting on a tree threshold the other way.
        (0..n).map(|i| cols.iter().map(|c| f64::from(c[i] as f32)).collect()).collect()
    }

    fn tree(&self, nodes: &[Node], x: &[f64]) -> f64 {
        let mut i = 0;
        loop {
            let n = &nodes[i];
            if n.leaf {
                return n.v;
            }
            let v = x[n.f];
            i = if v.is_nan() {
                if n.missing_left { n.l } else { n.r }
            } else if v <= n.t {
                n.l
            } else {
                n.r
            };
        }
    }

    /// Break probability of each second.
    pub fn probabilities(&self, inputs: &Inputs) -> Vec<f64> {
        let p = self.frame_probabilities(&inputs.rows);
        self.table(inputs, &p).iter().map(|x| sigmoid(self.m.baseline + self.m.trees.iter().map(|t| self.tree(t, x)).sum::<f64>())).collect()
    }

    pub fn breaks(&self, inputs: &Inputs) -> Vec<Break> {
        to_breaks(&self.probabilities(inputs), &inputs.edges, self.m.post.threshold, self.m.post.min_len, self.m.post.max_gap, self.m.post.snap)
    }
}

/// Per-second probabilities to breaks: runs at or above `threshold`, gaps up to `max_gap` closed, each end snapped
/// to the nearest edge within `snap` seconds, runs shorter than `min_len` dropped.
pub fn to_breaks(p: &[f64], edges: &[f64], threshold: f64, min_len: f64, max_gap: f64, snap: f64) -> Vec<Break> {
    let mut spans: Vec<(f64, f64)> = vec![];
    let mut start: Option<f64> = None;
    for (i, &v) in p.iter().enumerate() {
        let t = i as f64;
        match (v >= threshold, start) {
            (true, None) => start = Some(t),
            (false, Some(s)) => {
                spans.push((s, t));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        spans.push((s, (p.len().max(1) - 1) as f64 + 1.0));
    }
    let mut merged: Vec<(f64, f64)> = vec![];
    for s in spans {
        match merged.last_mut() {
            Some(m) if s.0 - m.1 <= max_gap => m.1 = s.1,
            _ => merged.push(s),
        }
    }
    let snapped =
        |x: f64| -> f64 { edges.iter().copied().min_by(|a, b| (a - x).abs().total_cmp(&(b - x).abs())).filter(|e| (e - x).abs() <= snap).unwrap_or(x) };
    merged
        .into_iter()
        .map(|(a, b)| (snapped(a), snapped(b)))
        .filter(|(a, b)| b - a >= min_len)
        .map(|(a, b)| Break { start: (a * 100.0).round() / 100.0, end: (b * 100.0).round() / 100.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_means_clip_at_the_ends() {
        assert_eq!(window_mean(&[1.0, 2.0, 3.0, 4.0], 1), vec![1.5, 2.0, 3.0, 3.5]);
        assert_eq!(window_mean(&[5.0], 10), vec![5.0]);
    }

    #[test]
    fn runs_to_breaks() {
        let mut p = vec![0.0; 200];
        p[20..60].iter_mut().for_each(|v| *v = 0.9);
        p[70..100].iter_mut().for_each(|v| *v = 0.9); // gap of 10 s: merged
        p[150..160].iter_mut().for_each(|v| *v = 0.9); // too short
        let b = to_breaks(&p, &[18.5, 101.2], 0.35, 30.0, 25.0, 20.0);
        assert_eq!(b, vec![Break { start: 18.5, end: 101.2 }]);
    }

    /// The model and test vectors written by `adbreak-ml export` (set ADBREAK_MODEL_DIR to their directory).
    #[test]
    fn matches_the_python_model() {
        let Some(dir) = std::env::var_os("ADBREAK_MODEL_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let m = Model::from_json(&std::fs::read_to_string(dir.join("model.json")).unwrap()).unwrap();
        type JsonRow = (Vec<f32>, bool, bool, bool, bool, f32, f32);
        #[derive(Deserialize)]
        struct Case {
            rows: Vec<JsonRow>,
            edges: Vec<f64>,
            probabilities: Vec<f64>,
            breaks: Vec<(f64, f64)>,
        }
        let c: Case = serde_json::from_str(&std::fs::read_to_string(dir.join("test_vector.json")).unwrap()).unwrap();
        let inputs = Inputs {
            rows: c
                .rows
                .into_iter()
                .map(|(embedding, has_captions, caption_visible, caption_roll_up, caption_new, black, silent)| Row {
                    embedding,
                    has_captions,
                    caption_visible,
                    caption_roll_up,
                    caption_new,
                    black,
                    silent,
                })
                .collect(),
            edges: c.edges,
        };
        let p = m.probabilities(&inputs);
        let worst = p.iter().zip(&c.probabilities).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "probabilities differ by up to {worst}");
        let b: Vec<(f64, f64)> = m.breaks(&inputs).iter().map(|b| (b.start, b.end)).collect();
        assert_eq!(b, c.breaks);
    }
}
