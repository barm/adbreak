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

mod text;

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
    /// The captions: each cue's start and end (seconds) and its text, its lines joined with a space
    /// (`adbreak_captions::parse_vtt`). Read by models with text inputs.
    pub captions: Vec<(f64, f64, String)>,
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
    /// Also snap to black pictures without a silence ([`black_points`]). Absent in older models: edges only.
    #[serde(default)]
    snap_black: bool,
}

/// Another kind's output (`double_box`, `ad_read`): the picture alone, averaged over ±`smooth` seconds.
#[derive(Deserialize)]
struct KindHead {
    frame_coef: Vec<f32>,
    frame_intercept: f64,
    smooth: usize,
    post: Post,
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
    /// The other kinds' outputs, by kind name (absent from models that find commercial breaks only).
    #[serde(default)]
    kinds: std::collections::BTreeMap<String, KindHead>,
    /// Caption text inputs (absent from older models, whose trees read no text columns).
    #[serde(default)]
    text: Option<text::Text>,
}

/// A stretch of one kind: `ad` (a commercial break), `double_box` or `ad_read`.
#[derive(Clone, Debug, PartialEq)]
pub struct KindBreak {
    pub start: f64,
    pub end: f64,
    pub kind: String,
}

/// A trained model.
pub struct Model {
    m: Json,
}

/// The trailing text score's recent maximum is taken over these many seconds.
const TEXT_MAX: [usize; 2] = [30, 60];

/// The input columns the trees read, in order, for a model with caption text inputs or without; the model file must
/// list the same.
pub fn feature_names_with(windows: &[usize], text: bool) -> Vec<String> {
    let mut v = feature_names(windows);
    if text {
        v.extend(["text_now", "text_trail", "text_now_w7", "text_now_w30"].map(String::from));
        v.extend(TEXT_MAX.iter().map(|w| format!("text_trail_max{w}")));
    }
    v
}

/// The input columns of a model without caption text inputs.
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
        if m.frame_coef.len() != m.embedding_dim || m.kinds.values().any(|h| h.frame_coef.len() != m.embedding_dim) {
            return Err("frame weights don't match the embedding size".into());
        }
        if m.text.as_ref().is_some_and(|t| [&t.now, &t.trail_head].into_iter().flatten().any(|h| h.coef.len() != t.buckets) || t.trail == 0) {
            return Err("text weights don't match their bucket count".into());
        }
        if m.features != feature_names_with(&m.windows, m.text.is_some()) {
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
        self.linear(rows, &self.m.frame_coef, self.m.frame_intercept)
    }

    /// A logistic model of the picture embedding, each second; a second without a picture repeats the last.
    fn linear(&self, rows: &[Row], coef: &[f32], intercept: f64) -> Vec<f64> {
        let mut last = 0.5;
        rows.iter()
            .map(|r| {
                if r.embedding.len() == self.m.embedding_dim {
                    // In f64, as the training code computes it: a second right on a tree threshold must go the same way.
                    let z: f64 = r.embedding.iter().zip(coef).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() + intercept;
                    last = sigmoid(z);
                }
                last
            })
            .collect()
    }

    /// The other kinds this model finds besides commercial breaks (`double_box`, `ad_read`).
    pub fn kinds(&self) -> Vec<&str> {
        self.m.kinds.keys().map(String::as_str).collect()
    }

    /// Probability of each second being of `kind` (one of [`Model::kinds`]).
    pub fn kind_probabilities(&self, kind: &str, rows: &[Row]) -> Option<Vec<f64>> {
        let h = self.m.kinds.get(kind)?;
        Some(window_mean(&self.linear(rows, &h.frame_coef, h.frame_intercept), h.smooth))
    }

    /// Commercial breaks and the other kinds as one list, in time order. A double box takes its time from a
    /// commercial break it overlaps (it is often the break's tail); an ad read loses any part inside a commercial
    /// break or a double box; pieces shorter than their kind's `min_len` are dropped.
    pub fn kind_breaks(&self, inputs: &Inputs) -> Vec<KindBreak> {
        let spans = |b: Vec<Break>| -> Vec<(f64, f64)> { b.into_iter().map(|b| (b.start, b.end)).collect() };
        let ad = spans(self.breaks(inputs));
        let runs = |kind: &str| -> (Vec<(f64, f64)>, f64) {
            match (self.m.kinds.get(kind), self.kind_probabilities(kind, &inputs.rows)) {
                (Some(h), Some(p)) => (spans(to_breaks(&p, &inputs.edges, h.post.threshold, h.post.min_len, h.post.max_gap, h.post.snap)), h.post.min_len),
                _ => (vec![], 0.0),
            }
        };
        let (db, db_min) = runs("double_box");
        let (reads, read_min) = runs("ad_read");
        let db: Vec<(f64, f64)> = db.into_iter().filter(|(s, e)| e - s >= db_min).collect();
        let ads: Vec<(f64, f64)> = subtract(ad, &db).into_iter().filter(|(s, e)| e - s >= self.m.post.min_len).collect();
        let cuts: Vec<(f64, f64)> = ads.iter().chain(&db).copied().collect();
        let reads: Vec<(f64, f64)> = subtract(reads, &cuts).into_iter().filter(|(s, e)| e - s >= read_min).collect();
        let mut out: Vec<KindBreak> = [(ads, "ad"), (db, "double_box"), (reads, "ad_read")]
            .into_iter()
            .flat_map(|(v, k)| v.into_iter().map(move |(start, end)| KindBreak { start, end, kind: k.into() }))
            .collect();
        out.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)).then(a.kind.cmp(&b.kind)));
        out
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
        if let Some(t) = &self.m.text {
            let (now_sets, trail_sets) = text::per_second(&inputs.captions, n, t.buckets, t.trail);
            let now = text::score(&now_sets, t.now.as_ref());
            let trail = text::score(&trail_sets, t.trail_head.as_ref());
            cols.push(now.clone());
            cols.push(trail.clone());
            cols.push(window_mean(&now, 7));
            cols.push(window_mean(&now, 30));
            for w in TEXT_MAX {
                cols.push((0..n).map(|k| trail[k.saturating_sub(w)..=k].iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect());
            }
        }
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
        let black = if self.m.post.snap_black { black_points(&inputs.rows) } else { vec![] };
        let p = &self.m.post;
        to_breaks_snapping(&self.probabilities(inputs), &inputs.edges, &black, p.threshold, p.min_len, p.max_gap, p.snap)
    }
}

/// `spans` with every part inside `cuts` taken out.
fn subtract(mut spans: Vec<(f64, f64)>, cuts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    for &(x, y) in cuts {
        spans = spans.into_iter().flat_map(|(s, e)| [(s, e.min(x)), (s.max(y), e)]).filter(|(s, e)| e > s).collect();
    }
    spans
}

/// A model.json's version: FNV-1a 64 of its bytes, in hex. Recorders keep it with the breaks a model found.
pub fn version(model_json: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model_json.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// A test vector row: embedding, has_captions, caption_visible, caption_roll_up, caption_new, black, silent.
type JsonRow = (Vec<f32>, bool, bool, bool, bool, f32, f32);

/// `test_vector.json` from `adbreak-ml export`: inputs and the outputs the Python model gave for them.
#[derive(Debug, Deserialize)]
pub struct TestVector {
    /// The version of the model.json it was made for (absent before versions were recorded).
    #[serde(default)]
    pub model: Option<String>,
    rows: Vec<JsonRow>,
    edges: Vec<f64>,
    /// The captions of its span (for models with text inputs).
    #[serde(default)]
    captions: Vec<(f64, f64, String)>,
    probabilities: Vec<f64>,
    breaks: Vec<(f64, f64)>,
    /// The other kinds' probabilities, and every kind's breaks as [`Model::kind_breaks`] gives them.
    #[serde(default)]
    kind_probabilities: std::collections::BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    kind_breaks: Vec<(f64, f64, String)>,
}

impl TestVector {
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

impl Model {
    /// Checks that this model gives the test vector's outputs: a recorder can refuse a model that doesn't.
    pub fn check(&self, tv: &TestVector) -> Result<(), String> {
        let inputs = Inputs {
            rows: tv
                .rows
                .iter()
                .map(|(embedding, has_captions, caption_visible, caption_roll_up, caption_new, black, silent)| Row {
                    embedding: embedding.clone(),
                    has_captions: *has_captions,
                    caption_visible: *caption_visible,
                    caption_roll_up: *caption_roll_up,
                    caption_new: *caption_new,
                    black: *black,
                    silent: *silent,
                })
                .collect(),
            edges: tv.edges.clone(),
            captions: tv.captions.clone(),
        };
        let p = self.probabilities(&inputs);
        if p.len() != tv.probabilities.len() {
            return Err(format!("{} probabilities, the test vector has {}", p.len(), tv.probabilities.len()));
        }
        let worst = p.iter().zip(&tv.probabilities).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if worst >= 1e-4 {
            return Err(format!("probabilities differ from the test vector's by up to {worst}"));
        }
        let b: Vec<(f64, f64)> = self.breaks(&inputs).iter().map(|b| (b.start, b.end)).collect();
        if b != tv.breaks {
            return Err(format!("breaks {b:?}, the test vector has {:?}", tv.breaks));
        }
        for (kind, want) in &tv.kind_probabilities {
            let p = self.kind_probabilities(kind, &inputs.rows).ok_or_else(|| format!("no {kind} output, the test vector has one"))?;
            let worst = p.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            if p.len() != want.len() || worst >= 1e-4 {
                return Err(format!("{kind} probabilities differ from the test vector's by up to {worst}"));
            }
        }
        if !tv.kind_probabilities.is_empty() {
            let kb: Vec<(f64, f64, String)> = self.kind_breaks(&inputs).into_iter().map(|b| (b.start, b.end, b.kind)).collect();
            if kb != tv.kind_breaks {
                return Err(format!("kind breaks {kb:?}, the test vector has {:?}", tv.kind_breaks));
            }
        }
        Ok(())
    }
}

/// The middle of each run of seconds with any black picture: where a break can start or end on a channel that marks
/// it with a black frame but no silence.
pub fn black_points(rows: &[Row]) -> Vec<f64> {
    let mut pts = vec![];
    let mut i = 0;
    while i < rows.len() {
        if rows[i].black > 0.0 {
            let mut j = i;
            while j + 1 < rows.len() && rows[j + 1].black > 0.0 {
                j += 1;
            }
            pts.push((i + j + 1) as f64 / 2.0);
            i = j + 1;
        } else {
            i += 1;
        }
    }
    pts
}

/// Per-second probabilities to breaks: runs at or above `threshold`, gaps up to `max_gap` closed, each end snapped
/// to the nearest edge within `snap` seconds, runs shorter than `min_len` dropped.
pub fn to_breaks(p: &[f64], edges: &[f64], threshold: f64, min_len: f64, max_gap: f64, snap: f64) -> Vec<Break> {
    to_breaks_snapping(p, edges, &[], threshold, min_len, max_gap, snap)
}

/// [`to_breaks`], with `fallback` points (black pictures) to snap to where no edge is within `snap`.
pub fn to_breaks_snapping(p: &[f64], edges: &[f64], fallback: &[f64], threshold: f64, min_len: f64, max_gap: f64, snap: f64) -> Vec<Break> {
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
    let near = |points: &[f64], x: f64| points.iter().copied().min_by(|a, b| (a - x).abs().total_cmp(&(b - x).abs())).filter(|e| (e - x).abs() <= snap);
    // A black-and-silent edge if one is near (its time is exact), else a black picture (to the half second).
    let snapped = |x: f64| -> f64 { near(edges, x).or_else(|| near(fallback, x)).unwrap_or(x) };
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

    /// The model and test vector written by `adbreak-ml export` (set ADBREAK_MODEL_DIR to their directory).
    #[test]
    fn matches_the_python_model() {
        let Some(dir) = std::env::var_os("ADBREAK_MODEL_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let text = std::fs::read_to_string(dir.join("model.json")).unwrap();
        let m = Model::from_json(&text).unwrap();
        let tv = TestVector::from_json(&std::fs::read_to_string(dir.join("test_vector.json")).unwrap()).unwrap();
        assert_eq!(tv.model.as_deref().unwrap_or(&version(&text)), version(&text), "the test vector is for another model");
        m.check(&tv).unwrap();
    }

    #[test]
    fn version_is_fnv1a() {
        assert_eq!(version(""), "cbf29ce484222325");
        assert_eq!(version("a"), "af63dc4c8601ec8c");
    }

    #[test]
    fn black_points_are_the_middle_of_black_runs() {
        let row = |black: f32| Row { black, ..Default::default() };
        let rows: Vec<Row> = [0.0, 0.3, 0.0, 0.2, 1.0, 0.0, 0.0].into_iter().map(row).collect();
        assert_eq!(black_points(&rows), vec![1.5, 4.0]);
    }
}
