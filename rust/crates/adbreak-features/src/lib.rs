//! Per-second inputs for [`adbreak_model`] from a recording, computed the same way for training (`adbreak
//! features`) and for detection (a recorder scanning as it records).
//!
//! Feed an [`Extractor`] what a decoder produces, in presentation order and on the recording's clock:
//! - small grey pictures and mono sound ([`Extractor::push_frame`], [`Extractor::push_audio`]), about 10 a second
//!   at 480×270, for black, silence and the black-and-silent edges (the adbreak detector);
//! - one colour picture a second ([`Extractor::push_color`]), the whole picture squashed to the embedder's size,
//!   planar RGB 0–1, for the picture embedding.
//!
//! Then [`Extractor::inputs`] gives the rows, with the recording's captions (WebVTT).
//!
//! [`store`] keeps them in a file (what `adbreak features` writes for training, and what a recorder keeps to find a
//! recording's breaks again with a new model without decoding it again); [`with_captions`] fills in the captions.

#![forbid(unsafe_code)]

use adbreak::{Config, Detector, Frame};
use adbreak_model::{Inputs, Row};
use std::collections::BTreeMap;

/// Turns one colour picture into an embedding.
pub trait Embedder {
    /// Pictures it takes: `size`×`size`.
    fn size(&self) -> usize;
    fn embed(&mut self, rgb: &[f32]) -> anyhow::Result<Vec<f32>>;
}

/// Collects a recording's per-second inputs as it is decoded.
pub struct Extractor<E> {
    detector: Detector,
    embedder: E,
    /// Embedding per whole second.
    embeddings: BTreeMap<i64, Vec<f32>>,
    /// Latest presentation time seen, seconds.
    end: f64,
}

impl<E: Embedder> Extractor<E> {
    pub fn new(embedder: E) -> Self {
        Self { detector: Detector::new(Config::default()), embedder, embeddings: BTreeMap::new(), end: 0.0 }
    }

    pub fn push_frame(&mut self, pts: f64, width: usize, height: usize, luma: &[u8]) {
        self.end = self.end.max(pts);
        self.detector.push_frame(Frame { pts, width, height, luma });
    }

    pub fn push_audio(&mut self, pts: f64, rate: u32, samples: &[f32]) {
        self.detector.push_audio(pts, rate, samples);
    }

    /// A colour picture shown at `pts`; the first of each second is embedded, the rest ignored.
    pub fn push_color(&mut self, pts: f64, size: usize, rgb: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(size == self.embedder.size(), "colour pictures are {size} square; the embedder takes {}", self.embedder.size());
        let s = pts.floor() as i64;
        if s >= 0 && !self.embeddings.contains_key(&s) {
            let e = self.embedder.embed(rgb)?;
            self.embeddings.insert(s, e);
        }
        Ok(())
    }

    /// Close black or silent stretches still open at the end. More may be pushed afterwards.
    pub fn finish(&mut self) {
        self.detector.finish();
    }

    /// The detector, e.g. for its own (logo-based) breaks.
    pub fn detector(&self) -> &Detector {
        &self.detector
    }

    /// The inputs so far: one row per whole second up to the last picture, and the edges. `captions` is the
    /// recording's WebVTT ("" if it has none).
    pub fn inputs(&self, captions: &str) -> Inputs {
        let n = (self.end.floor() as usize) + 1;
        let black = cover(self.detector.black_stretches(), n);
        let silent = cover(self.detector.silences(), n);
        let rows = (0..n)
            .map(|s| Row { embedding: self.embeddings.get(&(s as i64)).cloned().unwrap_or_default(), black: black[s], silent: silent[s], ..Row::default() })
            .collect();
        let mut inputs = Inputs { rows, edges: self.detector.edges(), captions: vec![] };
        with_captions(&mut inputs, captions);
        inputs
    }
}

/// Sets the caption inputs of every row, and the caption texts, from a recording's WebVTT ("" if it has none).
pub fn with_captions(inputs: &mut Inputs, captions: &str) {
    let n = inputs.rows.len();
    let cues = adbreak_captions::parse_vtt(captions);
    let has = !cues.is_empty();
    let on = adbreak_captions::at_seconds(&cues, n);
    for (r, o) in inputs.rows.iter_mut().zip(&on) {
        r.has_captions = has;
        r.caption_visible = o.visible;
        r.caption_roll_up = o.roll_up;
        r.caption_new = o.new_cue;
    }
    inputs.captions = cues.iter().map(|c| (c.start, c.end, c.lines.join(" "))).collect();
}

/// A recording's inputs in a file pair: `<name>.f32`, one row per second (the embedding, then has_captions,
/// visible, roll_up, new_cue, black, silent; little-endian f32; a second without a picture has NaN embeddings), and
/// `<name>.json` (rows, dim, columns, edges, image_model, size). The `.json` is written last, so a reader that finds
/// it finds the whole `.f32`.
pub mod store {
    use adbreak_model::{Inputs, Row};
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};

    /// Columns after the embedding.
    pub const COLUMNS: [&str; 6] = ["has_captions", "visible", "roll_up", "new_cue", "black", "silent"];

    /// Inputs read back, with the image model their embeddings came from.
    pub struct Stored {
        /// The rows and edges; caption texts are not kept (see [`super::with_captions`]).
        pub inputs: Inputs,
        pub image_model: String,
        /// The square picture size the image model took.
        pub size: usize,
        /// Seconds without a picture.
        pub missing: usize,
    }

    fn paths(dir: &Path, name: &str) -> (PathBuf, PathBuf) {
        (dir.join(format!("{name}.f32")), dir.join(format!("{name}.json")))
    }

    /// Writes `inputs` to `dir` as `name`. Returns the number of seconds without a picture.
    pub fn write(dir: &Path, name: &str, inputs: &Inputs, image_model: &str, size: usize) -> Result<usize> {
        let dim = inputs.rows.iter().map(|r| r.embedding.len()).max().unwrap_or(0);
        let mut bytes = Vec::with_capacity(inputs.rows.len() * (dim + COLUMNS.len()) * 4);
        let mut missing = 0;
        for r in &inputs.rows {
            let have = r.embedding.len() == dim;
            missing += usize::from(!have);
            let flag = |b: bool| f32::from(u8::from(b));
            let tail = [flag(r.has_captions), flag(r.caption_visible), flag(r.caption_roll_up), flag(r.caption_new), r.black, r.silent];
            for v in (0..dim).map(|i| if have { r.embedding[i] } else { f32::NAN }).chain(tail) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::create_dir_all(dir)?;
        let (data, meta) = paths(dir, name);
        let json = serde_json::json!({
            "rows": inputs.rows.len(), "dim": dim, "columns": COLUMNS, "edges": inputs.edges, "image_model": image_model, "size": size,
        });
        // The .json marks the pair complete: gone while the .f32 is replaced, back once it is whole.
        let _ = std::fs::remove_file(&meta);
        let tmp = dir.join(format!(".{name}.tmp"));
        std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, &data)).with_context(|| format!("writing {}", data.display()))?;
        std::fs::write(&tmp, serde_json::to_string(&json)?)
            .and_then(|()| std::fs::rename(&tmp, &meta))
            .with_context(|| format!("writing {}", meta.display()))?;
        Ok(missing)
    }

    /// Whether `dir` has a complete `name` pair.
    pub fn exists(dir: &Path, name: &str) -> bool {
        paths(dir, name).1.is_file()
    }

    /// Removes the `name` pair from `dir`, if there.
    pub fn remove(dir: &Path, name: &str) {
        let (data, meta) = paths(dir, name);
        let _ = std::fs::remove_file(meta);
        let _ = std::fs::remove_file(data);
    }

    pub fn read(dir: &Path, name: &str) -> Result<Stored> {
        let (data, meta) = paths(dir, name);
        let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta).with_context(|| format!("reading {}", meta.display()))?)?;
        let num = |k: &str| m[k].as_u64().map(|v| v as usize).with_context(|| format!("{}: no {k}", meta.display()));
        let (rows, dim) = (num("rows")?, num("dim")?);
        anyhow::ensure!(m["columns"] == serde_json::json!(COLUMNS), "{}: other columns than these", meta.display());
        let bytes = std::fs::read(&data).with_context(|| format!("reading {}", data.display()))?;
        let width = dim + COLUMNS.len();
        anyhow::ensure!(bytes.len() == rows * width * 4, "{}: {} bytes, expected {}", data.display(), bytes.len(), rows * width * 4);
        let vals: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let mut missing = 0;
        let rows = vals
            .chunks_exact(width)
            .map(|r| {
                let have = dim > 0 && !r[0].is_nan();
                missing += usize::from(!have);
                Row {
                    embedding: if have { r[..dim].to_vec() } else { vec![] },
                    has_captions: r[dim] != 0.0,
                    caption_visible: r[dim + 1] != 0.0,
                    caption_roll_up: r[dim + 2] != 0.0,
                    caption_new: r[dim + 3] != 0.0,
                    black: r[dim + 4],
                    silent: r[dim + 5],
                }
            })
            .collect();
        let edges = m["edges"].as_array().map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect()).unwrap_or_default();
        Ok(Stored {
            inputs: Inputs { rows, edges, captions: vec![] },
            image_model: m["image_model"].as_str().unwrap_or_default().to_string(),
            size: num("size")?,
            missing,
        })
    }
}

/// Share of each whole second 0..n covered by the spans.
pub fn cover(spans: &[(f64, f64)], n: usize) -> Vec<f32> {
    let mut out = vec![0f64; n];
    for &(a, b) in spans {
        let (lo, hi) = (a.floor().max(0.0) as usize, (b.ceil().max(0.0) as usize).min(n));
        for (s, o) in out.iter_mut().enumerate().take(hi).skip(lo) {
            *o += (b.min(s as f64 + 1.0) - a.max(s as f64)).max(0.0);
        }
    }
    out.into_iter().map(|v| v.clamp(0.0, 1.0) as f32).collect()
}

/// The image model through Core ML: an `.mlpackage` (or compiled `.mlmodelc`) taking `image` [1,3,size,size]
/// planar RGB 0–1 and giving `embedding`, as `adbreak-ml export` writes it.
#[cfg(feature = "coreml")]
pub struct CoreMlEmbedder {
    model: apple_coreml::Model,
    size: usize,
}

#[cfg(feature = "coreml")]
impl CoreMlEmbedder {
    pub fn load(path: &std::path::Path, size: usize) -> anyhow::Result<Self> {
        Ok(Self { model: apple_coreml::Model::load(path, apple_coreml::Compute::All)?, size })
    }
}

#[cfg(feature = "coreml")]
impl Embedder for CoreMlEmbedder {
    fn size(&self) -> usize {
        self.size
    }
    fn embed(&mut self, rgb: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.model.predict("image", &[1, 3, self.size, self.size], rgb, "embedding")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mean;
    impl Embedder for Mean {
        fn size(&self) -> usize {
            2
        }
        fn embed(&mut self, rgb: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(vec![rgb.iter().sum::<f32>() / rgb.len() as f32])
        }
    }

    #[test]
    fn covers_seconds() {
        assert_eq!(cover(&[(0.5, 2.25)], 4), vec![0.5, 1.0, 0.25, 0.0]);
        assert_eq!(cover(&[(3.0, 9.0)], 4), vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rows_from_pictures_sound_and_captions() {
        let mut x = Extractor::new(Mean);
        let bright = vec![200u8; 48 * 27];
        let black = vec![16u8; 48 * 27];
        for i in 0..40 {
            let t = i as f64 / 10.0;
            // Black and silent from 2.0 to 2.6 s.
            let dark = (2.0..2.6).contains(&t);
            x.push_frame(t, 48, 27, if dark { &black } else { &bright });
            x.push_audio(t, 1000, &[if dark { 0.0 } else { 0.3 }; 100]);
            x.push_color(t, 2, &[if dark { 0.0 } else { 0.5 }; 12]).unwrap();
        }
        x.finish();
        let vtt = "WEBVTT\n\n00:00.500 --> 00:01.500\nHELLO\n";
        let inp = x.inputs(vtt);
        assert_eq!(inp.rows.len(), 4);
        assert_eq!(inp.rows[0].embedding, vec![0.5]);
        assert_eq!(inp.rows[2].embedding, vec![0.0], "the first picture of second 2 is dark");
        assert!(inp.rows[1].has_captions && inp.rows[1].caption_visible && !inp.rows[3].caption_visible);
        assert!(inp.rows[2].black > 0.5 && inp.rows[0].black == 0.0, "{:?}", inp.rows[2]);
        assert!(inp.rows[2].silent > 0.5);
        assert_eq!(inp.edges.len(), 1, "{:?}", inp.edges);
    }

    #[test]
    fn stores_and_reads_back() {
        let mut x = Extractor::new(Mean);
        for i in 0..30 {
            let t = i as f64 / 10.0;
            x.push_frame(t, 48, 27, &vec![if i < 5 { 16u8 } else { 200 }; 48 * 27]);
            x.push_audio(t, 1000, &[if i < 5 { 0.0 } else { 0.3 }; 100]);
            if t >= 1.0 {
                x.push_color(t, 2, &[0.25; 12]).unwrap(); // no picture for second 0
            }
        }
        x.finish();
        let vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\nHI\n";
        let inp = x.inputs(vtt);
        let dir = std::env::temp_dir().join(format!("adbreak-store-{}", std::process::id()));
        assert_eq!(store::write(&dir, "t", &inp, "m", 2).unwrap(), 1, "second 0 has no picture");
        assert!(store::exists(&dir, "t"));
        let mut back = store::read(&dir, "t").unwrap();
        assert_eq!((back.image_model.as_str(), back.size, back.missing), ("m", 2, 1));
        assert_eq!(back.inputs.edges, inp.edges);
        assert!(back.inputs.captions.is_empty(), "caption texts are not kept");
        with_captions(&mut back.inputs, vtt);
        assert_eq!(back.inputs.rows, inp.rows);
        assert_eq!(back.inputs.captions, inp.captions);
        store::remove(&dir, "t");
        assert!(!store::exists(&dir, "t"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
