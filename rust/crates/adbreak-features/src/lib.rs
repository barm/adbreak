//! Per-second inputs for [`adbreak_model`] from a recording, computed the same way for training (`adbreak
//! features`) and for detection (the DVR).
//!
//! Feed an [`Extractor`] what a decoder produces, in presentation order and on the recording's clock:
//! - small grey pictures and mono sound ([`Extractor::push_frame`], [`Extractor::push_audio`]), about 10 a second
//!   at 480×270 (as the DVR scans), for black, silence and the black-and-silent edges (the adbreak detector);
//! - one colour picture a second ([`Extractor::push_color`]), the whole picture squashed to the embedder's size,
//!   planar RGB 0–1, for the picture embedding.
//!
//! Then [`Extractor::inputs`] gives the rows, with the recording's captions (WebVTT).

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
        let cues = adbreak_captions::parse_vtt(captions);
        let has = !cues.is_empty();
        let on = adbreak_captions::at_seconds(&cues, n);
        let black = cover(self.detector.black_stretches(), n);
        let silent = cover(self.detector.silences(), n);
        let rows = (0..n)
            .map(|s| Row {
                embedding: self.embeddings.get(&(s as i64)).cloned().unwrap_or_default(),
                has_captions: has,
                caption_visible: on[s].visible,
                caption_roll_up: on[s].roll_up,
                caption_new: on[s].new_cue,
                black: black[s],
                silent: silent[s],
            })
            .collect();
        Inputs { rows, edges: self.detector.edges() }
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
}
