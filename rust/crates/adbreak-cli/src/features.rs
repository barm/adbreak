//! `adbreak features`: the model's per-second inputs from a recording's copy, through the same code a recorder uses to detect breaks
//! (avkit's decoder, adbreak-features), written for training: `features/rust-v1.f32` (rows of the embedding then
//! has_captions, visible, roll_up, new_cue, black, silent; little-endian f32) and `rust-v1.json` (shape, edges).
//!
//! From the full-resolution recording as well, when there is one (ingest has it until it is removed): the same files
//! named `full-v1`. These are the pictures a recorder scans, which the small copy only approximates (its embeddings
//! differ about five times as much as the recorder's own compression changes them), so training uses both.

use crate::dataset::Dataset;
use adbreak_features::{CoreMlEmbedder, Extractor};
use anyhow::{Context, Result};
use std::path::Path;

pub const NAME: &str = "rust-v1";
/// The inputs from the full-resolution recording.
pub const FULL: &str = "full-v1";
const SIZE: usize = 256;

struct Feed(Extractor<CoreMlEmbedder>, Option<anyhow::Error>);

impl fmp4_scan::Sink for Feed {
    fn picture(&mut self, pts: f64, width: usize, height: usize, luma: &[u8]) {
        self.0.push_frame(pts, width, height, luma);
    }
    fn sound(&mut self, pts: f64, rate: u32, samples: &[f32]) {
        self.0.push_audio(pts, rate, samples);
    }
    fn color(&mut self, pts: f64, size: usize, rgb: &[f32]) {
        if let Err(e) = self.0.push_color(pts, size, rgb) {
            self.1.get_or_insert(e);
        }
    }
}

/// From the recording's copy (`rust-v1`), or from `full`, the full-resolution recording (`full-v1`).
pub fn compute(data: &Dataset, id: &str, model_dir: &Path, full: Option<&Path>, force: bool) -> Result<()> {
    let dir = data.media_dir(id);
    let out = dir.join("features");
    let name = if full.is_some() { FULL } else { NAME };
    if !force && out.join(format!("{name}.json")).exists() {
        eprintln!("{id}: {name} features already computed");
        return Ok(());
    }
    let started = std::time::Instant::now();
    let embedder = CoreMlEmbedder::load(&model_dir.join("mobileclip2-s0-image.mlpackage"), SIZE)?;
    let mut feed = Feed(Extractor::new(embedder), None);
    let media = full.map_or_else(|| dir.join("media.mp4"), Path::to_path_buf);
    let (init, fragments) = fmp4::stream::open(&media, 16 << 20)?;
    let mut scan = fmp4_scan::Scan::new(&init, 480, 270, 10.0)?.with_color(SIZE);
    for batch in fragments {
        scan.feed_stream(&batch?, &mut feed)?;
        if let Some(e) = feed.1.take() {
            return Err(e.context("embedding a picture"));
        }
    }
    feed.0.finish();
    let captions = std::fs::read_to_string(dir.join("captions.vtt")).unwrap_or_default();
    let inputs = feed.0.inputs(&captions);
    let missing = adbreak_features::store::write(&out, name, &inputs, "mobileclip2-s0", SIZE).context("writing features")?;
    eprintln!(
        "{id} ({name}): {} s, {} edges, {missing} seconds without a picture, in {:.0} s",
        inputs.rows.len(),
        inputs.edges.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
