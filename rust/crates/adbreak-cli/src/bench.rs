//! `adbreak bench`: how fast the break scan runs on a recording against Comskip, on the same file. Three timed
//! passes: decoding alone (the pictures the scan takes, nothing done with them), the whole scan (decoding, the
//! image model, black and silence: what a recorder runs), and Comskip. Each as a multiple of real time.

use adbreak_features::{CoreMlEmbedder, Extractor};
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

const SIZE: usize = 256;

/// Takes the pictures and sound and does nothing with them; notes how far the recording went.
struct Null(f64);

impl fmp4_scan::Sink for Null {
    fn picture(&mut self, pts: f64, _: usize, _: usize, _: &[u8]) {
        self.0 = self.0.max(pts);
    }
    fn sound(&mut self, _: f64, _: u32, _: &[f32]) {}
    fn color(&mut self, _: f64, _: usize, _: &[f32]) {}
}

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

/// Decodes `media` as the scan does, handing everything to `sink`.
fn decode(media: &Path, sink: &mut impl fmp4_scan::Sink) -> Result<()> {
    let (init, fragments) = fmp4::stream::open(media, 16 << 20)?;
    let mut scan = fmp4_scan::Scan::new(&init, 480, 270, 10.0)?.with_color(SIZE);
    for batch in fragments {
        scan.feed_stream(&batch?, sink)?;
    }
    Ok(())
}

pub fn run(media: &Path, model_dir: &Path, comskip: Option<&Path>) -> Result<()> {
    let t = Instant::now();
    let mut null = Null(0.0);
    decode(media, &mut null).with_context(|| format!("decoding {}", media.display()))?;
    let (decode_secs, length) = (t.elapsed().as_secs_f64(), null.0);
    let line = |what: &str, secs: f64| println!("  {what:34} {secs:8.1} s   {:7.1}× real time   {:5.1} s per hour", length / secs, 3600.0 * secs / length);
    println!("{}: {:.1} min", media.display(), length / 60.0);
    line("decoding alone", decode_secs);

    let t = Instant::now();
    let embedder = CoreMlEmbedder::load(&model_dir.join("mobileclip2-s0-image.mlpackage"), SIZE)?;
    let load_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut feed = Feed(Extractor::new(embedder), None);
    decode(media, &mut feed)?;
    if let Some(e) = feed.1.take() {
        return Err(e.context("embedding a picture"));
    }
    feed.0.finish();
    let inputs = feed.0.inputs("");
    let scan_secs = t.elapsed().as_secs_f64();
    line("the scan (decode, image model, b/s)", scan_secs);
    println!("  (the image model loaded in {load_secs:.1} s, once; {} edges found)", inputs.edges.len());

    let Some(comskip) = comskip else {
        println!("  no Comskip (--comskip or ADBREAK_COMSKIP): no comparison");
        return Ok(());
    };
    let work = std::env::temp_dir().join(format!("adbreak-bench-{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    let t = Instant::now();
    let found = crate::tools::comskip(comskip, media, &work);
    let comskip_secs = t.elapsed().as_secs_f64();
    let _ = std::fs::remove_dir_all(&work);
    let (breaks, _) = found?;
    line("Comskip", comskip_secs);
    println!("  ({} breaks)", breaks.len());
    println!("  the scan is {:.1}× as fast as Comskip (target: 10×)", comskip_secs / scan_secs);
    Ok(())
}
