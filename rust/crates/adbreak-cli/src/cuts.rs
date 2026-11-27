//! `adbreak cuts`: the shot cuts in a recording's copy, every picture compared with the one before
//! (`adbreak::CutDetector`), written to `features/cuts.json` for the labelling page to show and snap to.

use crate::dataset::Dataset;
use adbreak::CutDetector;
use anyhow::{Context, Result};

struct Feed(CutDetector);

impl fmp4_scan::Sink for Feed {
    fn picture(&mut self, pts: f64, _: usize, _: usize, luma: &[u8]) {
        self.0.push(pts, luma);
    }
    fn sound(&mut self, _: f64, _: u32, _: &[f32]) {}
}

pub fn compute(data: &Dataset, id: &str, force: bool) -> Result<()> {
    let dir = data.media_dir(id);
    let out = dir.join("features/cuts.json");
    if !force && out.exists() {
        eprintln!("{id}: cuts already found");
        return Ok(());
    }
    let started = std::time::Instant::now();
    let (init, fragments) = fmp4::stream::open(&dir.join("media.mp4"), 16 << 20)?;
    // Every picture (up to 120 a second), small: a cut shows at any size.
    let mut scan = fmp4_scan::Scan::new(&init, 64, 36, 120.0)?;
    let mut feed = Feed(CutDetector::new());
    for batch in fragments {
        scan.feed_stream(&batch?, &mut feed)?;
    }
    let cuts: Vec<f64> = feed.0.cuts().iter().map(|t| (t * 1000.0).round() / 1000.0).collect();
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, serde_json::to_string(&serde_json::json!({ "cuts": cuts }))?).context("writing cuts")?;
    eprintln!("{id}: {} cuts in {:.0} s", cuts.len(), started.elapsed().as_secs_f64());
    Ok(())
}
