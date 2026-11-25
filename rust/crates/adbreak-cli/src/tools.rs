//! Media work for `import` and `suggest`: avkit's hardware decoding (the copy, the detector's pictures and sound)
//! and Comskip, the one outside program (optional).

use adbreak::{Config, Detector, Frame};
use adbreak_labels::Span;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The picture size and rate the detector is tuned for.
const W: usize = 480;
const H: usize = 270;
const FPS: f64 = 10.0;

/// `name` from the flag or environment, else from `PATH`, else from Homebrew's prefix.
pub fn find(given: Option<&Path>, name: &str) -> Option<PathBuf> {
    if let Some(p) = given {
        return Some(p.to_path_buf());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path).chain([PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")]).map(|d| d.join(name)).find(|p| p.is_file())
}

/// Hands what avkit decodes to the detector.
struct Feed(Detector);

impl fmp4_scan::Sink for Feed {
    fn picture(&mut self, pts: f64, width: usize, height: usize, luma: &[u8]) {
        self.0.push_frame(Frame { pts, width, height, luma });
    }
    fn sound(&mut self, pts: f64, rate: u32, samples: &[f32]) {
        self.0.push_audio(pts, rate, samples);
    }
}

/// Runs the adbreak detector over a recording (a fragmented MP4: HEVC, AAC).
pub fn detect(media: &Path) -> Result<Vec<Span>> {
    let mut feed = Feed(Detector::new(Config::default()));
    fmp4_scan::scan_file(media, W, H, FPS, &mut feed).with_context(|| format!("decoding {}", media.display()))?;
    feed.0.finish();
    Ok(feed.0.breaks().into_iter().map(|b| Span::new(b.start, b.end)).collect())
}

/// Makes the dataset's copy of a recording (a fragmented MP4: HEVC, AAC) in `out`: `height` lines, every picture
/// (break edges are a few pictures long), mono sound; `media.mp4` with `index.m3u8` to seek by. Returns its length.
pub fn shrink(input: &Path, out: &Path, height: u32) -> Result<f64> {
    let settings = fmp4_shrink::Settings { height, ..Default::default() };
    let started = std::time::Instant::now();
    let mut shown = 0.0;
    let s = fmp4_shrink::shrink(input, out, &settings, &mut |secs| {
        if secs - shown >= 1200.0 {
            shown = secs;
            eprintln!("  {:.0} min, {:.0}x real time", secs / 60.0, secs / started.elapsed().as_secs_f64().max(0.001));
        }
    })
    .with_context(|| format!("making the copy of {}", input.display()))?;
    let kbps = s.bytes as f64 * 8.0 / s.duration.max(1.0) / 1000.0;
    eprintln!("  {}x{}, {} MB, {kbps:.0} kbit/s, in {:.0} s", s.width, s.height, s.bytes / 1_000_000, started.elapsed().as_secs_f64());
    Ok(s.duration)
}

/// Settings checked against frames on US network recordings: every detection method, an EDL out.
const COMSKIP_INI: &str = "detect_method=255\noutput_edl=1\noutput_default=0\nverbose=0\n";

/// Runs Comskip over a media file and reads its breaks (EDL). Returns its version line too.
pub fn comskip(comskip: &Path, media: &Path, work: &Path) -> Result<(Vec<Span>, String)> {
    let ini = work.join("comskip.ini");
    std::fs::write(&ini, COMSKIP_INI)?;
    let out = Command::new(comskip)
        .arg(format!("--ini={}", ini.display()))
        .arg(format!("--output={}", work.display()))
        .arg(media)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running {}", comskip.display()))?;
    let log = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let version = log.lines().find(|l| l.contains("Comskip")).unwrap_or("comskip").trim().to_string();
    let edl = work.join(media.with_extension("edl").file_name().unwrap_or_default());
    // Comskip exits 1 when it found no breaks, and writes no EDL then.
    let text = match std::fs::read_to_string(&edl) {
        Ok(t) => t,
        Err(_) if out.status.code() == Some(1) => String::new(),
        Err(e) => bail!("Comskip wrote no {} ({e}); its output:\n{log}", edl.display()),
    };
    Ok((parse_edl(&text), version))
}

/// EDL lines: `start end action`, seconds.
pub fn parse_edl(text: &str) -> Vec<Span> {
    text.lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace().map(str::parse::<f64>);
            Some(Span::new(f.next()?.ok()?, f.next()?.ok()?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn edl() {
        let s = super::parse_edl("455.44\t525.99\t0\n983.57\t1124.16\t0\n\n");
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].start, 983.57);
    }
}
