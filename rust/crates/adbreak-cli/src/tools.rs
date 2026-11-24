//! The outside programs `suggest` runs: ffmpeg (decoding for the detector) and Comskip.

use adbreak::{Config, Detector, Frame};
use adbreak_labels::Span;
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;

/// The same picture size and rate the DVR scans at.
const W: usize = 480;
const H: usize = 270;
const FPS: f64 = 10.0;
const RATE: u32 = 16_000;

/// `name` from the flag or environment, else from `PATH`, else from Homebrew's prefix.
pub fn find(given: Option<&Path>, name: &str) -> Option<PathBuf> {
    if let Some(p) = given {
        return Some(p.to_path_buf());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path).chain([PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")]).map(|d| d.join(name)).find(|p| p.is_file())
}

enum Msg {
    Frame(f64, Vec<u8>),
    Audio(f64, Vec<f32>),
}

/// Runs the adbreak detector over a media file, decoded by ffmpeg (pictures and sound in two processes).
pub fn detect(ffmpeg: &Path, media: &Path) -> Result<Vec<Span>> {
    let spawn = |args: &[&str]| {
        Command::new(ffmpeg)
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(media)
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("running {}", ffmpeg.display()))
    };
    let vf = format!("fps={FPS},scale={W}:{H},format=gray");
    let mut video = spawn(&["-an", "-sn", "-vf", &vf, "-f", "rawvideo", "-"])?;
    let rate = RATE.to_string();
    let mut audio = spawn(&["-vn", "-sn", "-ac", "1", "-ar", &rate, "-f", "f32le", "-"])?;
    let (tx, rx) = mpsc::sync_channel::<Msg>(64);
    let (mut vout, mut aout) = (video.stdout.take().unwrap(), audio.stdout.take().unwrap());
    let vtx = tx.clone();
    let vthread = std::thread::spawn(move || -> std::io::Result<()> {
        for i in 0u64.. {
            let mut buf = vec![0u8; W * H];
            if read_full(&mut vout, &mut buf)? < buf.len() {
                return Ok(());
            }
            if vtx.send(Msg::Frame(i as f64 / FPS, buf)).is_err() {
                return Ok(());
            }
        }
        Ok(())
    });
    let athread = std::thread::spawn(move || -> std::io::Result<()> {
        let chunk = RATE as usize / 10;
        let mut at = 0u64;
        loop {
            let mut buf = vec![0u8; chunk * 4];
            let n = read_full(&mut aout, &mut buf)? / 4;
            if n == 0 {
                return Ok(());
            }
            let samples = buf.as_chunks::<4>().0.iter().take(n).map(|b| f32::from_le_bytes(*b)).collect();
            if tx.send(Msg::Audio(at as f64 / RATE as f64, samples)).is_err() {
                return Ok(());
            }
            at += n as u64;
        }
    });
    let mut det = Detector::new(Config::default());
    for m in rx {
        match m {
            Msg::Frame(pts, luma) => det.push_frame(Frame { pts, width: W, height: H, luma: &luma }),
            Msg::Audio(pts, s) => det.push_audio(pts, RATE, &s),
        }
    }
    vthread.join().expect("video reader")?;
    athread.join().expect("audio reader")?;
    for (what, mut child) in [("video", video), ("audio", audio)] {
        if !child.wait()?.success() {
            bail!("ffmpeg ({what}) failed on {}", media.display());
        }
    }
    det.finish();
    Ok(det.breaks().into_iter().map(|b| Span::new(b.start, b.end)).collect())
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
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

/// Makes the dataset's copy of a recording: `height` lines, the original frame rate (break edges are a few frames
/// long), mono 16 kHz sound, as one MP4 that plays straight from a file. About 0.5 GB for three hours at 360p.
pub fn transcode(ffmpeg: &Path, input: &Path, output: &Path, height: u32) -> Result<()> {
    let tmp = output.with_extension("partial.mp4");
    let scale = format!("scale=-2:{height}");
    // Hardware HEVC at constant quality on macOS; software H.264 elsewhere.
    let video: &[&str] = if cfg!(target_os = "macos") {
        &["-c:v", "hevc_videotoolbox", "-q:v", "50", "-tag:v", "hvc1"]
    } else {
        &["-c:v", "libx264", "-crf", "28", "-preset", "faster"]
    };
    let hwaccel: &[&str] = if cfg!(target_os = "macos") { &["-hwaccel", "videotoolbox"] } else { &[] };
    let status = Command::new(ffmpeg)
        .args(["-nostdin", "-v", "error", "-stats", "-y"])
        .args(hwaccel)
        .arg("-i")
        .arg(input)
        .args(["-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn", "-vf", &scale])
        .args(video)
        .args(["-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k", "-movflags", "+faststart", "-f", "mp4"])
        .arg(&tmp)
        .status()
        .with_context(|| format!("running {}", ffmpeg.display()))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("ffmpeg couldn't transcode {}", input.display());
    }
    std::fs::rename(&tmp, output)?;
    Ok(())
}

/// The media's length in seconds, from ffprobe next to ffmpeg.
pub fn duration(ffmpeg: &Path, media: &Path) -> Result<f64> {
    let ffprobe = ffmpeg.with_file_name("ffprobe");
    let out = Command::new(&ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(media)
        .output()
        .with_context(|| format!("running {}", ffprobe.display()))?;
    String::from_utf8_lossy(&out.stdout).trim().parse().with_context(|| format!("no duration for {}", media.display()))
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
