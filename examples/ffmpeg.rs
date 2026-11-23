//! Find the breaks in a recording, decoding it with the ffmpeg command line (pictures to stdout, sound to a
//! temporary file; the sound is resampled against its timestamps, so gaps in it don't shift it against the
//! pictures). `cargo run --release --example ffmpeg -- <file or URL>`. Set ADBREAK_HWACCEL=videotoolbox (or
//! another ffmpeg hwaccel) to decode in hardware.

use adbreak::{Config, Detector, Frame};
use std::io::Read;
use std::process::{Command, Stdio};

const W: usize = 480;
const H: usize = 270;
const FPS: f64 = 10.0;
const RATE: u32 = 16_000;

fn main() -> std::io::Result<()> {
    let input = std::env::args().nth(1).expect("usage: ffmpeg <file or URL>");
    let audio_path = std::env::temp_dir().join(format!("adbreak-{}.f32", std::process::id()));
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    if let Ok(hw) = std::env::var("ADBREAK_HWACCEL") {
        cmd.args(["-hwaccel", &hw]);
    }
    cmd.args(["-i", &input, "-map", "0:v:0", "-vf"]).arg(format!("fps={FPS},scale={W}:{H},format=gray"));
    cmd.args(["-f", "rawvideo", "pipe:1", "-map", "0:a:0", "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", &RATE.to_string(), "-f", "f32le", "-y"]);
    cmd.arg(&audio_path).stdout(Stdio::piped());
    let mut child = cmd.spawn()?;
    let mut out = child.stdout.take().expect("piped");

    let mut d = Detector::new(Config::default());
    let mut buf = vec![0u8; W * H];
    let mut i = 0u64;
    while out.read_exact(&mut buf).is_ok() {
        d.push_frame(Frame { pts: i as f64 / FPS, width: W, height: H, luma: &buf });
        i += 1;
    }
    child.wait()?;
    let bytes = std::fs::read(&audio_path)?;
    let _ = std::fs::remove_file(&audio_path);
    let samples: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    d.push_audio(0.0, RATE, &samples);
    d.finish();

    let hms = |t: f64| format!("{}:{:02}:{:02}", t as u64 / 3600, t as u64 % 3600 / 60, t as u64 % 60);
    let breaks = d.breaks();
    let total: f64 = breaks.iter().map(|b| b.end - b.start).sum();
    println!("{} frames, {} edges, {} breaks, {:.1} min", i, d.edges().len(), breaks.len(), total / 60.0);
    if std::env::var("ADBREAK_DEBUG").is_ok() {
        println!("black: {:?}", d.black_stretches().iter().map(|(a, b)| format!("{}+{:.2}", hms(*a), b - a)).collect::<Vec<_>>());
        println!("silence: {:?}", d.silences().iter().map(|(a, b)| format!("{}+{:.2}", hms(*a), b - a)).collect::<Vec<_>>());
        println!("audio samples: {} ({:.0} s)", samples.len(), samples.len() as f64 / f64::from(RATE));
    }
    for b in breaks {
        println!("  {} - {}  ({:.1} min)", hms(b.start), hms(b.end), (b.end - b.start) / 60.0);
    }
    Ok(())
}
