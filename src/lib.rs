//! Find commercial breaks in TV recordings.
//!
//! Over-the-air TV carries no markers for its ad breaks, so they have to be found in the picture and sound. This
//! crate does that from decoded data only, with no dependencies: feed it frames (8-bit luma, any size) and mono
//! audio samples with their timestamps, in any order and as they arrive, and ask for the breaks at any time.
//! Decoding is up to you (a platform decoder, a software one, …).
//!
//! How it decides, tuned on US network broadcasts:
//! - A break **starts** at an *edge*: a moment that is both black and silent, which networks put around breaks
//!   (and, at the thresholds used, not between the ads inside one).
//! - Which side of an edge is the ad comes from the **network logo**: the corner bug shown during the programme
//!   and not during ads. It is learned from the recording itself (the corner whose outline recurs the most), so no
//!   template is needed per channel.
//! - A break **ends** when the logo is back for good (`logo_back_secs`), snapped to an edge if one is close. That
//!   also ends breaks that come back through a split-screen ad, with no black frame.
//!
//! ```
//! use adbreak::{Config, Detector, Frame};
//! let mut d = Detector::new(Config::default());
//! let luma = vec![16u8; 480 * 270];
//! d.push_frame(Frame { pts: 0.0, width: 480, height: 270, luma: &luma });
//! d.push_audio(0.0, 16_000, &[0.0; 1600]);
//! d.finish();
//! assert!(d.breaks().is_empty());
//! ```

mod black;
mod breaks;
mod logo;
mod silence;

pub use breaks::Break;

/// Tuning. The defaults suit 8-bit limited-range ("TV range", 16–235) video and normal broadcast audio.
#[derive(Clone, Debug)]
pub struct Config {
    /// A pixel at or below this luma counts as black (ffmpeg `blackdetect`'s `pix_th=0.10` on limited range).
    pub black_luma: u8,
    /// A frame is black when at least this share of its pixels is.
    pub black_ratio: f32,
    /// Shortest black stretch that counts, in seconds.
    pub min_black_secs: f64,
    /// Audio at or below this level (dBFS) is silence.
    pub silence_db: f32,
    /// Shortest silence that counts, in seconds.
    pub min_silence_secs: f64,
    /// A black stretch and a silence this close (seconds) make an edge.
    pub edge_slack_secs: f64,
    /// Breaks shorter or longer than these (seconds) are ignored. The maximum allows for halftime.
    pub min_break_secs: f64,
    pub max_break_secs: f64,
    /// The logo has to be back for this long (seconds) to end a break.
    pub logo_back_secs: usize,
    /// A break's end snaps to an edge this close (seconds) to where the logo came back.
    pub snap_secs: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            black_luma: 38,
            black_ratio: 0.98,
            min_black_secs: 0.1,
            silence_db: -50.0,
            min_silence_secs: 0.3,
            edge_slack_secs: 0.5,
            min_break_secs: 15.0,
            max_break_secs: 25.0 * 60.0,
            logo_back_secs: 10,
            snap_secs: 5.0,
        }
    }
}

/// One decoded picture: its 8-bit luma (Y) plane, `width` × `height`, rows packed (no padding), and when it is
/// shown in seconds. Any size works; 480×270 is plenty. Frames need not be every frame: 10 per second is enough.
pub struct Frame<'a> {
    pub pts: f64,
    pub width: usize,
    pub height: usize,
    pub luma: &'a [u8],
}

/// Collects the signals (black stretches, silences, logo presence per second) and turns them into breaks.
pub struct Detector {
    cfg: Config,
    black: black::Black,
    silence: silence::Silence,
    logo: logo::Logo,
}

impl Detector {
    pub fn new(cfg: Config) -> Self {
        Self { black: black::Black::default(), silence: silence::Silence::default(), logo: logo::Logo::default(), cfg }
    }

    /// Add a picture. Frames of one stream should come in presentation order.
    pub fn push_frame(&mut self, f: Frame<'_>) {
        assert!(f.luma.len() >= f.width * f.height, "luma plane smaller than width × height");
        self.black.push(&self.cfg, f.pts, &f.luma[..f.width * f.height]);
        self.logo.push(f.pts, f.width, f.height, f.luma);
    }

    /// Add mono audio samples (−1.0..=1.0) starting at `pts` seconds, at `rate` samples per second. Keep the sound
    /// on the same clock as the pictures: broadcast audio has small gaps, and samples written back to back drift
    /// (with ffmpeg, `aresample=async=1:first_pts=0` fills them). Edges need picture and sound within
    /// `edge_slack_secs` of each other.
    pub fn push_audio(&mut self, pts: f64, rate: u32, samples: &[f32]) {
        self.silence.push(&self.cfg, pts, rate, samples);
    }

    /// Close stretches still open at the end of the input (a black or silent ending). Pushing more afterwards is
    /// fine: a stretch that continues simply starts again.
    pub fn finish(&mut self) {
        self.black.finish(&self.cfg);
        self.silence.finish(&self.cfg);
    }

    /// The breaks found so far, in order. Cheap enough to call after every chunk of a live recording; results can
    /// change as more arrives (the logo is learned from everything seen).
    pub fn breaks(&self) -> Vec<Break> {
        breaks::find(&self.cfg, &self.edges(), &self.logo.presence())
    }

    /// Black-and-silent edges (seconds), the candidate break boundaries.
    pub fn edges(&self) -> Vec<f64> {
        let s = self.cfg.edge_slack_secs;
        let mut e: Vec<f64> =
            self.black.runs.iter().filter(|(a, b)| self.silence.runs.iter().any(|(sa, sb)| *sa <= b + s && *sb >= a - s)).map(|(a, b)| (a + b) / 2.0).collect();
        e.sort_by(f64::total_cmp);
        e
    }

    /// Black stretches found, (start, end) in seconds.
    pub fn black_stretches(&self) -> &[(f64, f64)] {
        &self.black.runs
    }

    /// Silences found, (start, end) in seconds.
    pub fn silences(&self) -> &[(f64, f64)] {
        &self.silence.runs
    }

    /// Whether the logo was on screen, per whole second from 0 (None: no frame that second).
    pub fn logo_presence(&self) -> Vec<Option<bool>> {
        self.logo.presence()
    }
}
