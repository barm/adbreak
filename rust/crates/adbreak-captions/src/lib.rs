//! Commercial-break hints from closed captions.
//!
//! Checked against frames (November 2026, US network news and late-night shows): programmes are captioned
//! 81–93% of the time and breaks 42–49%, so long stretches without captions are a good hint of a break. The
//! caption *style* helps only on live programming: live shows caption roll-up (each cue keeps the previous one's
//! lines), ads pop-on (whole new cues), but pre-produced shows are pop-on too.
//!
//! ```
//! // Captioned for the first minute, then nothing until 2:30.
//! let vtt = "WEBVTT\n\n00:00.000 --> 01:00.000\nNEWS\n\n02:30.000 --> 03:20.000\nNEWS\n";
//! let secs = adbreak_captions::seconds(&adbreak_captions::parse_vtt(vtt), 200.0);
//! assert!(secs[30].captioned && !secs[100].captioned);
//! assert_eq!(adbreak_captions::uncaptioned(&secs, &Default::default()), vec![(60.0, 150.0)]);
//! ```

#![forbid(unsafe_code)]

/// One caption cue.
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub lines: Vec<String>,
}

/// Parses WebVTT cues (timings and text; settings, ids and styling are ignored).
pub fn parse_vtt(text: &str) -> Vec<Cue> {
    let mut cues = vec![];
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((a, b)) = line.split_once("-->") else { continue };
        let (Some(start), Some(end)) = (timestamp(a.trim()), timestamp(b.split_whitespace().next().unwrap_or(""))) else { continue };
        let mut text = vec![];
        while let Some(l) = lines.peek() {
            if l.trim().is_empty() {
                break;
            }
            text.push(lines.next().unwrap_or_default().trim().to_string());
        }
        cues.push(Cue { start, end, lines: text });
    }
    cues
}

/// `hh:mm:ss.mmm` or `mm:ss.mmm`.
fn timestamp(s: &str) -> Option<f64> {
    let mut secs = 0.0;
    for part in s.split(':') {
        secs = secs * 60.0 + part.parse::<f64>().ok()?;
    }
    Some(secs)
}

/// What the captions were doing during one second.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Second {
    /// A cue was on screen.
    pub captioned: bool,
    /// The cue on screen carried a line over from the one before (roll-up captioning, as live shows use).
    pub roll_up: bool,
}

/// One entry per second of a recording `duration` seconds long.
pub fn seconds(cues: &[Cue], duration: f64) -> Vec<Second> {
    let n = duration.max(0.0).ceil() as usize;
    let mut out = vec![Second::default(); n];
    let mut prev: &[String] = &[];
    for c in cues {
        let roll_up = c.lines.first().is_some_and(|first| prev.contains(first));
        let (a, b) = (c.start.max(0.0).floor() as usize, (c.end.max(0.0).ceil() as usize).min(n));
        for s in &mut out[a.min(n)..b] {
            s.captioned = true;
            s.roll_up |= roll_up;
        }
        prev = &c.lines;
    }
    out
}

/// What is on screen at one instant (see [`at_seconds`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OnScreen {
    /// A cue is showing (counting half a second after it ends, as cues are often cut slightly short).
    pub visible: bool,
    /// That cue kept a line from the cue before it (roll-up captioning).
    pub roll_up: bool,
    /// That cue started within the last second.
    pub new_cue: bool,
}

/// What is on screen at each whole second 0, 1, … n-1: the latest cue started at or before that instant. These are
/// the caption inputs adbreak's model is trained on; keep them as they are, or retrain.
pub fn at_seconds(cues: &[Cue], n: usize) -> Vec<OnScreen> {
    let roll: Vec<bool> = cues.iter().enumerate().map(|(i, c)| i > 0 && c.lines.first().is_some_and(|f| cues[i - 1].lines.contains(f))).collect();
    let mut out = vec![OnScreen::default(); n];
    let mut i = 0usize;
    let mut cur: Option<usize> = None;
    for (s, o) in out.iter_mut().enumerate() {
        let t = s as f64;
        while i < cues.len() && cues[i].start <= t {
            cur = Some(i);
            i += 1;
        }
        if let Some(c) = cur
            && t < cues[c].end + 0.5
        {
            *o = OnScreen { visible: true, roll_up: roll[c], new_cue: cues[c].start > t - 1.0 };
        }
    }
    out
}

/// Settings for [`uncaptioned`].
#[derive(Clone, Debug)]
pub struct Quiet {
    /// Window (seconds) over which coverage is averaged.
    pub window: usize,
    /// A second is quiet when the window around it is captioned less than this share of the time.
    pub max_coverage: f64,
    /// Quiet stretches shorter than this (seconds) are dropped.
    pub min_secs: f64,
}

impl Default for Quiet {
    fn default() -> Self {
        Quiet { window: 30, max_coverage: 0.5, min_secs: 30.0 }
    }
}

/// Stretches (start, end seconds) where captions are mostly absent: break candidates.
pub fn uncaptioned(secs: &[Second], q: &Quiet) -> Vec<(f64, f64)> {
    let n = secs.len();
    let half = q.window / 2;
    let mut sum = vec![0usize; n + 1];
    for (i, s) in secs.iter().enumerate() {
        sum[i + 1] = sum[i] + s.captioned as usize;
    }
    let quiet = |i: usize| {
        let (a, b) = (i.saturating_sub(half), (i + half + 1).min(n));
        ((sum[b] - sum[a]) as f64) < q.max_coverage * (b - a) as f64
    };
    let mut out = vec![];
    let mut start = None;
    for i in 0..=n {
        match (i < n && quiet(i), start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                // Trim to the captions themselves: the stretch runs from the last cue before to the first after.
                let a = (a..i).find(|&j| !secs[j].captioned).unwrap_or(a);
                let b = (a..i).rev().find(|&j| !secs[j].captioned).map_or(i, |j| j + 1);
                if (b - a) as f64 >= q.min_secs {
                    out.push((a as f64, b as f64));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_detects_roll_up() {
        let vtt = "WEBVTT\n\n00:00.651 --> 00:02.953\nnow and six feet\n\n00:02.953 --> 00:05.722\nnow and six feet\nAnd that means\n\n1:00:00.000 --> 1:00:01.500 line:90%\nPOP ON\n";
        let cues = parse_vtt(vtt);
        assert_eq!(cues.len(), 3);
        assert_eq!(cues[2].start, 3600.0);
        assert_eq!(cues[1].lines, vec!["now and six feet", "And that means"]);
        let s = seconds(&cues, 3602.0);
        assert!(s[0].captioned && !s[0].roll_up && s[4].roll_up);
        assert!(s[3600].captioned && !s[3600].roll_up);
        assert!(!s[100].captioned);
    }

    #[test]
    fn on_screen_each_second() {
        let cues = vec![
            Cue { start: 0.5, end: 2.0, lines: vec!["a".into()] },
            Cue { start: 2.0, end: 3.2, lines: vec!["a".into(), "b".into()] },
            Cue { start: 6.0, end: 7.0, lines: vec!["c".into()] },
        ];
        let o = at_seconds(&cues, 9);
        assert!(!o[0].visible);
        assert!(o[1].visible && o[1].new_cue && !o[1].roll_up);
        assert!(o[2].visible && o[2].roll_up && o[2].new_cue);
        assert!(o[3].visible && !o[3].new_cue, "{:?}", o[3]);
        assert!(!o[4].visible && !o[5].visible);
        assert!(o[6].visible && !o[6].roll_up && o[7].visible && !o[8].visible);
    }

    #[test]
    fn quiet_stretch_with_a_few_captions_inside() {
        // 0–100 captioned, 100–200 an ad break with two captioned ads, 200–300 captioned.
        let mut cues = vec![Cue { start: 0.0, end: 100.0, lines: vec!["a".into()] }, Cue { start: 200.0, end: 300.0, lines: vec!["b".into()] }];
        cues.push(Cue { start: 120.0, end: 130.0, lines: vec!["ad".into()] });
        cues.push(Cue { start: 160.0, end: 165.0, lines: vec!["ad".into()] });
        let q = uncaptioned(&seconds(&cues, 300.0), &Quiet::default());
        assert_eq!(q, vec![(100.0, 200.0)]);
    }
}
