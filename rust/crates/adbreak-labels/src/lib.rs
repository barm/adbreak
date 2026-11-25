//! Labelled commercial breaks: the dataset format, suggestions to review, and scoring a detector against labels.
//!
//! A recording's labels say where its breaks are, as checked by a person, and which parts of it were checked
//! (`reviewed`): scoring only counts those, so a recording can be labelled a bit at a time. Suggestions are what
//! the automatic sources (Comskip, captions, a detector) said about the same recording, kept beside the labels to
//! review against. Both are JSON; times are seconds from the start of the recording's media.
//!
//! ```
//! use adbreak_labels::{Break, Kind, Labels, Recording, Span, score};
//! let mut labels = Labels::new(Recording { duration: 600.0, ..Default::default() });
//! labels.reviewed = vec![Span::new(0.0, 600.0)];
//! labels.breaks = vec![Break { start: 100.0, end: 220.0, kind: Kind::Ad }];
//! let s = score(&labels, &[Span::new(110.0, 220.0)]);
//! assert_eq!(s.matched, 1);
//! assert!(s.recall() > 0.9 && s.precision() == 1.0);
//! ```

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// The version written into every file; bumped when the format changes incompatibly.
pub const FORMAT: u32 = 1;

/// A stretch of a recording, in seconds from the start of its media.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

impl Span {
    pub fn new(start: f64, end: f64) -> Self {
        Span { start, end }
    }
    pub fn len(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0.0
    }
    pub fn contains(&self, t: f64) -> bool {
        self.start <= t && t < self.end
    }
    /// Seconds shared with `other`.
    pub fn overlap(&self, other: &Span) -> f64 {
        (self.end.min(other.end) - self.start.max(other.start)).max(0.0)
    }
}

/// What a labelled span is. Everything unlabelled in reviewed time is programme. The kinds say what is on screen,
/// not what to do about it: whether to skip each is up to the player. See `docs/labels.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A commercial break: the programme is gone and advertising fills the screen (commercials, the network's
    /// promos for its own shows, station spots and bumpers). What a detector is scored on.
    #[default]
    Ad,
    /// The programme shrunk beside or under an ad (a squeeze-back). Not scored as break or programme, and a break's
    /// edge anywhere across an adjacent double box counts as right.
    DoubleBox,
    /// Advertising over the full-screen programme: an announcer's sponsor read, sponsor graphics or banners over
    /// play, a promo for what's coming up next. Programme as far as break scoring goes.
    AdRead,
    /// A whole programme that is an advertisement (an infomercial). Not scored either way.
    PaidProgramming,
}

impl Kind {
    /// Every kind, in the order the labelling page offers them.
    pub const ALL: [Kind; 4] = [Kind::Ad, Kind::DoubleBox, Kind::AdRead, Kind::PaidProgramming];

    /// Whether it is a commercial break, which a detector should find.
    pub fn scored(self) -> bool {
        self == Kind::Ad
    }

    /// Whether break scoring leaves it out, counting it as neither break nor programme.
    pub fn ignored(self) -> bool {
        matches!(self, Kind::DoubleBox | Kind::PaidProgramming)
    }
}

/// A labelled break.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Break {
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub kind: Kind,
}

impl From<Span> for Break {
    /// A commercial break over `s`.
    fn from(s: Span) -> Self {
        Break { start: s.start, end: s.end, kind: Kind::Ad }
    }
}

impl Source {
    /// What it found of one kind.
    pub fn spans(&self, kind: Kind) -> Vec<Span> {
        self.breaks.iter().filter(|b| b.kind == kind).map(Break::span).collect()
    }

    /// Whether it finds kinds besides commercial breaks.
    pub fn kinds(&self) -> Vec<Kind> {
        Kind::ALL.into_iter().filter(|&k| k != Kind::Ad && self.breaks.iter().any(|b| b.kind == k)).collect()
    }
}

impl Break {
    pub fn span(&self) -> Span {
        Span::new(self.start, self.end)
    }
}

/// Which recording a file is about.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Recording {
    /// Where it came from, e.g. `"recorder"` (with `job` the recorder's id for it).
    pub source: String,
    pub job: i64,
    /// Virtual channel, e.g. `"2.1"`.
    pub channel: String,
    #[serde(default)]
    pub network: String,
    pub title: String,
    /// When recording started, RFC 3339.
    pub start: String,
    /// Length of the media, seconds.
    pub duration: f64,
}

impl Recording {
    /// A name for its directory in the dataset: date, channel and job, e.g. `2026-11-23_5.1_33`.
    pub fn id(&self) -> String {
        let date = self.start.get(..10).unwrap_or("undated");
        let safe = |s: &str| s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '-' }).collect::<String>();
        format!("{}_{}_{}", safe(date), safe(&self.channel), self.job)
    }
}

/// A person's labels for one recording.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    pub format: u32,
    pub recording: Recording,
    /// The parts checked completely: every break in them is listed. Scoring is limited to these.
    pub reviewed: Vec<Span>,
    pub breaks: Vec<Break>,
    /// Parts left out entirely (bad reception, say): never scored or trained on, and not counted as left to review.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded: Vec<Span>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

impl Labels {
    pub fn new(recording: Recording) -> Self {
        Labels { format: FORMAT, recording, reviewed: vec![], breaks: vec![], excluded: vec![], notes: String::new() }
    }

    /// The reviewed parts, less any excluded: what is scored and trained on.
    pub fn usable_reviewed(&self) -> Vec<Span> {
        let mut out = merge(&self.reviewed);
        for x in &self.excluded {
            out = out
                .into_iter()
                .flat_map(|r| [Span::new(r.start, r.end.min(x.start)), Span::new(r.start.max(x.end), r.end)])
                .filter(|r| r.end > r.start)
                .collect();
        }
        out
    }

    /// Seconds of the recording not excluded: what there is to review.
    pub fn usable_seconds(&self) -> f64 {
        (self.recording.duration - merge(&self.excluded).iter().map(Span::len).sum::<f64>()).max(0.0)
    }

    /// Sorts breaks and merges reviewed spans; errors on a break that is empty, outside the media or overlapping
    /// another.
    pub fn normalise(&mut self) -> Result<(), String> {
        let d = self.recording.duration;
        self.breaks.sort_by(|a, b| a.start.total_cmp(&b.start));
        for b in &self.breaks {
            if !b.start.is_finite() || !b.end.is_finite() || b.end <= b.start || b.start < 0.0 || (d > 0.0 && b.end > d + 1.0) {
                return Err(format!("break {:.1}–{:.1} s is empty or outside the recording", b.start, b.end));
            }
        }
        for w in self.breaks.windows(2) {
            if w[1].start < w[0].end {
                return Err(format!("breaks at {:.1} s and {:.1} s overlap", w[0].start, w[1].start));
            }
        }
        self.reviewed = merge(&self.reviewed);
        Ok(())
    }

    /// Whether all of it has been reviewed.
    pub fn complete(&self) -> bool {
        let d = self.recording.duration;
        d > 0.0 && self.reviewed.iter().map(Span::len).sum::<f64>() >= d - 1.0
    }
}

/// Sorted, with overlapping and touching spans joined.
pub fn merge(spans: &[Span]) -> Vec<Span> {
    let mut v: Vec<Span> = spans.iter().copied().filter(|s| !s.is_empty()).collect();
    v.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<Span> = vec![];
    for s in v {
        match out.last_mut() {
            Some(l) if s.start <= l.end => l.end = l.end.max(s.end),
            _ => out.push(s),
        }
    }
    out
}

/// What the automatic sources said about a recording.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestions {
    pub format: u32,
    pub recording: Recording,
    pub sources: Vec<Source>,
}

/// One source's opinion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// e.g. `"comskip"`, `"captions"`, `"logo-detector"`, `"model"`.
    pub name: String,
    /// Version or settings, so results from different runs can be told apart.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
    /// What it found. A detector that tells the kinds apart marks each; others find commercial breaks (`ad`).
    pub breaks: Vec<Break>,
    /// For a learned source: the labels it was trained against (`Dataset` fingerprint). Empty for rule-based ones.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub trained_on: String,
}

/// How a detector did against labels, within the reviewed parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    /// Seconds that were scored (reviewed, and not paid programming).
    pub seconds: f64,
    /// Seconds of break found, of programme marked as break, and of break missed.
    pub true_pos: f64,
    pub false_pos: f64,
    pub false_neg: f64,
    /// Labelled breaks, breaks found, and labelled breaks matched by a found one (overlapping at least half of it).
    pub breaks: usize,
    pub found: usize,
    pub matched: usize,
    /// Over matched breaks: total distance (seconds) between the labelled and found start, and end.
    pub start_error: f64,
    pub end_error: f64,
    /// Reviewed seconds labelled double box, and how many of them were marked as break.
    pub double_box_seconds: f64,
    pub double_box_marked: f64,
    /// Reviewed seconds labelled ad read, and how many of them were marked as break (counted as false positives too).
    pub ad_read_seconds: f64,
    pub ad_read_marked: f64,
}

impl Score {
    pub fn precision(&self) -> f64 {
        ratio(self.true_pos, self.true_pos + self.false_pos)
    }
    pub fn recall(&self) -> f64 {
        ratio(self.true_pos, self.true_pos + self.false_neg)
    }
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        ratio(2.0 * p * r, p + r)
    }
    /// Mean distance of matched breaks' starts and ends, seconds.
    pub fn mean_start_error(&self) -> f64 {
        ratio(self.start_error, self.matched as f64)
    }
    pub fn mean_end_error(&self) -> f64 {
        ratio(self.end_error, self.matched as f64)
    }
    /// Seconds of programme a viewer would lose to skipping, per hour of scored time (programme, ad reads included,
    /// marked as break).
    pub fn wrongly_skipped_per_hour(&self) -> f64 {
        3600.0 * ratio(self.false_pos, self.seconds)
    }
    /// Seconds of commercial break a viewer would still sit through, per hour of scored time.
    pub fn missed_per_hour(&self) -> f64 {
        3600.0 * ratio(self.false_neg, self.seconds)
    }
    /// Totals over several recordings.
    pub fn add(&mut self, o: &Score) {
        self.seconds += o.seconds;
        self.true_pos += o.true_pos;
        self.false_pos += o.false_pos;
        self.false_neg += o.false_neg;
        self.breaks += o.breaks;
        self.found += o.found;
        self.matched += o.matched;
        self.start_error += o.start_error;
        self.end_error += o.end_error;
        self.double_box_seconds += o.double_box_seconds;
        self.double_box_marked += o.double_box_marked;
        self.ad_read_seconds += o.ad_read_seconds;
        self.ad_read_marked += o.ad_read_marked;
    }
}

fn ratio(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}

/// From `edge`, the range covered by a chain of spans touching it (within a second), going earlier (`dir` < 0) or
/// later.
fn stretch(edge: f64, spans: &[Span], dir: f64) -> (f64, f64) {
    let mut far = edge;
    loop {
        let next = spans.iter().find(|s| if dir < 0.0 { (s.end - far).abs() <= 1.0 && s.start < far } else { (s.start - far).abs() <= 1.0 && s.end > far });
        match next {
            Some(s) => far = if dir < 0.0 { s.start } else { s.end },
            None => break,
        }
    }
    if dir < 0.0 { (far, edge) } else { (edge, far) }
}

/// Distance from `x` to the range `lo..=hi`.
fn outside(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo - x
    } else if x > hi {
        x - hi
    } else {
        0.0
    }
}

/// Scores `found` breaks against `labels`: only the reviewed parts, leaving out double box and paid programming.
pub fn score(labels: &Labels, found: &[Span]) -> Score {
    score_kind(labels, Kind::Ad, found)
}

/// Scores `found` spans of one kind against the labels of that kind, within the reviewed parts. For commercial
/// breaks ([`score`]) double box and paid programming are left out, and an edge anywhere across an adjacent double
/// box counts as right; for the other kinds only paid programming is left out.
pub fn score_kind(labels: &Labels, kind: Kind, found: &[Span]) -> Score {
    const STEP: f64 = 0.1;
    let reviewed = labels.usable_reviewed();
    let of = |k: Kind| -> Vec<Span> { labels.breaks.iter().filter(|b| b.kind == k).map(Break::span).collect() };
    let truth = of(kind);
    let ignored: Vec<Span> =
        labels.breaks.iter().filter(|b| if kind == Kind::Ad { b.kind.ignored() } else { b.kind == Kind::PaidProgramming }).map(Break::span).collect();
    let (double_box, ad_read) = (of(Kind::DoubleBox), of(Kind::AdRead));
    let tolerance = if kind == Kind::Ad { double_box.clone() } else { vec![] };
    let mut s = Score::default();
    let within = |t: f64, v: &[Span]| v.iter().any(|x| x.contains(t));
    for r in &reviewed {
        let mut t = r.start + STEP / 2.0;
        while t < r.end {
            if within(t, &double_box) {
                s.double_box_seconds += STEP;
                if within(t, found) {
                    s.double_box_marked += STEP;
                }
            }
            if within(t, &ad_read) {
                s.ad_read_seconds += STEP;
                if within(t, found) {
                    s.ad_read_marked += STEP;
                }
            }
            if !within(t, &ignored) {
                s.seconds += STEP;
                match (within(t, &truth), within(t, found)) {
                    (true, true) => s.true_pos += STEP,
                    (false, true) => s.false_pos += STEP,
                    (true, false) => s.false_neg += STEP,
                    (false, false) => {}
                }
            }
            t += STEP;
        }
    }
    let fully_reviewed = |x: &Span| reviewed.iter().any(|r| r.start <= x.start && x.end <= r.end);
    let truth: Vec<Span> = truth.into_iter().filter(fully_reviewed).collect();
    let found: Vec<Span> = found.iter().copied().filter(|f| reviewed.iter().any(|r| r.overlap(f) > 0.0)).collect();
    s.breaks = truth.len();
    s.found = found.len();
    // Greedy by overlap: each found break matches at most one labelled break.
    let mut pairs: Vec<(f64, usize, usize)> = vec![];
    for (i, t) in truth.iter().enumerate() {
        for (j, f) in found.iter().enumerate() {
            let o = t.overlap(f);
            if o >= t.len() / 2.0 {
                pairs.push((o, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut ti, mut fj) = (vec![false; truth.len()], vec![false; found.len()]);
    for (_, i, j) in pairs {
        if !ti[i] && !fj[j] {
            ti[i] = true;
            fj[j] = true;
            s.matched += 1;
            // An edge anywhere across a double box next to it is right: it is neither break nor programme.
            let (lo, _) = stretch(truth[i].start, &tolerance, -1.0);
            let (_, hi) = stretch(truth[i].end, &tolerance, 1.0);
            s.start_error += outside(found[j].start, lo, truth[i].start);
            s.end_error += outside(found[j].end, truth[i].end, hi);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(breaks: &[(f64, f64, Kind)]) -> Labels {
        let mut l = Labels::new(Recording {
            source: "recorder".into(),
            job: 7,
            channel: "2.1".into(),
            start: "2026-11-23T07:06:41Z".into(),
            duration: 1000.0,
            ..Default::default()
        });
        l.reviewed = vec![Span::new(0.0, 1000.0)];
        l.breaks = breaks.iter().map(|&(start, end, kind)| Break { start, end, kind }).collect();
        l
    }

    #[test]
    fn perfect_and_empty() {
        let l = labels(&[(100.0, 200.0, Kind::Ad), (500.0, 620.0, Kind::Ad)]);
        let s = score(&l, &[Span::new(100.0, 200.0), Span::new(500.0, 620.0)]);
        assert_eq!((s.breaks, s.found, s.matched), (2, 2, 2));
        assert!((s.recall() - 1.0).abs() < 1e-6 && (s.precision() - 1.0).abs() < 1e-6);
        assert!(s.mean_start_error() < 1e-6);
        let s = score(&l, &[]);
        assert_eq!(s.matched, 0);
        assert_eq!(s.recall(), 0.0);
        assert!((s.false_neg - 220.0).abs() < 0.5);
    }

    #[test]
    fn short_and_extra() {
        // The CBS case: a 26 s tail of a 200 s break is found, plus a false 50 s break.
        let l = labels(&[(100.0, 300.0, Kind::Ad)]);
        let s = score(&l, &[Span::new(274.0, 300.0), Span::new(700.0, 750.0)]);
        assert_eq!((s.found, s.matched), (2, 0));
        assert!((s.true_pos - 26.0).abs() < 0.5 && (s.false_pos - 50.0).abs() < 0.5 && (s.false_neg - 174.0).abs() < 0.5);
    }

    #[test]
    fn paid_programming_and_unreviewed_are_not_scored() {
        let mut l = labels(&[(100.0, 200.0, Kind::Ad), (400.0, 600.0, Kind::PaidProgramming)]);
        l.reviewed = vec![Span::new(0.0, 700.0)];
        let s = score(&l, &[Span::new(100.0, 200.0), Span::new(400.0, 600.0), Span::new(800.0, 900.0)]);
        assert_eq!((s.found, s.matched), (2, 1));
        assert_eq!(s.false_pos, 0.0);
        assert!((s.seconds - 500.0).abs() < 0.5);
    }

    #[test]
    fn double_box_tails_are_fair_either_way() {
        // Ads 100–200, then a double box 200–230 before the game resumes full screen.
        let l = labels(&[(100.0, 200.0, Kind::Ad), (200.0, 230.0, Kind::DoubleBox)]);
        for end in [200.0, 215.0, 230.0] {
            let s = score(&l, &[Span::new(100.0, end)]);
            assert_eq!(s.end_error, 0.0, "ending anywhere across the double box is right ({end})");
            assert!(s.false_pos.abs() < 0.5, "double box is not programme ({end})");
            assert!((s.double_box_seconds - 30.0).abs() < 0.5 && (s.double_box_marked - (end - 200.0)).abs() < 0.5);
        }
        let s = score(&l, &[Span::new(100.0, 250.0)]);
        assert!((s.end_error - 20.0).abs() < 1e-9 && (s.false_pos - 20.0).abs() < 0.5, "past the double box is wrong");
    }

    #[test]
    fn ad_reads_are_programme() {
        // Ads 100–200, then the announcer's ad read 200–220 over the full-screen game.
        let l = labels(&[(100.0, 200.0, Kind::Ad), (200.0, 220.0, Kind::AdRead)]);
        let s = score(&l, &[Span::new(100.0, 220.0)]);
        assert!((s.end_error - 20.0).abs() < 1e-9 && (s.false_pos - 20.0).abs() < 0.5, "an ad read is not break");
        assert!((s.ad_read_seconds - 20.0).abs() < 0.5 && (s.ad_read_marked - 20.0).abs() < 0.5);
    }

    #[test]
    fn kind_names() {
        let k: Vec<Kind> = serde_json::from_str(r#"["ad", "double_box", "ad_read", "paid_programming"]"#).unwrap();
        assert_eq!(k, Kind::ALL);
        assert!(serde_json::from_str::<Kind>(r#""sponsor""#).is_err(), "old names are gone");
    }

    #[test]
    fn normalise_and_format() {
        let mut l = labels(&[(500.0, 600.0, Kind::Ad), (100.0, 200.0, Kind::Ad)]);
        l.reviewed = vec![Span::new(300.0, 1000.0), Span::new(0.0, 300.0)];
        l.normalise().unwrap();
        assert_eq!(l.breaks[0].start, 100.0);
        assert_eq!(l.reviewed, vec![Span::new(0.0, 1000.0)]);
        assert!(l.complete());
        assert_eq!(l.recording.id(), "2026-11-23_2.1_7");
        let json = serde_json::to_string(&l).unwrap();
        assert!(json.contains(r#""kind":"ad""#));
        assert_eq!(serde_json::from_str::<Labels>(&json).unwrap(), l);
        l.breaks.push(Break { start: 150.0, end: 160.0, kind: Kind::Ad });
        assert!(l.normalise().is_err());
    }
}
