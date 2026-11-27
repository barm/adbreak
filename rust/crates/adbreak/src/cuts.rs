//! Shot cuts: where one picture is unlike the one before far more than pictures have been lately. A cross-fade
//! changes a little each frame and isn't one. Fed every decoded picture (any size; 64×36 grey is plenty).

/// Finds cuts in pictures pushed in presentation order.
pub struct CutDetector {
    prev: Option<Vec<u8>>,
    /// Recent frame-to-frame changes (mean absolute luma difference, 0–1), oldest first.
    recent: Vec<f32>,
    cuts: Vec<f64>,
}

/// A cut's change is at least this (a twelfth of the full luma range, averaged over the picture)…
const MIN_CHANGE: f32 = 0.08;
/// …and at least this many times the median of the last RECENT changes.
const OVER_RECENT: f32 = 4.0;
const RECENT: usize = 12;
/// Cuts closer than this (seconds) are one: the larger change wins.
const MIN_GAP: f64 = 0.25;

impl Default for CutDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CutDetector {
    pub fn new() -> Self {
        CutDetector { prev: None, recent: vec![], cuts: vec![] }
    }

    /// One picture's luma, shown at `pts` seconds. Pictures of another size than the first are skipped.
    pub fn push(&mut self, pts: f64, luma: &[u8]) {
        let Some(prev) = self.prev.as_ref().filter(|p| p.len() == luma.len()) else {
            if self.prev.is_none() {
                self.prev = Some(luma.to_vec());
            }
            return;
        };
        let change = prev.iter().zip(luma).map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs()).sum::<u32>() as f32 / (255.0 * luma.len() as f32);
        let mut sorted = self.recent.clone();
        sorted.sort_by(f32::total_cmp);
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        if change >= MIN_CHANGE && (self.recent.len() < 3 || change >= OVER_RECENT * median) {
            match self.cuts.last() {
                Some(&last) if pts - last < MIN_GAP => {}
                _ => self.cuts.push(pts),
            }
        }
        self.recent.push(change);
        if self.recent.len() > RECENT {
            self.recent.remove(0);
        }
        self.prev = Some(luma.to_vec());
    }

    /// The cuts so far: the times of the first picture of each new shot.
    pub fn cuts(&self) -> &[f64] {
        &self.cuts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(v: u8) -> Vec<u8> {
        vec![v; 64 * 36]
    }

    #[test]
    fn a_hard_cut_is_found_and_a_fade_is_not() {
        let mut d = CutDetector::new();
        let mut t = 0.0;
        let mut push = |d: &mut CutDetector, v: u8| {
            d.push(t, &frame(v));
            t += 1.0 / 30.0;
        };
        // A steady shot with a little noise, a hard cut, a steady shot, then a slow fade over a second.
        for i in 0..30 {
            push(&mut d, 60 + (i % 2) as u8);
        }
        for _ in 0..30 {
            push(&mut d, 180);
        }
        for i in 0..30 {
            push(&mut d, 180 - (i * 4) as u8);
        }
        assert_eq!(d.cuts().len(), 1, "{:?}", d.cuts());
        assert!((d.cuts()[0] - 1.0).abs() < 0.05, "{:?}", d.cuts());
    }
}
