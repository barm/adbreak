//! Caption text as model input, computed as the training code computes it (`ml/adbreak_ml/text.py`): the words on
//! screen, hashed, and two learned scores per second, one for what is on screen ("now": the caption belongs to a
//! commercial) and one for what was said in the last `trail` seconds ("trail": a break is coming).
//!
//! Tokens: lowercased (ASCII) runs of letters, digits and apostrophes, single words and adjacent pairs, FNV-1a 32 of
//! their bytes modulo the number of buckets; present or not.

use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub(crate) struct Head {
    pub coef: Vec<f32>,
    pub intercept: f64,
}

#[derive(Deserialize)]
pub(crate) struct Text {
    pub buckets: usize,
    pub trail: usize,
    pub now: Option<Head>,
    pub trail_head: Option<Head>,
}

fn fnv32(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    h
}

/// The token buckets of a caption's text.
pub(crate) fn tokens(text: &str, buckets: usize) -> BTreeSet<usize> {
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'')).filter(|w| !w.is_empty()).collect();
    let mut out: BTreeSet<usize> = words.iter().map(|w| fnv32(w) as usize % buckets).collect();
    for p in words.windows(2) {
        out.insert(fnv32(&format!("{} {}", p[0], p[1])) as usize % buckets);
    }
    out
}

/// Per second 0..n: the tokens on screen during it, and those of cues that started in the last `trail` seconds.
pub(crate) fn per_second(cues: &[(f64, f64, String)], n: usize, buckets: usize, trail: usize) -> (Vec<BTreeSet<usize>>, Vec<BTreeSet<usize>>) {
    let mut now = vec![BTreeSet::new(); n];
    let mut started = vec![BTreeSet::new(); n];
    for (a, b, text) in cues {
        let tk = tokens(text, buckets);
        let (lo, hi) = ((a.floor().max(0.0)) as usize, (b.ceil().max(0.0) as usize).min(n));
        for s in now.iter_mut().take(hi).skip(lo) {
            s.extend(&tk);
        }
        if *a >= 0.0 && (a.floor() as usize) < n {
            started[a.floor() as usize].extend(&tk);
        }
    }
    let trail_sets = (0..n)
        .map(|t| {
            let mut s = BTreeSet::new();
            for set in &started[t.saturating_sub(trail - 1)..=t] {
                s.extend(set);
            }
            s
        })
        .collect();
    (now, trail_sets)
}

/// The score of each second's tokens: the weights of those present, in bucket order, plus the intercept, through the
/// logistic; 0 where there are none (no evidence either way).
pub(crate) fn score(sets: &[BTreeSet<usize>], head: Option<&Head>) -> Vec<f64> {
    sets.iter()
        .map(|s| match head {
            Some(h) if !s.is_empty() => {
                let mut z = 0.0;
                for &j in s {
                    z += f64::from(h.coef[j]);
                }
                1.0 / (1.0 + (-(z + h.intercept)).exp())
            }
            _ => 0.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_words_and_pairs_lowercased() {
        let t = tokens(">> WE'LL be right Back!", 1 << 14);
        let want: BTreeSet<usize> =
            ["we'll", "be", "right", "back", "we'll be", "be right", "right back"].iter().map(|w| fnv32(w) as usize % (1 << 14)).collect();
        assert_eq!(t, want);
        // FNV-1a 32 of "a", as Python's text.fnv32 gives it.
        assert_eq!(fnv32("a"), 0xe40c292c);
    }

    #[test]
    fn trailing_tokens_cover_the_last_seconds() {
        let cues = vec![(2.4, 4.0, "stay with us".to_string())];
        let (now, trail) = per_second(&cues, 30, 1 << 14, 20);
        assert!(now[1].is_empty() && !now[2].is_empty() && !now[3].is_empty() && now[4].is_empty());
        assert!(trail[1].is_empty() && !trail[2].is_empty() && !trail[21].is_empty() && trail[22].is_empty());
    }
}
