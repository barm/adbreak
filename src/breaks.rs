use crate::Config;

/// A commercial break: from `start` to `end`, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Break {
    pub start: f64,
    pub end: f64,
}

/// Breaks from the edges (black-and-silent moments, sorted) and the logo presence per second.
pub(crate) fn find(cfg: &Config, edges: &[f64], logo: &[Option<bool>]) -> Vec<Break> {
    // Without a learned logo there is no telling programme from ads.
    if logo.iter().all(Option::is_none) {
        return Vec::new();
    }
    let n = logo.len();
    let present = |i: usize| logo.get(i).copied().flatten().unwrap_or(false);
    // Share of the seconds in [a, b) that show the logo (seconds without a frame don't count).
    let share = |a: f64, b: f64| {
        let (from, to) = ((a.max(0.0) as usize).min(n), (b.max(0.0) as usize).min(n));
        let seen: Vec<bool> = logo[from..to.max(from)].iter().flatten().copied().collect();
        let on = seen.iter().filter(|&&p| p).count();
        let seen = seen.len();
        if seen == 0 { 1.0 } else { on as f64 / seen as f64 }
    };
    // The first second at or after `t` from which the logo stays for `logo_back_secs`.
    let back = |t: f64| (t.max(0.0) as usize..n.saturating_sub(cfg.logo_back_secs)).find(|&i| (i..i + cfg.logo_back_secs).all(present));

    let mut out: Vec<Break> = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let start = edges[i];
        // An edge into a stretch without the logo starts a break.
        if share(start, start + 15.0) >= 0.3 {
            i += 1;
            continue;
        }
        // Still in the break at the end of what has been seen (a live recording): it runs to there for now.
        let end = match back(start) {
            Some(b) => edges.iter().copied().find(|e| (e - b as f64).abs() <= cfg.snap_secs).unwrap_or(b as f64),
            None => n as f64,
        };
        let len = end - start;
        if len >= cfg.min_break_secs && len <= cfg.max_break_secs {
            match out.last_mut() {
                Some(prev) if start - prev.end < 2.0 => prev.end = end,
                _ => out.push(Break { start, end }),
            }
        }
        // Skip the edges inside this break, and the one that ends it.
        while i < edges.len() && edges[i] < end - 1.0 {
            i += 1;
        }
        if i < edges.len() && edges[i] <= end + 1.0 && share(edges[i], edges[i] + 15.0) >= 0.3 {
            i += 1;
        }
    }
    out
}
