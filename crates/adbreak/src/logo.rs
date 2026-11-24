//! The network logo: once a second, an edge map of each corner. The logo is the corner outline that recurs in most
//! seconds; a second "has the logo" when most of that outline is there.

/// Corner box, as fractions of the picture: 10 % wide, 6 % high, inset 1.5 % from the side and 2.5 % from the edge.
const BOX_W: f64 = 0.10;
const BOX_H: f64 = 0.06;
const INSET_X: f64 = 0.015;
const INSET_Y: f64 = 0.025;
/// The box is averaged down to this grid before looking for edges.
const GW: usize = 64;
const GH: usize = 24;
/// Neighbouring cells differing by more than this (luma) are an edge.
const EDGE: i32 = 40;
const BITS: usize = (GW - 1) * (GH - 1);
const WORDS: usize = BITS.div_ceil(64);
type Map = [u64; WORDS];

#[derive(Default)]
pub(crate) struct Logo {
    /// Per second (index = whole second), the edge map of each corner: top-left, top-right, bottom-left, bottom-right.
    seconds: Vec<Option<[Map; 4]>>,
}

impl Logo {
    pub fn push(&mut self, pts: f64, w: usize, h: usize, luma: &[u8]) {
        if pts.is_nan() || pts < 0.0 || w < GW || h < GH {
            return;
        }
        let sec = pts as usize;
        if self.seconds.len() <= sec {
            self.seconds.resize(sec + 1, None);
        }
        if self.seconds[sec].is_some() {
            return; // one sample per second is enough
        }
        let mut maps = [[0u64; WORDS]; 4];
        for (c, map) in maps.iter_mut().enumerate() {
            let x0 = if c % 2 == 0 { INSET_X } else { 1.0 - INSET_X - BOX_W };
            let y0 = if c < 2 { INSET_Y } else { 1.0 - INSET_Y - BOX_H };
            *map = edge_map(
                w,
                h,
                luma,
                (x0 * w as f64) as usize,
                (y0 * h as f64) as usize,
                ((BOX_W * w as f64) as usize).max(GW),
                ((BOX_H * h as f64) as usize).max(GH),
            );
        }
        self.seconds[sec] = Some(maps);
    }

    /// Per second, whether the logo is on screen (None where no frame was seen).
    pub fn presence(&self) -> Vec<Option<bool>> {
        let seen: Vec<&[Map; 4]> = self.seconds.iter().flatten().collect();
        if seen.is_empty() {
            return vec![None; self.seconds.len()];
        }
        // The logo's outline: the cells that are an edge in most seconds. The corner where that outline is
        // largest is where the logo is.
        let mut best: Option<(usize, Map, u32)> = None;
        for c in 0..4 {
            let mut count = [0u32; BITS];
            for m in &seen {
                for (b, n) in count.iter_mut().enumerate() {
                    *n += u32::from(bit(&m[c], b));
                }
            }
            let mut tmpl = [0u64; WORDS];
            for (b, n) in count.iter().enumerate() {
                if *n as usize * 2 > seen.len() {
                    tmpl[b / 64] |= 1 << (b % 64);
                }
            }
            let size: u32 = tmpl.iter().map(|w| w.count_ones()).sum();
            if best.as_ref().is_none_or(|(_, _, s)| size > *s) {
                best = Some((c, tmpl, size));
            }
        }
        let (corner, tmpl, size) = best.expect("four corners");
        // Too small an outline is noise, not a logo: then there is nothing to tell programme from ads.
        if size < 20 {
            return vec![None; self.seconds.len()];
        }
        self.seconds.iter().map(|s| s.as_ref().map(|m| overlap(&m[corner], &tmpl) * 2 > size)).collect()
    }
}

fn bit(m: &Map, b: usize) -> bool {
    m[b / 64] >> (b % 64) & 1 == 1
}

fn overlap(a: &Map, b: &Map) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

/// Average the box (x0, y0, bw × bh) down to GW × GH and mark cells that differ sharply from their right or lower
/// neighbour.
fn edge_map(w: usize, h: usize, luma: &[u8], x0: usize, y0: usize, bw: usize, bh: usize) -> Map {
    let (bw, bh) = (bw.min(w - x0.min(w - 1)), bh.min(h - y0.min(h - 1)));
    let mut g = [[0i32; GW]; GH];
    for (gy, row) in g.iter_mut().enumerate() {
        let (ya, yb) = (y0 + gy * bh / GH, (y0 + (gy + 1) * bh / GH).max(y0 + gy * bh / GH + 1));
        for (gx, cell) in row.iter_mut().enumerate() {
            let (xa, xb) = (x0 + gx * bw / GW, (x0 + (gx + 1) * bw / GW).max(x0 + gx * bw / GW + 1));
            let (mut sum, mut n) = (0u32, 0u32);
            for y in ya..yb.min(h) {
                for x in xa..xb.min(w) {
                    sum += u32::from(luma[y * w + x]);
                    n += 1;
                }
            }
            *cell = (sum / n.max(1)) as i32;
        }
    }
    let mut m = [0u64; WORDS];
    for y in 0..GH - 1 {
        for x in 0..GW - 1 {
            if (g[y][x] - g[y][x + 1]).abs() > EDGE || (g[y][x] - g[y + 1][x]).abs() > EDGE {
                let b = y * (GW - 1) + x;
                m[b / 64] |= 1 << (b % 64);
            }
        }
    }
    m
}
