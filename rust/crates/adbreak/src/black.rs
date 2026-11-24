use crate::Config;

/// Black stretches: runs of frames where nearly every pixel is dark.
#[derive(Default)]
pub(crate) struct Black {
    /// The open run: (first black frame, last black frame).
    run: Option<(f64, f64)>,
    /// Closed runs, (start, end) in seconds; the end is the first non-black frame.
    pub runs: Vec<(f64, f64)>,
}

impl Black {
    pub fn push(&mut self, cfg: &Config, pts: f64, luma: &[u8]) {
        let dark = luma.iter().filter(|&&p| p <= cfg.black_luma).count();
        let black = !luma.is_empty() && dark as f32 >= cfg.black_ratio * luma.len() as f32;
        match (black, self.run) {
            (true, Some((start, _))) => self.run = Some((start, pts)),
            (true, None) => self.run = Some((pts, pts)),
            (false, Some((start, _))) => {
                self.run = None;
                if pts - start >= cfg.min_black_secs {
                    self.runs.push((start, pts));
                }
            }
            (false, None) => {}
        }
    }

    pub fn finish(&mut self, cfg: &Config) {
        if let Some((start, last)) = self.run.take()
            && last - start >= cfg.min_black_secs
        {
            self.runs.push((start, last));
        }
    }
}
