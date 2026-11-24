use crate::Config;

/// Silences: runs of samples all below the threshold.
#[derive(Default)]
pub(crate) struct Silence {
    /// Start of the open run, and where the last pushed sample ended.
    run: Option<f64>,
    end: f64,
    /// Closed runs, (start, end) in seconds.
    pub runs: Vec<(f64, f64)>,
}

impl Silence {
    pub fn push(&mut self, cfg: &Config, pts: f64, rate: u32, samples: &[f32]) {
        let th = 10f32.powf(cfg.silence_db / 20.0);
        let rate = f64::from(rate.max(1));
        for (i, s) in samples.iter().enumerate() {
            let t = pts + i as f64 / rate;
            if s.abs() <= th {
                self.run.get_or_insert(t);
            } else if let Some(start) = self.run.take()
                && t - start >= cfg.min_silence_secs
            {
                self.runs.push((start, t));
            }
        }
        self.end = pts + samples.len() as f64 / rate;
    }

    pub fn finish(&mut self, cfg: &Config) {
        if let Some(start) = self.run.take()
            && self.end - start >= cfg.min_silence_secs
        {
            self.runs.push((start, self.end));
        }
    }
}
