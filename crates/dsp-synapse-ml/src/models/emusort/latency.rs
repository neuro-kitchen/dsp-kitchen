//! Conduction-velocity and latency alignment across muscle electrode channels.

/// Cross-channel conduction latency aligner for Motor Unit Action Potentials (MUAPs).
#[derive(Debug, Clone)]
pub struct EmusortLatencyAligner {
    pub max_lag_samples: usize,
    pub min_conduction_velocity: f32,
    pub max_conduction_velocity: f32,
}

impl Default for EmusortLatencyAligner {
    fn default() -> Self {
        Self {
            max_lag_samples: 25,
            min_conduction_velocity: 2.5,
            max_conduction_velocity: 6.0,
        }
    }
}

impl EmusortLatencyAligner {
    /// Creates a new latency aligner with the specified maximum lag in samples.
    pub fn new(max_lag_samples: usize) -> Self {
        Self {
            max_lag_samples: max_lag_samples.max(1),
            ..Self::default()
        }
    }

    /// Estimates integer sample lag of each channel relative to `ref_ch` using normalized cross-correlation.
    pub fn estimate_channel_lags(
        &self,
        snippet: &[f32],
        channels: usize,
        samples: usize,
        ref_ch: usize,
    ) -> Vec<isize> {
        assert_eq!(snippet.len(), channels * samples);
        assert!(ref_ch < channels);

        let mut lags = vec![0isize; channels];
        let ref_trace = &snippet[ref_ch * samples..(ref_ch + 1) * samples];
        let max_l = self.max_lag_samples.min(samples / 3);

        for ch in 0..channels {
            if ch == ref_ch {
                continue;
            }
            let ch_trace = &snippet[ch * samples..(ch + 1) * samples];
            let mut best_lag = 0isize;
            let mut max_corr = -f32::INFINITY;

            for lag in -(max_l as isize)..=(max_l as isize) {
                let mut sum = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;

                for t in 0..samples {
                    let t_shifted = t as isize + lag;
                    if t_shifted >= 0 && (t_shifted as usize) < samples {
                        let a = ref_trace[t];
                        let b = ch_trace[t_shifted as usize];
                        sum += a * b;
                        norm_a += a * a;
                        norm_b += b * b;
                    }
                }

                let denom = (norm_a * norm_b).sqrt();
                let corr = if denom > 1e-9 { sum / denom } else { 0.0 };
                if corr > max_corr {
                    max_corr = corr;
                    best_lag = lag;
                }
            }

            lags[ch] = best_lag;
        }

        lags
    }

    /// Time-shifts each channel's row in `snippet` by `-lag` so all channels align to the reference channel.
    pub fn align_snippet(
        &self,
        snippet: &[f32],
        channels: usize,
        samples: usize,
        lags: &[isize],
    ) -> Vec<f32> {
        assert_eq!(snippet.len(), channels * samples);
        assert_eq!(lags.len(), channels);

        let mut aligned = vec![0.0f32; channels * samples];
        for ch in 0..channels {
            let ch_in = &snippet[ch * samples..(ch + 1) * samples];
            let ch_out = &mut aligned[ch * samples..(ch + 1) * samples];
            let lag = lags[ch];

            for t in 0..samples {
                let src_t = t as isize + lag;
                if src_t >= 0 && (src_t as usize) < samples {
                    ch_out[t] = ch_in[src_t as usize];
                } else {
                    ch_out[t] = 0.0;
                }
            }
        }

        aligned
    }
}

/// Backward-compatible type alias for [`EmusortLatencyAligner`].
pub type MyomatrixLatencyAligner = EmusortLatencyAligner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_aligner_recovers_and_compensates_delay() {
        let aligner = EmusortLatencyAligner::new(10);
        let channels = 3usize;
        let samples = 60usize;
        let mut snippet = vec![0.0f32; channels * samples];

        // Reference channel peak at sample 30
        snippet[0 * samples + 30] = 100.0;
        snippet[0 * samples + 31] = 80.0;

        // Channel 1 delayed by +3 samples (peak at 33)
        snippet[1 * samples + 33] = 90.0;
        snippet[1 * samples + 34] = 72.0;

        // Channel 2 advanced by -2 samples (peak at 28)
        snippet[2 * samples + 28] = 95.0;
        snippet[2 * samples + 29] = 76.0;

        let lags = aligner.estimate_channel_lags(&snippet, channels, samples, 0);
        assert_eq!(lags[0], 0);
        assert_eq!(lags[1], 3);
        assert_eq!(lags[2], -2);

        let aligned = aligner.align_snippet(&snippet, channels, samples, &lags);
        // In aligned output, all peaks should now be at sample 30
        assert_eq!(aligned[0 * samples + 30], 100.0);
        assert_eq!(aligned[1 * samples + 30], 90.0);
        assert_eq!(aligned[2 * samples + 30], 95.0);
    }
}
