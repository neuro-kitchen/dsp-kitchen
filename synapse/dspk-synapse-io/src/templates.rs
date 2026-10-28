//! Dense spike templates as sorting files store them: one `[count, samples, channels]` array
//! (Phy / Kilosort `templates.npy`), converted from the `[count, channels, samples]` order of
//! Zarr / NWB `waveform_mean` and from Kilosort 1–3 sparse templates.

/// Storage axis order of a 3-D template array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateAxisOrder {
    /// `[count, samples, channels]` (Phy / Kilosort `templates.npy`).
    SamplesChannels,
    /// `[count, channels, samples]` (Zarr / NWB `/units/waveform_mean`).
    ChannelsSamples,
}

/// Dense templates `[count, samples, channels]` (C order).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DenseTemplates {
    pub data: Vec<f32>,
    pub count: usize,
    pub samples: usize,
    pub channels: usize,
}

impl DenseTemplates {
    /// Densifies sparse templates (`[count, samples, stored_channels]` plus the
    /// `[count, stored_channels]` channel table of Kilosort 1–3) onto `total_channels`.
    pub fn from_sparse(
        sparse: &[f32],
        count: usize,
        samples: usize,
        stored_channels: usize,
        ind: &[usize],
        total_channels: usize,
    ) -> Self {
        let channels = total_channels.max(stored_channels);
        let mut dense = vec![0.0f32; count * samples * channels];
        for t in 0..count {
            for k in 0..stored_channels {
                if let Some(&ch) = ind.get(t * stored_channels + k)
                    && ch < channels
                {
                    for s in 0..samples {
                        dense[(t * samples + s) * channels + ch] = sparse[(t * samples + s) * stored_channels + k];
                    }
                }
            }
        }
        Self { data: dense, count, samples, channels }
    }

    /// Converts a `[count, channels, samples]` buffer.
    pub fn from_channels_samples(buf: &[f32], count: usize, channels: usize, samples: usize) -> Self {
        let mut data = vec![0.0f32; count * samples * channels];
        for t in 0..count {
            for c in 0..channels {
                for s in 0..samples {
                    data[(t * samples + s) * channels + c] = buf[(t * channels + c) * samples + s];
                }
            }
        }
        Self { data, count, samples, channels }
    }

    /// Template `t` on `channel` over time (zeros when out of range).
    pub fn trace(&self, t: usize, channel: usize) -> Vec<f32> {
        if t >= self.count || channel >= self.channels {
            return vec![0.0; self.samples];
        }
        (0..self.samples).map(|s| self.data[(t * self.samples + s) * self.channels + channel]).collect()
    }

    /// Peak-to-peak of template `t` on each channel.
    pub fn peak_to_peak(&self, t: usize) -> Vec<f32> {
        if t >= self.count {
            return vec![0.0; self.channels];
        }
        (0..self.channels)
            .map(|c| {
                let trace = self.trace(t, c);
                let (lo, hi) = trace.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                if trace.is_empty() { 0.0 } else { hi - lo }
            })
            .collect()
    }

    /// Channel where template `t` is largest (peak-to-peak).
    pub fn best_channel(&self, t: usize) -> usize {
        self.top_channels(t, 1).first().copied().unwrap_or(0)
    }

    /// The `k` channels where template `t` is largest, largest first.
    pub fn top_channels(&self, t: usize, k: usize) -> Vec<usize> {
        let ptp = self.peak_to_peak(t);
        let mut order: Vec<usize> = (0..self.channels).collect();
        order.sort_by(|&a, &b| ptp[b].total_cmp(&ptp[a]).then(a.cmp(&b)));
        order.truncate(k.min(self.channels));
        order
    }

    /// Largest peak-to-peak of template `t` over its channels.
    pub fn amplitude(&self, t: usize) -> f32 {
        self.peak_to_peak(t).into_iter().fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_and_channel_major_densify() {
        // 1 template, 2 samples, stored on channel 2 of 3
        let t = DenseTemplates::from_sparse(&[-5.0, 3.0], 1, 2, 1, &[2], 3);
        assert_eq!(t.trace(0, 2), vec![-5.0, 3.0]);
        assert_eq!((t.best_channel(0), t.amplitude(0)), (2, 8.0));
        let c = DenseTemplates::from_channels_samples(&[0.0, 0.0, 0.0, 0.0, -5.0, 3.0], 1, 3, 2);
        assert_eq!(c, t);
    }
}
