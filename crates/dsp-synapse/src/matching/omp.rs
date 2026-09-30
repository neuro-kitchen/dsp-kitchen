//! Orthogonal Matching Pursuit (OMP) / Wobble Template Deconvolution (`omp.rs`).
//!
//! Resolves colliding/overlapping action potentials by iteratively matching unit
//! templates against the multi-channel residual, fitting amplitude scaling
//! $a \in [a_{\min}, a_{\max}]$, and subtracting the fitted waveform.
//! Implements `crate::traits::SpikeMatcher`.

use crate::extraction::parabolic_subsample_offset;
use crate::metrics::WaveformTemplate;
use crate::traits::{MatchedSpike, SpikeMatcher};

/// Deconvolves multi-channel continuous data using Orthogonal Matching Pursuit (OMP)
/// against a dictionary of `WaveformTemplate`s.
///
/// Each template row is compared with the recording channel it belongs to
/// (`channel_ids`); rows on channels outside the recording are ignored. All templates must have the
/// same length. A match starting at sample `s` is reported at its trough, `s + trough_index`.
#[allow(clippy::too_many_arguments)]
pub fn match_spikes_omp(
    data: &[f32],
    channels: usize,
    samples: usize,
    templates: &[WaveformTemplate],
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
) -> Vec<MatchedSpike> {
    assert_eq!(data.len(), channels * samples);
    if templates.is_empty() || channels == 0 || samples == 0 {
        return Vec::new();
    }

    let t_len = templates[0].num_samples;
    assert!(templates.iter().all(|t| t.num_samples == t_len), "templates must share one length");
    if t_len == 0 || samples <= t_len {
        return Vec::new();
    }

    // (template row, recording channel) pairs inside the recording, per template.
    let rows: Vec<Vec<(usize, usize)>> = templates
        .iter()
        .map(|t| t.channel_ids.iter().enumerate().filter(|(_, c)| **c < channels).map(|(r, &c)| (r, c)).collect())
        .collect();

    // Energy ||W_u||^2 over the rows that take part in the match
    let energies: Vec<f32> = templates
        .iter()
        .zip(&rows)
        .map(|(t, rs)| rs.iter().map(|&(r, _)| t.row(r).iter().map(|v| v * v).sum::<f32>()).sum::<f32>().max(1e-8))
        .collect();

    let mut residual = data.to_vec();
    let mut matched = Vec::new();
    let valid_starts = samples - t_len;

    for _pass in 0..max_passes.max(1) {
        let mut added_in_pass = 0usize;

        // Compute best template and projection score at each time step s
        let mut best_unit_at = vec![0usize; valid_starts];
        let mut best_scale_at = vec![0.0f32; valid_starts];
        let mut best_gain_at = vec![0.0f32; valid_starts];

        for s in 0..valid_starts {
            let mut max_gain = 0.0f32;
            let mut best_u = 0usize;
            let mut best_a = 0.0f32;

            for (u, tmpl) in templates.iter().enumerate() {
                let mut dot = 0.0f32;
                for &(r, c) in &rows[u] {
                    let res = &residual[c * samples + s..c * samples + s + t_len];
                    dot += res.iter().zip(tmpl.row(r)).map(|(x, w)| x * w).sum::<f32>();
                }

                let raw_a = dot / energies[u];
                if raw_a >= min_amplitude_scale && raw_a <= max_amplitude_scale {
                    // Energy reduction = 2 * a * dot - a^2 * ||W||^2
                    let gain = 2.0 * raw_a * dot - raw_a * raw_a * energies[u];
                    if gain > max_gain {
                        max_gain = gain;
                        best_u = u;
                        best_a = raw_a;
                    }
                }
            }

            best_unit_at[s] = best_u;
            best_scale_at[s] = best_a;
            best_gain_at[s] = max_gain;
        }

        // Greedily pick local maxima in energy reduction that exceed `min_explained_energy`
        let mut s = 1usize;
        while s + 1 < valid_starts {
            let g = best_gain_at[s];
            if g >= min_explained_energy && g >= best_gain_at[s - 1] && g >= best_gain_at[s + 1] {
                let u = best_unit_at[s];
                let scale = best_scale_at[s];
                let tmpl = &templates[u];

                let sub_lag = parabolic_subsample_offset(
                    -best_gain_at[s - 1],
                    -best_gain_at[s],
                    -best_gain_at[s + 1],
                );

                // Subtract scaled template from residual
                for &(r, c) in &rows[u] {
                    let res = &mut residual[c * samples + s..c * samples + s + t_len];
                    for (x, w) in res.iter_mut().zip(tmpl.row(r)) {
                        *x -= scale * w;
                    }
                }

                matched.push(MatchedSpike {
                    unit_id: u,
                    sample_index: (s + tmpl.trough_index) as u64,
                    subsample_lag: sub_lag,
                    amplitude_scale: scale,
                    score: g.sqrt(),
                });
                added_in_pass += 1;
                // Advance by `t_len` inside this pass so any overlapping spike within `< t_len`
                // is resolved in the next OMP pass after `residual` has been updated!
                s += t_len;
            } else {
                s += 1;
            }
        }

        if added_in_pass == 0 {
            break;
        }
    }

    matched.sort_by_key(|m| m.sample_index);
    matched
}

/// Orthogonal Matching Pursuit template matcher implementing [`SpikeMatcher`].
#[derive(Debug, Clone, Copy)]
pub struct OmpSpikeMatcher {
    pub min_amplitude_scale: f32,
    pub max_amplitude_scale: f32,
    pub min_explained_energy: f32,
    pub max_passes: usize,
}

impl Default for OmpSpikeMatcher {
    fn default() -> Self {
        Self {
            min_amplitude_scale: 0.65,
            max_amplitude_scale: 1.45,
            min_explained_energy: 500.0,
            max_passes: 4,
        }
    }
}

impl SpikeMatcher for OmpSpikeMatcher {
    fn match_spikes(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        templates: &[WaveformTemplate],
    ) -> Vec<MatchedSpike> {
        match_spikes_omp(
            data,
            channels,
            samples,
            templates,
            self.min_amplitude_scale,
            self.max_amplitude_scale,
            self.min_explained_energy,
            self.max_passes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omp_resolves_two_colliding_spikes() {
        let channels = 2;
        let t_len = 20;
        let samples = 200;

        // Unit 0 strong on channel 0, Unit 1 strong on channel 1
        let mut mean0 = vec![0.0f32; channels * t_len];
        let mut mean1 = vec![0.0f32; channels * t_len];
        for i in 0..t_len {
            let x = (i as f32 - 10.0) / 1.5;
            let wave = -100.0 * (1.0 - 0.35 * x * x) * (-0.5 * x * x).exp();
            mean0[i] = wave;
            mean0[t_len + i] = wave * 0.25;

            mean1[i] = wave * 0.20;
            mean1[t_len + i] = wave * 1.10;
        }

        let templates = vec![
            WaveformTemplate::new(vec![0, 1], t_len, mean0.clone(), vec![1.0; channels * t_len]),
            WaveformTemplate::new(vec![0, 1], t_len, mean1.clone(), vec![1.0; channels * t_len]),
        ];

        // Inject BOTH Unit 0 at s=80 and Unit 1 at s=84 (overlapping by 16 of 20 samples!)
        let mut raw = vec![0.0f32; channels * samples];
        for c in 0..channels {
            for i in 0..t_len {
                raw[c * samples + 80 + i] += mean0[c * t_len + i];
                raw[c * samples + 84 + i] += mean1[c * t_len + i];
            }
        }

        let matched = match_spikes_omp(&raw, channels, samples, &templates, 0.7, 1.3, 1000.0, 4);
        assert_eq!(matched.len(), 2);
        // Reported at the trough (sample 10 of the template), not the template centre.
        assert_eq!(templates[0].trough_index, 10);
        assert_eq!(matched[0].unit_id, 0);
        assert_eq!(matched[0].sample_index, 90);
        assert_eq!(matched[1].unit_id, 1);
        assert_eq!(matched[1].sample_index, 94);
    }

    #[test]
    fn test_omp_uses_template_channels_and_reports_trough() {
        // 32-channel recording; unit 0 on channels 20-23, unit 1 on channels 24-27.
        let (channels, samples, t_len, trough) = (32usize, 1_000usize, 30usize, 10usize);
        let wave = |i: usize| {
            let x = (i as f32 - trough as f32) / 2.0;
            -100.0 * (1.0 - 0.3 * x * x) * (-0.5 * x * x).exp()
        };
        let template = |gains: [f32; 4], first: usize| {
            let mut mean = Vec::new();
            for g in gains {
                mean.extend((0..t_len).map(|i| g * wave(i)));
            }
            WaveformTemplate::new((first..first + 4).collect(), t_len, mean, vec![1.0; 4 * t_len])
        };
        let templates = vec![template([1.0, 0.6, 0.3, 0.1], 20), template([0.2, 1.0, 0.7, 0.4], 24)];
        assert_eq!(templates[0].trough_index, trough);

        let mut raw = vec![0.0f32; channels * samples];
        let events = [(0usize, 200usize), (1, 215), (0, 600), (1, 800)];
        for &(u, start) in &events {
            let t = &templates[u];
            for (r, &c) in t.channel_ids.iter().enumerate() {
                for (i, w) in t.row(r).iter().enumerate() {
                    raw[c * samples + start + i] += w;
                }
            }
        }
        // A large unrelated signal on channels 0-3 (where the old code looked) must not match.
        for c in 0..4 {
            for i in 0..t_len {
                raw[c * samples + 400 + i] += 3.0 * wave(i);
            }
        }

        let matched = match_spikes_omp(&raw, channels, samples, &templates, 0.7, 1.3, 1_000.0, 4);
        let got: Vec<(usize, u64)> = matched.iter().map(|m| (m.unit_id, m.sample_index)).collect();
        let expected: Vec<(usize, u64)> = events.iter().map(|&(u, s)| (u, (s + trough) as u64)).collect();
        assert_eq!(got, expected);
    }
}
