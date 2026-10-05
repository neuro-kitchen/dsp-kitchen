//! Orthogonal Matching Pursuit (OMP) / Wobble Template Deconvolution (`omp.rs`).
//!
//! Resolves colliding/overlapping action potentials by iteratively matching unit
//! templates against the multi-channel residual, fitting amplitude scaling
//! $a \in [a_{\min}, a_{\max}]$, and subtracting the fitted waveform.
//! Implements `crate::core::SpikeMatcher`.

use cubecl::prelude::*;
use dsp_core::compute::{ComputeTarget, ComputeTask, LaunchGeometry};
use dsp_core::{DspError, DspResult};

use crate::core::{MatchedSpike, SpikeMatcher, WaveformTemplate};
use dsp_base::math::parabolic_vertex_offset;
use super::kernels::{omp_score_kernel, omp_subtract_kernel};

/// OMP settings shared by the host entry points.
#[derive(Debug, Clone, Copy)]
struct OmpParams {
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
}

/// Deconvolves multi-channel continuous data using Orthogonal Matching Pursuit (OMP)
/// against a dictionary of `WaveformTemplate`s, on the default compute target
/// ([`ComputeTarget::from_env`]); see [`match_spikes_omp_on`].
///
/// Fails when no compute runtime is compiled in.
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
) -> DspResult<Vec<MatchedSpike>> {
    struct Task<'a> {
        data: &'a [f32],
        channels: usize,
        samples: usize,
        templates: &'a [WaveformTemplate],
        params: OmpParams,
    }
    impl ComputeTask for Task<'_> {
        type Output = Vec<MatchedSpike>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Vec<MatchedSpike> {
            omp_on(&client, self.data, self.channels, self.samples, self.templates, self.params)
        }
    }
    let params = OmpParams { min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes };
    ComputeTarget::from_env()
        .and_then(|target| target.run(Task { data, channels, samples, templates, params }))
        .map_err(|e| DspError::ComputeError(e.to_string()))
}

/// [`match_spikes_omp`] on `client`'s runtime.
///
/// Each template row is compared with the recording channel it belongs to
/// (`channel_ids`); rows on channels outside the recording are ignored. All templates must have the
/// same length. A match starting at sample `s` is reported at its trough, `s + trough_index`.
///
/// Every pass scores all starts × templates on the device (best template, amplitude in
/// `[min, max]` and energy reduction per start), picks local maxima of the reduction above
/// `min_explained_energy` on the host, and subtracts them from the device residual; overlapping
/// spikes are resolved by later passes.
#[allow(clippy::too_many_arguments)]
pub fn match_spikes_omp_on<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    channels: usize,
    samples: usize,
    templates: &[WaveformTemplate],
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
) -> Vec<MatchedSpike> {
    let params = OmpParams { min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes };
    omp_on(client, data, channels, samples, templates, params)
}

fn omp_on<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    channels: usize,
    samples: usize,
    templates: &[WaveformTemplate],
    params: OmpParams,
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

    // Template rows on channels inside the recording, flattened per unit
    let mut row_offsets = vec![0u32];
    let mut row_channels = Vec::new();
    let mut row_data = Vec::new();
    let mut energies = Vec::with_capacity(templates.len());
    for t in templates {
        let mut energy = 0.0f32;
        for (r, &c) in t.channel_ids.iter().enumerate().filter(|(_, c)| **c < channels) {
            row_channels.push(c as u32);
            row_data.extend_from_slice(t.row(r));
            energy += t.row(r).iter().map(|v| v * v).sum::<f32>();
        }
        row_offsets.push(row_channels.len() as u32);
        energies.push(energy.max(1e-8));
    }
    let max_rows = row_offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0) as usize;
    if max_rows == 0 {
        return Vec::new();
    }
    let num_units = templates.len();
    let valid_starts = samples - t_len;

    let residual = client.create_from_slice(f32::as_bytes(data));
    let offsets_h = client.create_from_slice(u32::as_bytes(&row_offsets));
    let channels_h = client.create_from_slice(u32::as_bytes(&row_channels));
    let data_h = client.create_from_slice(f32::as_bytes(&row_data));
    let energies_h = client.create_from_slice(f32::as_bytes(&energies));
    let unit_h = client.empty(valid_starts * 4);
    let scale_h = client.empty(valid_starts * 4);
    let gain_h = client.empty(valid_starts * 4);
    let rows_len = row_channels.len();

    let mut matched = Vec::new();
    for _pass in 0..params.max_passes.max(1) {
        let geom = LaunchGeometry::elementwise(client, valid_starts);
        unsafe {
            omp_score_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(residual.clone(), channels * samples),
                ArrayArg::from_raw_parts(offsets_h.clone(), num_units + 1),
                ArrayArg::from_raw_parts(channels_h.clone(), rows_len),
                ArrayArg::from_raw_parts(data_h.clone(), rows_len * t_len),
                ArrayArg::from_raw_parts(energies_h.clone(), num_units),
                ArrayArg::from_raw_parts(unit_h.clone(), valid_starts),
                ArrayArg::from_raw_parts(scale_h.clone(), valid_starts),
                ArrayArg::from_raw_parts(gain_h.clone(), valid_starts),
                samples as u32,
                valid_starts as u32,
                num_units as u32,
                t_len as u32,
                params.min_amplitude_scale,
                params.max_amplitude_scale,
            );
        }
        let best_unit_at = u32::from_bytes(&client.read_one(unit_h.clone()).expect("VRAM read units")).to_vec();
        let best_scale_at = f32::from_bytes(&client.read_one(scale_h.clone()).expect("VRAM read scales")).to_vec();
        let best_gain_at = f32::from_bytes(&client.read_one(gain_h.clone()).expect("VRAM read gains")).to_vec();

        // Greedily pick local maxima in energy reduction that exceed `min_explained_energy`
        let (mut pick_units, mut pick_starts, mut pick_scales) = (Vec::new(), Vec::new(), Vec::new());
        let mut s = 1usize;
        while s + 1 < valid_starts {
            let g = best_gain_at[s];
            if g >= params.min_explained_energy && g >= best_gain_at[s - 1] && g >= best_gain_at[s + 1] {
                let u = best_unit_at[s] as usize;
                let scale = best_scale_at[s];
                let sub_lag = parabolic_vertex_offset(-best_gain_at[s - 1], -best_gain_at[s], -best_gain_at[s + 1]);
                pick_units.push(u as u32);
                pick_starts.push(s as u32);
                pick_scales.push(scale);
                matched.push(MatchedSpike {
                    unit_id: u,
                    sample_index: (s + templates[u].trough_index) as u64,
                    subsample_lag: sub_lag,
                    amplitude_scale: scale,
                    score: g.sqrt(),
                });
                // Skip the template length within this pass: a spike overlapping this one is
                // resolved in the next pass, after the residual has been updated
                s += t_len;
            } else {
                s += 1;
            }
        }
        if pick_units.is_empty() {
            break;
        }

        let picks = pick_units.len();
        let geom = LaunchGeometry::elementwise(client, picks * max_rows * t_len);
        unsafe {
            omp_subtract_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(residual.clone(), channels * samples),
                ArrayArg::from_raw_parts(offsets_h.clone(), num_units + 1),
                ArrayArg::from_raw_parts(channels_h.clone(), rows_len),
                ArrayArg::from_raw_parts(data_h.clone(), rows_len * t_len),
                ArrayArg::from_raw_parts(client.create_from_slice(u32::as_bytes(&pick_units)), picks),
                ArrayArg::from_raw_parts(client.create_from_slice(u32::as_bytes(&pick_starts)), picks),
                ArrayArg::from_raw_parts(client.create_from_slice(f32::as_bytes(&pick_scales)), picks),
                samples as u32,
                picks as u32,
                max_rows as u32,
                t_len as u32,
            );
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
    ) -> DspResult<Vec<MatchedSpike>> {
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

        let matched = match_spikes_omp(&raw, channels, samples, &templates, 0.7, 1.3, 1000.0, 4).unwrap();
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

        let matched = match_spikes_omp(&raw, channels, samples, &templates, 0.7, 1.3, 1_000.0, 4).unwrap();
        let got: Vec<(usize, u64)> = matched.iter().map(|m| (m.unit_id, m.sample_index)).collect();
        let expected: Vec<(usize, u64)> = events.iter().map(|&(u, s)| (u, (s + trough) as u64)).collect();
        assert_eq!(got, expected);
    }
}
