//! Template matching by greedy matching pursuit with bounded amplitudes.
//!
//! Resolves colliding / overlapping action potentials by repeatedly matching unit templates
//! against the multi-channel residual, fitting an amplitude scale $a \in [a_{\min}, a_{\max}]$ and
//! subtracting the fitted waveform. Earlier picks are not re-fitted (not *orthogonal* matching
//! pursuit). Implements `crate::core::SpikeMatcher`.

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_base::peaks::{find_peak_candidates, select_by_distance, DistanceRule, Polarity};
use dsp_core::compute::LaunchGeometry;
use dsp_core::DspResult;

use crate::core::{MatchedSpike, SpikeMatcher, WaveformTemplate};
use dsp_base::math::parabolic_vertex_offset;
use super::kernels::{mp_gather_picks_kernel, mp_score_kernel, mp_subtract_kernel};

/// Template energy floor (µV²) so an all-zero template divides safely.
const MIN_TEMPLATE_ENERGY: f32 = 1e-8;

/// Matching-pursuit settings.
#[derive(Debug, Clone, Copy)]
struct MatchingPursuitParams {
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
}

/// Deconvolves multi-channel continuous data against a dictionary of `WaveformTemplate`s on
/// `client`'s runtime.
///
/// Each template row is compared with the recording channel it belongs to
/// (`channel_ids`); rows on channels outside the recording are ignored. All templates must have the
/// same length. A match starting at sample `s` is reported at its trough, `s + trough_index`.
///
/// Every pass scores all starts × templates on the device (best template, amplitude in
/// `[min, max]` and energy reduction per start), finds the local maxima of the reduction above
/// `min_explained_energy` on the device (`dsp_base::peaks::find_peak_candidates`), keeps those
/// with no larger one within a template length (so a pass's picks never overlap), and subtracts
/// them from the device residual; overlapping spikes are resolved by later passes. Only the picks
/// are downloaded.
#[allow(clippy::too_many_arguments)]
pub fn match_spikes_matching_pursuit(
    client: &Client,
    data: &[f32],
    channels: usize,
    samples: usize,
    templates: &[WaveformTemplate],
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
) -> Vec<MatchedSpike> {
    let params = MatchingPursuitParams { min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes };
    matching_pursuit_on(client, data, channels, samples, templates, params)
}

fn matching_pursuit_on(
    client: &Client,
    data: &[f32],
    channels: usize,
    samples: usize,
    templates: &[WaveformTemplate],
    params: MatchingPursuitParams,
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
        energies.push(energy.max(MIN_TEMPLATE_ENERGY));
    }
    let max_rows = row_offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0) as usize;
    if max_rows == 0 {
        return Vec::new();
    }
    let num_units = templates.len();
    let valid_starts = samples - t_len;

    let residual = buffer::upload(client, data);
    let offsets_h = buffer::upload(client, &row_offsets);
    let channels_h = buffer::upload(client, &row_channels);
    let data_h = buffer::upload(client, &row_data);
    let energies_h = buffer::upload(client, &energies);
    let unit_h = buffer::empty::<u32>(client, valid_starts);
    let scale_h = buffer::empty::<f32>(client, valid_starts);
    let gain_h = buffer::empty::<f32>(client, valid_starts);
    let min_gain_h = buffer::upload(client, &[params.min_explained_energy]);
    let rows_len = row_channels.len();

    let mut matched = Vec::new();
    for _pass in 0..params.max_passes.max(1) {
        let geom = LaunchGeometry::elementwise(client, valid_starts);
        unsafe {
            mp_score_kernel::launch(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(residual.clone(), channels * samples),
                BufferArg::from_raw_parts(offsets_h.clone(), num_units + 1),
                BufferArg::from_raw_parts(channels_h.clone(), rows_len),
                BufferArg::from_raw_parts(data_h.clone(), rows_len * t_len),
                BufferArg::from_raw_parts(energies_h.clone(), num_units),
                BufferArg::from_raw_parts(unit_h.clone(), valid_starts),
                BufferArg::from_raw_parts(scale_h.clone(), valid_starts),
                BufferArg::from_raw_parts(gain_h.clone(), valid_starts),
                samples as u32,
                valid_starts as u32,
                num_units as u32,
                t_len as u32,
                params.min_amplitude_scale,
                params.max_amplitude_scale,
            );
        }
        // Local maxima of the energy reduction above the floor, found and compacted on the device
        let candidates = find_peak_candidates::<f32>(client, &gain_h, &min_gain_h, 1, valid_starts, 0..valid_starts, Polarity::Positive);
        let (starts, gains) = candidates.channel(0);
        if starts.is_empty() {
            break;
        }
        // A start stays unless a larger reduction lies within a template length: picks of one
        // pass never overlap, so the subtract kernel never races
        let positions: Vec<usize> = starts.iter().map(|&s| s as usize).collect();
        let keep = select_by_distance(&positions, gains, t_len, DistanceRule::LocallyExclusive);
        let kept: Vec<usize> = (0..positions.len()).filter(|&i| keep[i]).collect();
        let pick_starts: Vec<u32> = kept.iter().map(|&i| starts[i]).collect();
        let picks = pick_starts.len();

        let starts_h = buffer::upload(client, &pick_starts);
        let units_out = buffer::empty::<u32>(client, picks);
        let scales_out = buffer::empty::<f32>(client, picks);
        let prev_out = buffer::empty::<f32>(client, picks);
        let next_out = buffer::empty::<f32>(client, picks);
        let per_pick = LaunchGeometry::elementwise(client, picks);
        unsafe {
            mp_gather_picks_kernel::launch(
                client,
                per_pick.cube_count,
                per_pick.cube_dim,
                BufferArg::from_raw_parts(unit_h.clone(), valid_starts),
                BufferArg::from_raw_parts(scale_h.clone(), valid_starts),
                BufferArg::from_raw_parts(gain_h.clone(), valid_starts),
                BufferArg::from_raw_parts(starts_h.clone(), picks),
                BufferArg::from_raw_parts(units_out.clone(), picks),
                BufferArg::from_raw_parts(scales_out.clone(), picks),
                BufferArg::from_raw_parts(prev_out.clone(), picks),
                BufferArg::from_raw_parts(next_out.clone(), picks),
                picks as u32,
            );
        }
        let pick_units = buffer::download::<u32>(client, units_out);
        let pick_scales = buffer::download::<f32>(client, scales_out);
        let gain_prev = buffer::download::<f32>(client, prev_out);
        let gain_next = buffer::download::<f32>(client, next_out);
        for (n, &i) in kept.iter().enumerate() {
            let s = starts[i] as usize;
            let g = gains[i];
            // Sub-sample lag from the reductions around the pick
            let sub_lag = parabolic_vertex_offset(gain_prev[n], g, gain_next[n]);
            let u = pick_units[n] as usize;
            matched.push(MatchedSpike {
                unit_id: u,
                sample_index: (s + templates[u].trough_index) as u64,
                subsample_lag: sub_lag,
                amplitude_scale: pick_scales[n],
                score: g.sqrt(),
            });
        }

        let geom = LaunchGeometry::elementwise(client, picks * max_rows * t_len);
        unsafe {
            mp_subtract_kernel::launch(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(residual.clone(), channels * samples),
                BufferArg::from_raw_parts(offsets_h.clone(), num_units + 1),
                BufferArg::from_raw_parts(channels_h.clone(), rows_len),
                BufferArg::from_raw_parts(data_h.clone(), rows_len * t_len),
                BufferArg::from_raw_parts(buffer::upload(client, &pick_units), picks),
                BufferArg::from_raw_parts(starts_h, picks),
                BufferArg::from_raw_parts(buffer::upload(client, &pick_scales), picks),
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

/// Matching-pursuit template matcher implementing [`SpikeMatcher`].
#[derive(Debug, Clone, Copy)]
pub struct MatchingPursuitMatcher {
    /// Amplitude scale range a template may be fitted with.
    pub min_amplitude_scale: f32,
    pub max_amplitude_scale: f32,
    /// Energy reduction (µV², so it depends on the signal's scale) a match must explain.
    pub min_explained_energy: f32,
    /// Matching passes (overlapping spikes are resolved in later passes).
    pub max_passes: usize,
}

/// Defaults of [`MatchingPursuitMatcher`]: amplitudes within `0.65 – 1.45` of the template, 4
/// passes, and an energy floor of 500 µV² — tuned for µV extracellular recordings; rescale it with
/// the data (it grows with the square of the amplitude unit).
pub const DEFAULT_MIN_AMPLITUDE_SCALE: f32 = 0.65;
pub const DEFAULT_MAX_AMPLITUDE_SCALE: f32 = 1.45;
pub const DEFAULT_MIN_EXPLAINED_ENERGY_UV2: f32 = 500.0;
pub const DEFAULT_MAX_PASSES: usize = 4;

impl Default for MatchingPursuitMatcher {
    fn default() -> Self {
        Self {
            min_amplitude_scale: DEFAULT_MIN_AMPLITUDE_SCALE,
            max_amplitude_scale: DEFAULT_MAX_AMPLITUDE_SCALE,
            min_explained_energy: DEFAULT_MIN_EXPLAINED_ENERGY_UV2,
            max_passes: DEFAULT_MAX_PASSES,
        }
    }
}

impl SpikeMatcher for MatchingPursuitMatcher {
    fn match_spikes(
        &self,
        client: &Client,
        data: &[f32],
        channels: usize,
        samples: usize,
        templates: &[WaveformTemplate],
    ) -> DspResult<Vec<MatchedSpike>> {
        Ok(match_spikes_matching_pursuit(
            client,
            data,
            channels,
            samples,
            templates,
            self.min_amplitude_scale,
            self.max_amplitude_scale,
            self.min_explained_energy,
            self.max_passes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};

    /// Matches on every compiled-in runtime (each result checked by the caller).
    fn match_everywhere(raw: &[f32], channels: usize, samples: usize, templates: &[WaveformTemplate], max_scale: f32, floor: f32) -> Vec<Vec<MatchedSpike>> {
        struct Task<'a>(&'a [f32], usize, usize, &'a [WaveformTemplate], f32, f32);
        impl ComputeTask for Task<'_> {
            type Output = Vec<MatchedSpike>;
            fn run(self, client: Client) -> Vec<MatchedSpike> {
                let min_scale = 2.0 - self.4;
                match_spikes_matching_pursuit(&client, self.0, self.1, self.2, self.3, min_scale, self.4, self.5, 4)
            }
        }
        let targets = ComputeTarget::available();
        assert!(!targets.is_empty(), "no CubeCL runtime compiled in");
        targets.into_iter().map(|t| t.run(Task(raw, channels, samples, templates, max_scale, floor)).expect("runtime")).collect()
    }

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

        for matched in match_everywhere(&raw, channels, samples, &templates, 1.3, 1000.0) {
        assert_eq!(matched.len(), 2);
        // Reported at the trough (sample 10 of the template), not the template centre.
        assert_eq!(templates[0].trough_index, 10);
        assert_eq!(matched[0].unit_id, 0);
        assert_eq!(matched[0].sample_index, 90);
        assert_eq!(matched[1].unit_id, 1);
        assert_eq!(matched[1].sample_index, 94);
        }
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

        let expected: Vec<(usize, u64)> = events.iter().map(|&(u, s)| (u, (s + trough) as u64)).collect();
        for matched in match_everywhere(&raw, channels, samples, &templates, 1.3, 1_000.0) {
            let got: Vec<(usize, u64)> = matched.iter().map(|m| (m.unit_id, m.sample_index)).collect();
            assert_eq!(got, expected);
        }
    }
}
