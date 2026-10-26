//! Spatial Gaussian-Process / Kriging Drift Interpolation (`kriging.rs`).
//!
//! Computes spatial covariance weights $W = K_{\text{target}, \text{source}} (K_{\text{source}, \text{source}} + \lambda I)^{-1}$
//! to interpolate multi-channel waveforms onto drift-corrected electrode positions.

use std::collections::HashMap;

use dsp_base::linalg::cholesky_solve;
use dsp_core::{DspError, DspResult};
use dsp_io::neuro::probe::SensorLayout;
use crate::extraction::SnippetBatch;
use super::drift::DriftEstimate;

/// Computes the $[M \times K]$ spatial Kriging weight matrix mapping observed channels
/// at `source_xy` to drift-shifted target positions `target_xy`. Fails when the regularized kernel
/// matrix is not positive definite (e.g. non-finite positions).
///
/// # Errors
///
/// When the regularized kernel matrix is not positive definite (e.g. non-finite positions).
pub fn compute_kriging_weight_matrix(
    source_xy: &[[f32; 2]],
    target_xy: &[[f32; 2]],
    sigma_um: f32,
    regularization: f32,
) -> DspResult<Vec<f32>> {
    let k = source_xy.len();
    let m = target_xy.len();
    if k == 0 || m == 0 {
        return Ok(Vec::new());
    }

    let two_sigma_sq = (2.0 * (sigma_um as f64) * (sigma_um as f64)).max(MIN_TWO_SIGMA_SQ_UM2);
    let reg = regularization.max(MIN_REGULARIZATION) as f64;

    let mut k_ss = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let dx = (source_xy[i][0] - source_xy[j][0]) as f64;
            let dy = (source_xy[i][1] - source_xy[j][1]) as f64;
            let mut val = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
            if i == j {
                val += reg;
            }
            k_ss[i * k + j] = val;
        }
    }

    // Right-hand side: K_source_target of shape [k, m]
    let mut k_st = vec![0.0f64; k * m];
    for i in 0..k {
        for j in 0..m {
            let dx = (source_xy[i][0] - target_xy[j][0]) as f64;
            let dy = (source_xy[i][1] - target_xy[j][1]) as f64;
            k_st[i * m + j] = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
        }
    }

    // Solve K_ss * W^T = K_st -> W^T is [k, m], transpose to W of shape [m, k]
    let wt = cholesky_solve(&k_ss, &k_st, k, m).ok_or_else(|| {
        DspError::InvalidConfig(format!("kriging kernel of {k} sites is not positive definite (non-finite site positions?)"))
    })?;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            w[r * k + c] = wt[c * m + r] as f32;
        }
    }
    Ok(w)
}

/// Smallest `2σ²` (µm²) of the Gaussian kernel (σ below ~0.7 µm would make it singular-prone).
const MIN_TWO_SIGMA_SQ_UM2: f64 = 1.0;
/// Smallest ridge added to the kernel diagonal.
const MIN_REGULARIZATION: f32 = 1e-5;

/// Drift quantization for weight caching (µm).
const DRIFT_QUANTUM_UM: f32 = 0.1;

/// Ridge `λ` added to the kernel diagonal by the drift corrections (relative to the unit kernel
/// peak): keeps the solve stable when sites nearly coincide.
pub const KRIGING_REGULARIZATION: f32 = 1e-2;

fn quantize(d: f32) -> i64 {
    (d / DRIFT_QUANTUM_UM).round() as i64
}

/// `(x, y)` of a recording channel, or an error when the layout has no site for it.
fn site_xy(layout: &SensorLayout, ch: usize) -> DspResult<[f32; 2]> {
    layout
        .get_site(ch)
        .map(|s| [s.position.x_um, s.position.y_um])
        .map_err(|_| DspError::InvalidConfig(format!("probe layout has no site for channel {ch}")))
}

/// Applies spatial Kriging interpolation to a `SnippetBatch` to compensate for vertical probe drift
/// $d(t)$: row `r` becomes the waveform at `site(channel_ids[r]) + (0, d(t))`, interpolated from the
/// snippet's own channels.
///
/// Only the snippet's K channels are available here, so targets moved beyond them are
/// extrapolated (attenuated); prefer [`correct_traces_drift_kriging`] before extraction. Weights are
/// cached per channel set and drift (0.1 µm steps). Errors if a channel has no site in `layout`.
///
/// # Errors
///
/// When a snippet channel has no site in `layout`, or a weight matrix cannot be computed.
pub fn correct_snippet_batch_drift_kriging(
    batch: &SnippetBatch,
    layout: &SensorLayout,
    drift: &DriftEstimate,
    sample_rate_hz: f64,
    sigma_um: f32,
) -> DspResult<SnippetBatch> {
    let n = batch.num_spikes;
    let k = batch.num_channels;
    let t = batch.num_samples;
    if n == 0 || k == 0 || t == 0 {
        return Ok(batch.clone());
    }

    let mut corrected_data = vec![0.0f32; n * k * t];
    let mut cache: HashMap<(Vec<usize>, i64), Vec<f32>> = HashMap::new();

    for i in 0..n {
        let t_sec = (batch.center_samples[i] as f64) / sample_rate_hz.max(1.0);
        let key = quantize(drift.interpolate_drift_at(t_sec));
        let ch_ids = batch.spike_channel_ids(i);
        let w = match cache.get(&(ch_ids.to_vec(), key)) {
            Some(w) => w,
            None => {
                let source_xy = ch_ids.iter().map(|&c| site_xy(layout, c)).collect::<DspResult<Vec<_>>>()?;
                let dy = key as f32 * DRIFT_QUANTUM_UM;
                let target_xy: Vec<[f32; 2]> = source_xy.iter().map(|&[x, y]| [x, y + dy]).collect();
                let w = compute_kriging_weight_matrix(&source_xy, &target_xy, sigma_um, KRIGING_REGULARIZATION)?;
                cache.entry((ch_ids.to_vec(), key)).or_insert(w)
            }
        };
        let in_snip = batch.snippet_slice(i);
        let out_off = i * k * t;

        // Multiply [K, K] weight matrix by [K, T] snippet
        for r in 0..k {
            let dst_row = &mut corrected_data[out_off + r * t..out_off + (r + 1) * t];
            for c in 0..k {
                let weight = w[r * k + c];
                for (d, s) in dst_row.iter_mut().zip(&in_snip[c * t..(c + 1) * t]) {
                    *d += weight * s;
                }
            }
        }
    }

    Ok(SnippetBatch::from_raw_parts(
        corrected_data,
        n,
        k,
        t,
        batch.peak_index,
        batch.primary_channels.clone(),
        batch.center_samples.clone(),
        batch.subsample_offsets.clone(),
        batch.channel_ids.clone(),
    ))
}

/// Quantized drift runs of a chunk: maximal sample ranges with one drift step.
fn drift_runs(samples: usize, start_sample: u64, drift: &DriftEstimate, sample_rate_hz: f64) -> Vec<(std::ops::Range<usize>, i64)> {
    let key_at = |s: usize| quantize(drift.interpolate_drift_at((start_sample + s as u64) as f64 / sample_rate_hz));
    let mut runs = Vec::new();
    let mut s0 = 0usize;
    while s0 < samples {
        let key = key_at(s0);
        let mut s1 = s0 + 1;
        while s1 < samples && key_at(s1) == key {
            s1 += 1;
        }
        runs.push((s0..s1, key));
        s0 = s1;
    }
    runs
}

/// Kriging weights of one drift step: per enabled site, (source channels, weights).
type SiteWeights = Vec<(Vec<usize>, Vec<f32>)>;

/// Drift correction of `[channels, samples]` chunks by kriging, on the host ([`Self::correct`]) or
/// the device ([`Self::correct_in_vram`]). Channel `c` becomes the signal at `site(c) + (0, d(t))`,
/// kriged from every enabled site within `radius_um` of that target. Drift is evaluated per sample
/// and quantized to 0.1 µm; weights are cached per drift step across chunks. Channels without a site
/// keep their samples.
#[derive(Debug, Clone)]
pub struct TraceKriging {
    sites: Vec<(usize, [f32; 2])>,
    sigma_um: f32,
    radius_um: f32,
    cache: HashMap<i64, SiteWeights>,
}

impl TraceKriging {
    pub fn new(layout: &SensorLayout, sigma_um: f32, radius_um: f32) -> Self {
        let sites = layout
            .contacts
            .iter()
            .filter(|s| s.enabled)
            .map(|s| (s.channel_id, [s.position.x_um, s.position.y_um]))
            .collect();
        Self { sites, sigma_um, radius_um, cache: HashMap::new() }
    }

    fn check_channels(&self, channels: usize) -> DspResult<()> {
        match self.sites.iter().find(|(c, _)| *c >= channels) {
            Some((c, _)) => Err(DspError::InvalidConfig(format!("site channel {c} is outside the {channels}-channel chunk"))),
            None => Ok(()),
        }
    }

    fn weights(&mut self, key: i64) -> DspResult<&SiteWeights> {
        if !self.cache.contains_key(&key) {
            let dy = key as f32 * DRIFT_QUANTUM_UM;
            let w = self
                .sites
                .iter()
                .map(|&(_, [x, y])| {
                    let target = [x, y + dy];
                    let near: Vec<&(usize, [f32; 2])> = self
                        .sites
                        .iter()
                        .filter(|(_, p)| ((p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2)).sqrt() <= self.radius_um)
                        .collect();
                    let src: Vec<[f32; 2]> = near.iter().map(|(_, p)| *p).collect();
                    let w = compute_kriging_weight_matrix(&src, &[target], self.sigma_um, KRIGING_REGULARIZATION)?;
                    Ok((near.iter().map(|(c, _)| *c).collect(), w))
                })
                .collect::<DspResult<SiteWeights>>()?;
            self.cache.insert(key, w);
        }
        Ok(&self.cache[&key])
    }

    /// Corrects a host chunk (first sample at global `start_sample`). Errors if a site's channel is
    /// outside the chunk.
    ///
    /// # Errors
    ///
    /// [`DspError::InvalidConfig`] when a site's channel is outside the chunk; when a weight matrix
    /// cannot be computed.
    pub fn correct(
        &mut self,
        data: &[f32],
        channels: usize,
        samples: usize,
        start_sample: u64,
        drift: &DriftEstimate,
        sample_rate_hz: f64,
    ) -> DspResult<Vec<f32>> {
        assert_eq!(data.len(), channels * samples);
        self.check_channels(channels)?;
        let mut out = data.to_vec();
        for (run, key) in drift_runs(samples, start_sample, drift, sample_rate_hz) {
            let site_channels: Vec<usize> = self.sites.iter().map(|(c, _)| *c).collect();
            let weights = self.weights(key)?;
            for (&ch, (src, w)) in site_channels.iter().zip(weights.iter()) {
                let dst = &mut out[ch * samples + run.start..ch * samples + run.end];
                dst.fill(0.0);
                for (&c, &wc) in src.iter().zip(w) {
                    for (d, x) in dst.iter_mut().zip(&data[c * samples + run.start..c * samples + run.end]) {
                        *d += wc * x;
                    }
                }
            }
        }
        Ok(out)
    }

    /// [`Self::correct`] of the `[channels, samples]` device buffer `input` into `output` (`F`
    /// values, distinct buffers). Only the weights of the chunk's drift steps are uploaded (ELLPACK
    /// rows, one set per step); all runs are applied in one launch.
    ///
    /// # Errors
    ///
    /// As [`Self::correct`].
    #[allow(clippy::too_many_arguments)]
    pub fn correct_in_vram<F: dsp_base::core::DspFloat>(
        &mut self,
        client: &cubecl::prelude::Client,
        input: &cubecl::server::Handle,
        output: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
        start_sample: u64,
        drift: &DriftEstimate,
        sample_rate_hz: f64,
    ) -> DspResult<()> {
        use cubecl::prelude::*;
        use dsp_base::core::{buffer, cast_f32};
        use dsp_core::compute::LaunchGeometry;
        use super::kernels::kriging_runs_kernel;

        self.check_channels(channels)?;
        if channels == 0 || samples == 0 {
            return Ok(());
        }
        let runs = drift_runs(samples, start_sample, drift, sample_rate_hz);
        // One weight set per distinct drift step of the chunk
        let mut keys: Vec<i64> = runs.iter().map(|(_, k)| *k).collect();
        keys.sort_unstable();
        keys.dedup();
        let site_channels: Vec<usize> = self.sites.iter().map(|(c, _)| *c).collect();
        let mut sets: Vec<Vec<Vec<(u32, f32)>>> = Vec::with_capacity(keys.len());
        for &key in &keys {
            // Identity for channels without a site
            let mut rows: Vec<Vec<(u32, f32)>> = (0..channels).map(|c| vec![(c as u32, 1.0)]).collect();
            for (&ch, (src, w)) in site_channels.iter().zip(self.weights(key)?.iter()) {
                rows[ch] = src.iter().zip(w).map(|(&c, &wc)| (c as u32, wc)).collect();
            }
            sets.push(rows);
        }
        let width = sets.iter().flatten().map(Vec::len).max().unwrap_or(1).max(1);
        let mut values = vec![0.0f32; keys.len() * channels * width];
        let mut indices = vec![0u32; keys.len() * channels * width];
        for (slot, rows) in sets.iter().enumerate() {
            for (o, row) in rows.iter().enumerate() {
                let base = (slot * channels + o) * width;
                for (k, &(c, w)) in row.iter().enumerate() {
                    values[base + k] = w;
                    indices[base + k] = c;
                }
            }
        }
        let mut run_starts: Vec<u32> = runs.iter().map(|(r, _)| r.start as u32).collect();
        run_starts.push(samples as u32);
        let run_slots: Vec<u32> = runs.iter().map(|(_, k)| keys.binary_search(k).expect("key listed") as u32).collect();

        let values_f: Vec<F> = cast_f32::<F>(&values);
        let geom = LaunchGeometry::channels_samples(client, channels, samples);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            kriging_runs_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), channels * samples),
                BufferArg::from_raw_parts(buffer::upload(client, &values_f), values.len()),
                BufferArg::from_raw_parts(buffer::upload(client, &indices), indices.len()),
                BufferArg::from_raw_parts(buffer::upload(client, &run_starts), run_starts.len()),
                BufferArg::from_raw_parts(buffer::upload(client, &run_slots), run_slots.len()),
                BufferArg::from_raw_parts(output.clone(), channels * samples),
                channels as u32,
                samples as u32,
                width as u32,
                runs.len() as u32,
            );
        }
        Ok(())
    }
}

/// One-off [`TraceKriging::correct`] (weights are not kept between calls).
///
/// # Errors
///
/// As [`TraceKriging::correct`].
#[allow(clippy::too_many_arguments)]
pub fn correct_traces_drift_kriging(
    data: &[f32],
    channels: usize,
    samples: usize,
    start_sample: u64,
    layout: &SensorLayout,
    drift: &DriftEstimate,
    sample_rate_hz: f64,
    sigma_um: f32,
    radius_um: f32,
) -> DspResult<Vec<f32>> {
    TraceKriging::new(layout, sigma_um, radius_um).correct(data, channels, samples, start_sample, drift, sample_rate_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kriging_identity_at_zero_drift() {
        let coords = vec![[0.0, 0.0], [0.0, 20.0], [16.0, 10.0], [16.0, 30.0]];
        let w = compute_kriging_weight_matrix(&coords, &coords, 25.0, 1e-4).unwrap();
        assert_eq!(w.len(), 16);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((w[i * 4 + j] - expected).abs() < 1e-2);
            }
        }
    }

    fn linear_probe(n: usize, pitch: f32) -> SensorLayout {
        use dsp_io::neuro::probe::{Position3D, SensorSite};
        SensorLayout::new("line", (0..n).map(|c| SensorSite::new(c, Position3D::new(0.0, c as f32 * pitch, 0.0), 0)).collect())
    }

    fn constant_drift(d: f32) -> DriftEstimate {
        DriftEstimate {
            time_bin_centers_sec: vec![0.0],
            drift_um: vec![d],
            activity_map: Vec::new(),
            num_time_bins: 1,
            num_depth_bins: 0,
            depth_min_um: 0.0,
            depth_bin_size_um: 1.0,
        }
    }

    /// Smooth spatial field over depth, moving by `shift` µm.
    fn field(y: f32, shift: f32) -> f32 {
        -100.0 * (-0.5 * ((y - 300.0 - shift) / 40.0).powi(2)).exp()
    }

    #[test]
    fn trace_correction_recovers_known_shift() {
        let (n, pitch, samples) = (32usize, 20.0f32, 4usize);
        let layout = linear_probe(n, pitch);
        let shift = 15.0;
        let data: Vec<f32> = (0..n).flat_map(|c| std::iter::repeat_n(field(c as f32 * pitch, shift), samples)).collect();

        // Zero drift is (nearly) the identity.
        let same = correct_traces_drift_kriging(&data, n, samples, 0, &layout, &constant_drift(0.0), 30_000.0, 20.0, 60.0).unwrap();
        let err = same.iter().zip(&data).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 2.0, "identity error {err} µV");

        // Sampling the moved field at site + drift undoes the shift away from the probe ends.
        let fixed = correct_traces_drift_kriging(&data, n, samples, 0, &layout, &constant_drift(shift), 30_000.0, 20.0, 60.0).unwrap();
        for c in 4..n - 4 {
            let want = field(c as f32 * pitch, 0.0);
            let got = fixed[c * samples];
            assert!((got - want).abs() < 5.0, "channel {c}: {got} vs {want}");
        }
    }

    #[test]
    fn missing_site_is_an_error() {
        let layout = linear_probe(4, 20.0);
        let batch = SnippetBatch::from_raw_parts(vec![0.0; 6], 1, 2, 3, 1, vec![0], vec![0], vec![0.0], vec![0, 9]);
        assert!(correct_snippet_batch_drift_kriging(&batch, &layout, &constant_drift(1.0), 30_000.0, 20.0).is_err());
    }

    #[test]
    fn device_trace_correction_matches_host() {
        use cubecl::prelude::*;
        use dsp_base::core::buffer;
        use dsp_core::compute::{ComputeTarget, ComputeTask};

        let (n, pitch, samples, fs) = (16usize, 20.0f32, 300usize, 1_000.0f64);
        let mut layout = linear_probe(n, pitch);
        layout.contacts[3].enabled = false; // a channel without a site keeps its samples
        // Drift ramps 0 → 0.5 µm across the chunk: several 0.1 µm runs
        let drift = DriftEstimate { time_bin_centers_sec: vec![0.0, 0.3], drift_um: vec![0.0, 0.5], ..constant_drift(0.0) };
        let data: Vec<f32> = (0..n * samples).map(|i| field((i / samples) as f32 * pitch, (i % samples) as f32 * 0.05)).collect();
        let host = TraceKriging::new(&layout, 20.0, 60.0).correct(&data, n, samples, 0, &drift, fs).unwrap();

        struct Task<'a>(&'a [f32], &'a SensorLayout, &'a DriftEstimate, usize, usize, f64);
        impl ComputeTask for Task<'_> {
            type Output = Vec<f32>;
            fn run(self, client: Client) -> Vec<f32> {
                let (input, output) = (buffer::upload(&client, self.0), buffer::empty::<f32>(&client, self.0.len()));
                let mut kriging = TraceKriging::new(self.1, 20.0, 60.0);
                kriging.correct_in_vram::<f32>(&client, &input, &output, self.3, self.4, 0, self.2, self.5).unwrap();
                buffer::download::<f32>(&client, output)
            }
        }
        for target in ComputeTarget::available() {
            let device = target.run(Task(&data, &layout, &drift, n, samples, fs)).expect("runtime");
            let err = device.iter().zip(&host).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-3, "device vs host {err}");
        }
    }
}
