//! Spatial Gaussian-Process / Kriging Drift Interpolation (`kriging.rs`).
//!
//! Computes spatial covariance weights $W = K_{\text{target}, \text{source}} (K_{\text{source}, \text{source}} + \lambda I)^{-1}$
//! to interpolate multi-channel waveforms onto drift-corrected electrode positions.

use std::collections::HashMap;

use dsp_core::{DspError, DspResult, SensorLayout};
use crate::extraction::SnippetBatch;
use super::drift_map::DriftEstimate;

/// Solves a symmetric positive-definite linear system $A X = B$ where $A$ is `[k, k]` and $B$ is `[k, m]`
/// using Gauss-Jordan elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<f64>, b: &[f64], k: usize, m: usize) -> Vec<f64> {
    let mut x = b.to_vec();

    for col in 0..k {
        let mut pivot = col;
        let mut max_v = a[col * k + col].abs();
        for row in (col + 1)..k {
            let v = a[row * k + col].abs();
            if v > max_v {
                max_v = v;
                pivot = row;
            }
        }
        if max_v < 1e-12 {
            continue;
        }
        if pivot != col {
            for c in 0..k {
                a.swap(col * k + c, pivot * k + c);
            }
            for c in 0..m {
                x.swap(col * m + c, pivot * m + c);
            }
        }

        let diag = a[col * k + col];
        for c in col..k {
            a[col * k + c] /= diag;
        }
        for c in 0..m {
            x[col * m + c] /= diag;
        }

        for row in 0..k {
            if row != col {
                let factor = a[row * k + col];
                for c in col..k {
                    a[row * k + c] -= factor * a[col * k + c];
                }
                for c in 0..m {
                    x[row * m + c] -= factor * x[col * m + c];
                }
            }
        }
    }
    x
}

/// Computes the $[K \times K]$ spatial Kriging weight matrix mapping observed channels
/// at `source_xy` to drift-shifted target positions `target_xy`.
pub fn compute_kriging_weight_matrix(
    source_xy: &[[f32; 2]],
    target_xy: &[[f32; 2]],
    sigma_um: f32,
    regularization: f32,
) -> Vec<f32> {
    let k = source_xy.len();
    let m = target_xy.len();
    if k == 0 || m == 0 {
        return Vec::new();
    }

    let two_sigma_sq = (2.0 * (sigma_um as f64) * (sigma_um as f64)).max(1.0);
    let reg = regularization.max(1e-5) as f64;

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
    let wt = solve_linear_system(k_ss, &k_st, k, m);
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            w[r * k + c] = wt[c * m + r] as f32;
        }
    }
    w
}

/// Drift quantization for weight caching (µm).
const DRIFT_QUANTUM_UM: f32 = 0.1;

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
                let w = compute_kriging_weight_matrix(&source_xy, &target_xy, sigma_um, 1e-2);
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
        batch.primary_channels.clone(),
        batch.center_samples.clone(),
        batch.subsample_offsets.clone(),
        batch.channel_ids.clone(),
    ))
}

/// Drift-corrects a `[channels, samples]` chunk (rows = recording channels, first sample at global
/// `start_sample`): channel `c` becomes the signal at `site(c) + (0, d(t))`, kriged from every
/// enabled site within `radius_um` of that target. Drift is evaluated per sample and quantized to
/// 0.1 µm; weights are cached per drift step. Channels without a site keep their samples; errors if
/// a site's channel is outside the chunk.
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
    assert_eq!(data.len(), channels * samples);
    let sites: Vec<(usize, [f32; 2])> = layout
        .contacts
        .iter()
        .filter(|s| s.enabled)
        .map(|s| (s.channel_id, [s.position.x_um, s.position.y_um]))
        .collect();
    if let Some((c, _)) = sites.iter().find(|(c, _)| *c >= channels) {
        return Err(DspError::InvalidConfig(format!("site channel {c} is outside the {channels}-channel chunk")));
    }

    // Per drift step: for each site, (source rows, weights).
    let mut cache: HashMap<i64, Vec<(Vec<usize>, Vec<f32>)>> = HashMap::new();
    let weights_for = |key: i64| -> Vec<(Vec<usize>, Vec<f32>)> {
        let dy = key as f32 * DRIFT_QUANTUM_UM;
        sites
            .iter()
            .map(|&(_, [x, y])| {
                let target = [x, y + dy];
                let near: Vec<&(usize, [f32; 2])> = sites
                    .iter()
                    .filter(|(_, p)| ((p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2)).sqrt() <= radius_um)
                    .collect();
                let src: Vec<[f32; 2]> = near.iter().map(|(_, p)| *p).collect();
                let w = compute_kriging_weight_matrix(&src, &[target], sigma_um, 1e-2);
                (near.iter().map(|(c, _)| *c).collect(), w)
            })
            .collect()
    };

    let mut out = data.to_vec();
    let mut s0 = 0usize;
    while s0 < samples {
        let key_at = |s: usize| quantize(drift.interpolate_drift_at((start_sample + s as u64) as f64 / sample_rate_hz));
        let key = key_at(s0);
        let mut s1 = s0 + 1;
        while s1 < samples && key_at(s1) == key {
            s1 += 1;
        }
        let weights = cache.entry(key).or_insert_with(|| weights_for(key));
        for ((ch, _), (src, w)) in sites.iter().zip(weights.iter()) {
            let dst = &mut out[ch * samples + s0..ch * samples + s1];
            dst.fill(0.0);
            for (&c, &wc) in src.iter().zip(w) {
                for (d, x) in dst.iter_mut().zip(&data[c * samples + s0..c * samples + s1]) {
                    *d += wc * x;
                }
            }
        }
        s0 = s1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kriging_identity_at_zero_drift() {
        let coords = vec![[0.0, 0.0], [0.0, 20.0], [16.0, 10.0], [16.0, 30.0]];
        let w = compute_kriging_weight_matrix(&coords, &coords, 25.0, 1e-4);
        assert_eq!(w.len(), 16);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((w[i * 4 + j] - expected).abs() < 1e-2);
            }
        }
    }

    fn linear_probe(n: usize, pitch: f32) -> SensorLayout {
        use dsp_core::layout::{Position3D, SensorSite};
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
        let batch = SnippetBatch::from_raw_parts(vec![0.0; 6], 1, 2, 3, vec![0], vec![0], vec![0.0], vec![0, 9]);
        assert!(correct_snippet_batch_drift_kriging(&batch, &layout, &constant_drift(1.0), 30_000.0, 20.0).is_err());
    }
}
