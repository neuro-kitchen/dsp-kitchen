//! Spatial Gaussian-Process / Kriging Drift Interpolation (`kriging.rs`).
//!
//! Computes spatial covariance weights $W = K_{\text{target}, \text{source}} (K_{\text{source}, \text{source}} + \lambda I)^{-1}$
//! to interpolate multi-channel waveforms onto drift-corrected electrode positions.

use dsp_core::SensorLayout;
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

/// Applies spatial Kriging interpolation to a `SnippetBatch` to compensate for vertical probe drift $d(t)$.
pub fn correct_snippet_batch_drift_kriging(
    batch: &SnippetBatch,
    layout: &SensorLayout,
    drift: &DriftEstimate,
    sample_rate_hz: f64,
    sigma_um: f32,
) -> SnippetBatch {
    let n = batch.num_spikes;
    let k = batch.num_channels;
    let t = batch.num_samples;
    if n == 0 || k == 0 || t == 0 {
        return batch.clone();
    }

    let mut corrected_data = vec![0.0f32; n * k * t];
    let mut source_xy = vec![[0.0f32; 2]; k];
    let mut target_xy = vec![[0.0f32; 2]; k];

    for i in 0..n {
        let t_sec = (batch.center_samples[i] as f64) / sample_rate_hz.max(1.0);
        let dy_drift = drift.interpolate_drift_at(t_sec);
        let ch_ids = batch.spike_channel_ids(i);

        for (idx, &ch) in ch_ids.iter().enumerate() {
            let (x, y) = layout
                .get_site(ch)
                .map(|s| (s.position.x_um, s.position.y_um))
                .unwrap_or((0.0, 0.0));
            source_xy[idx] = [x, y];
            target_xy[idx] = [x, y + dy_drift];
        }

        let w = compute_kriging_weight_matrix(&source_xy, &target_xy, sigma_um, 1e-2);
        let in_snip = batch.snippet_slice(i);
        let out_off = i * k * t;

        // Multiply [K, K] weight matrix by [K, T] snippet
        for r in 0..k {
            for c in 0..k {
                let weight = w[r * k + c];
                let src_row = &in_snip[c * t..(c + 1) * t];
                let dst_row = &mut corrected_data[out_off + r * t..out_off + (r + 1) * t];
                for s in 0..t {
                    dst_row[s] += weight * src_row[s];
                }
            }
        }
    }

    SnippetBatch::from_raw_parts(
        corrected_data,
        n,
        k,
        t,
        batch.primary_channels.clone(),
        batch.center_samples.clone(),
        batch.subsample_offsets.clone(),
        batch.channel_ids.clone(),
    )
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
}
