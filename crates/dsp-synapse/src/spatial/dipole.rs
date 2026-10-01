//! Current Dipole Source Localization (`dipole.rs`).
//!
//! Fits a 3D extracellular current dipole source position $\mathbf{r}_0 = (x, y, z)$ and dipole
//! moment vector $\mathbf{p} = (p_x, p_y, p_z)$ to signed multi-channel potentials:
//! $$\hat{V}_k(\mathbf{r}_0, \mathbf{p}) = \frac{\mathbf{p} \cdot (\mathbf{r}_k - \mathbf{r}_0)}{\|\mathbf{r}_k - \mathbf{r}_0\|^3}$$
//! using Golub-Pereyra variable projection (exact 3x3 regularized linear solve for $\mathbf{p}(\mathbf{r}_0)$
//! at each candidate $\mathbf{r}_0$) combined with damped coordinate-wise Gauss-Newton steps.

use dsp_core::SensorLayout;
use serde::{Deserialize, Serialize};
use crate::core::{PeakLocalizer, SnippetBatch};
use super::center_of_mass::{localize_spike_center_of_mass, waveform_peak_to_peak};

/// Estimated 3D dipole position (`um`) and dipole moment vector $\mathbf{p}$ (`uV * um^2`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DipoleEstimate {
    pub position_um: [f32; 3],
    pub moment: [f32; 3],
    pub residual_error: f32,
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        let mut max_v = a[col][col].abs();
        for row in (col + 1)..3 {
            let v = a[row][col].abs();
            if v > max_v {
                max_v = v;
                pivot = row;
            }
        }
        if max_v < 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let diag = a[col][col];
        for c in col..3 {
            a[col][c] /= diag;
        }
        b[col] /= diag;
        for row in 0..3 {
            if row != col {
                let factor = a[row][col];
                for c in col..3 {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    Some(b)
}

/// Fits a 3D current dipole `(position_um, moment)` to observed multi-channel potentials `potentials_uv`.
pub fn localize_spike_dipole(
    channel_ids: &[usize],
    potentials_uv: &[f32],
    layout: &SensorLayout,
    max_iters: usize,
) -> DipoleEstimate {
    let abs_amps: Vec<f32> = potentials_uv.iter().map(|&v| v.abs()).collect();
    let init_com = localize_spike_center_of_mass(channel_ids, &abs_amps, layout, 2.0);

    let mut sites = Vec::with_capacity(channel_ids.len());
    let mut obs = Vec::with_capacity(channel_ids.len());
    for (&ch, &v) in channel_ids.iter().zip(potentials_uv.iter()) {
        if let Ok(s) = layout.get_site(ch) {
            sites.push([
                s.position.x_um as f64,
                s.position.y_um as f64,
                s.position.z_um as f64,
            ]);
            obs.push(v as f64);
        }
    }

    if sites.len() < 4 {
        return DipoleEstimate {
            position_um: init_com,
            moment: [0.0, 0.0, 0.0],
            residual_error: 0.0,
        };
    }

    // Variable projection: for any candidate r0 = [x, y, z], solve min_p ||G(r0) p - v||^2 + reg ||p||^2
    let solve_moment_and_cost = |r0: [f64; 3]| -> ([f64; 3], f64) {
        let mut gtg = [[0.0f64; 3]; 3];
        let mut gtv = [0.0f64; 3];

        for (s, &v_obs) in sites.iter().zip(obs.iter()) {
            let dx = s[0] - r0[0];
            let dy = s[1] - r0[1];
            let dz = s[2] - r0[2];
            let r2 = dx * dx + dy * dy + dz * dz + 4.0;
            let r3 = r2 * r2.sqrt();
            let g = [dx / r3, dy / r3, dz / r3];

            for i in 0..3 {
                gtv[i] += g[i] * v_obs;
                for j in 0..3 {
                    gtg[i][j] += g[i] * g[j];
                }
            }
        }
        let trace = (gtg[0][0] + gtg[1][1] + gtg[2][2]) / 3.0;
        let reg = (trace * 1e-6).max(1e-12);
        for i in 0..3 {
            gtg[i][i] += reg;
        }

        let p = solve_3x3(gtg, gtv).unwrap_or([0.0, 0.0, 0.0]);
        let mut cost = 0.0f64;
        for (s, &v_obs) in sites.iter().zip(obs.iter()) {
            let dx = s[0] - r0[0];
            let dy = s[1] - r0[1];
            let dz = s[2] - r0[2];
            let r2 = dx * dx + dy * dy + dz * dz + 4.0;
            let r3 = r2 * r2.sqrt();
            let pred = (p[0] * dx + p[1] * dy + p[2] * dz) / r3;
            let err = pred - v_obs;
            cost += err * err;
        }
        (p, cost)
    };

    let mut pos = [init_com[0] as f64, init_com[1] as f64, 20.0f64];
    let (mut moment, mut best_cost) = solve_moment_and_cost(pos);
    let mut step_size = 8.0f64;

    for _iter in 0..max_iters.max(1) {
        let mut improved = false;
        for axis in 0..3 {
            for &dir in &[-1.0f64, 1.0f64] {
                let mut cand = pos;
                cand[axis] += dir * step_size;
                if axis == 2 {
                    cand[2] = cand[2].clamp(2.0, 250.0);
                }
                let (cand_p, cand_cost) = solve_moment_and_cost(cand);
                if cand_cost < best_cost {
                    pos = cand;
                    moment = cand_p;
                    best_cost = cand_cost;
                    improved = true;
                }
            }
        }
        if !improved {
            step_size *= 0.5;
            if step_size < 0.05 {
                break;
            }
        }
    }

    DipoleEstimate {
        position_um: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
        moment: [moment[0] as f32, moment[1] as f32, moment[2] as f32],
        residual_error: best_cost.sqrt() as f32,
    }
}

/// Dipole Localizer implementing [`PeakLocalizer`].
#[derive(Debug, Clone, Copy)]
pub struct DipoleLocalizer {
    pub max_iterations: usize,
}

impl Default for DipoleLocalizer {
    fn default() -> Self {
        Self { max_iterations: 40 }
    }
}

impl PeakLocalizer for DipoleLocalizer {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> dsp_core::DspResult<Vec<[f32; 3]>> {
        let mut out = Vec::with_capacity(batch.num_spikes);
        let mut signed_peaks = vec![0.0f32; batch.num_channels];
        let center_t = batch.num_samples / 2;

        for i in 0..batch.num_spikes {
            for k in 0..batch.num_channels {
                let ch_wave = batch.channel_slice(i, k);
                signed_peaks[k] = if !ch_wave.is_empty() {
                    ch_wave[center_t.min(ch_wave.len() - 1)]
                } else {
                    waveform_peak_to_peak(ch_wave)
                };
            }
            let ch_ids = batch.spike_channel_ids(i);
            let est = localize_spike_dipole(ch_ids, &signed_peaks, layout, self.max_iterations);
            out.push(est.position_um);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::neuropixels_2_0;

    #[test]
    fn test_dipole_localizer_recovers_3d_position_and_orientation() {
        let layout = neuropixels_2_0();
        let true_pos = [16.0f32, 30.0, 25.0];
        let true_p = [500.0f32, -2500.0, 800.0];

        let ch_ids: Vec<usize> = (0..10).collect();
        let mut potentials = Vec::with_capacity(ch_ids.len());
        for &ch in &ch_ids {
            let s = layout.get_site(ch).unwrap();
            let dx = s.position.x_um - true_pos[0];
            let dy = s.position.y_um - true_pos[1];
            let dz = s.position.z_um - true_pos[2];
            let r2 = dx * dx + dy * dy + dz * dz + 4.0;
            let r3 = r2 * r2.sqrt();
            let v = (true_p[0] * dx + true_p[1] * dy + true_p[2] * dz) / r3;
            potentials.push(v);
        }

        let est = localize_spike_dipole(&ch_ids, &potentials, &layout, 60);
        assert!((est.position_um[0] - true_pos[0]).abs() < 3.0, "x={}", est.position_um[0]);
        assert!((est.position_um[1] - true_pos[1]).abs() < 3.0, "y={}", est.position_um[1]);
        assert!((est.position_um[2] - true_pos[2]).abs() < 4.0, "z={}", est.position_um[2]);
    }
}
