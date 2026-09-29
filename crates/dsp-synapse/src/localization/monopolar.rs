//! Monopolar Point-Source Coulomb Triangulation (`monopolar.rs`).
//!
//! Solves the non-linear least-squares problem for a physical current monopole:
//! $$\hat{V}_k(x, y, z, \alpha) = \frac{\alpha}{\sqrt{(x - x_k)^2 + (y - y_k)^2 + (z - z_k)^2}}$$
//! using damped Gauss-Newton (Levenberg-Marquardt) iterations initialized at the Center-of-Mass.

use dsp_core::SensorLayout;
use crate::extraction::SnippetBatch;
use crate::traits::PeakLocalizer;
use super::center_of_mass::{localize_spike_center_of_mass, waveform_peak_to_peak};

/// Solves a $4 \times 4$ linear system $A \mathbf{x} = \mathbf{b}$ via Gaussian elimination with partial pivoting.
fn solve_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..4 {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        let diag = a[col][col];
        for c in col..4 {
            a[col][c] /= diag;
        }
        b[col] /= diag;

        for row in 0..4 {
            if row != col {
                let factor = a[row][col];
                for c in col..4 {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    Some(b)
}

/// Estimates 3D source position `[x_um, y_um, z_um]` and monopole current magnitude `alpha`
/// for a single spike via Levenberg-Marquardt optimization.
pub fn localize_spike_monopolar(
    channel_ids: &[usize],
    ptp_amplitudes: &[f32],
    layout: &SensorLayout,
    max_iters: usize,
) -> ([f32; 3], f32) {
    let init_com = localize_spike_center_of_mass(channel_ids, ptp_amplitudes, layout, 2.0);
    let mut sites = Vec::with_capacity(channel_ids.len());
    let mut obs = Vec::with_capacity(channel_ids.len());

    for (&ch, &amp) in channel_ids.iter().zip(ptp_amplitudes.iter()) {
        if let Ok(s) = layout.get_site(ch) {
            sites.push([
                s.position.x_um as f64,
                s.position.y_um as f64,
                s.position.z_um as f64,
            ]);
            obs.push(amp.max(1e-3) as f64);
        }
    }

    if sites.len() < 3 {
        return (init_com, ptp_amplitudes.first().copied().unwrap_or(100.0));
    }

    let mut x = init_com[0] as f64;
    let mut y = init_com[1] as f64;
    let mut z = 15.0f64; // Initial perpendicular offset from electrode plane
    let mut lambda = 1e-2f64;

    // Variable Projection (Golub-Pereyra): for any (px, py, pz), the optimal linear
    // scale alpha*(px, py, pz) has exact closed form: sum(v_obs / r) / sum(1 / r^2).
    let optimal_alpha = |px: f64, py: f64, pz: f64| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (s, &v_obs) in sites.iter().zip(obs.iter()) {
            let dx = px - s[0];
            let dy = py - s[1];
            let dz = pz - s[2];
            let inv_r2 = 1.0 / (dx * dx + dy * dy + dz * dz + 1.0);
            let inv_r = inv_r2.sqrt();
            num += v_obs * inv_r;
            den += inv_r2;
        }
        (num / den.max(1e-12)).max(1.0)
    };

    let mut alpha = optimal_alpha(x, y, z);

    let eval_cost = |px: f64, py: f64, pz: f64, p_alpha: f64| -> f64 {
        let mut cost = 0.0f64;
        for (s, &v_obs) in sites.iter().zip(obs.iter()) {
            let dx = px - s[0];
            let dy = py - s[1];
            let dz = pz - s[2];
            let r = (dx * dx + dy * dy + dz * dz + 1.0).sqrt();
            let pred = p_alpha / r;
            let err = pred - v_obs;
            cost += err * err;
        }
        cost
    };

    let mut current_cost = eval_cost(x, y, z, alpha);

    for _ in 0..max_iters {
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];

        for (s, &v_obs) in sites.iter().zip(obs.iter()) {
            let dx = x - s[0];
            let dy = y - s[1];
            let dz = z - s[2];
            let r2 = dx * dx + dy * dy + dz * dz + 1.0;
            let r = r2.sqrt();
            let r3 = r2 * r;

            let pred = alpha / r;
            let res = v_obs - pred;

            let j = [
                -alpha * dx / r3,
                -alpha * dy / r3,
                -alpha * dz / r3,
                1.0 / r,
            ];

            for r_idx in 0..4 {
                jtr[r_idx] += j[r_idx] * res;
                for c_idx in 0..4 {
                    jtj[r_idx][c_idx] += j[r_idx] * j[c_idx];
                }
            }
        }

        for d in 0..4 {
            jtj[d][d] += lambda * (jtj[d][d].max(1e-6));
        }

        let Some(step) = solve_4x4(jtj, jtr) else {
            break;
        };

        let cand_x = x + step[0].clamp(-25.0, 25.0);
        let cand_y = y + step[1].clamp(-25.0, 25.0);
        let cand_z = (z + step[2].clamp(-20.0, 20.0)).clamp(1.0, 250.0);
        let cand_alpha = optimal_alpha(cand_x, cand_y, cand_z);

        let cand_cost = eval_cost(cand_x, cand_y, cand_z, cand_alpha);
        if cand_cost < current_cost {
            x = cand_x;
            y = cand_y;
            z = cand_z;
            alpha = cand_alpha;
            let rel_imp = (current_cost - cand_cost) / current_cost.max(1e-12);
            current_cost = cand_cost;
            lambda = (lambda * 0.3).max(1e-8);
            if rel_imp < 1e-9 && current_cost < 1e-4 {
                break;
            }
        } else {
            lambda = (lambda * 4.0).min(1e4);
        }
    }

    ([x as f32, y as f32, z as f32], alpha as f32)
}

/// Monopolar Triangulation localizer implementing [`PeakLocalizer`].
#[derive(Debug, Clone, Copy)]
pub struct MonopolarTriangulator {
    pub max_iterations: usize,
}

impl Default for MonopolarTriangulator {
    fn default() -> Self {
        Self { max_iterations: 35 }
    }
}

impl PeakLocalizer for MonopolarTriangulator {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(batch.num_spikes);
        let mut ptp_buf = vec![0.0f32; batch.num_channels];

        for i in 0..batch.num_spikes {
            for k in 0..batch.num_channels {
                ptp_buf[k] = waveform_peak_to_peak(batch.channel_slice(i, k));
            }
            let ch_ids = batch.spike_channel_ids(i);
            let (coords, _alpha) =
                localize_spike_monopolar(ch_ids, &ptp_buf, layout, self.max_iterations);
            out.push(coords);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::neuropixels_2_0;

    #[test]
    fn test_monopolar_triangulation_recovers_3d_source() {
        let layout = neuropixels_2_0();
        // True source at (12.0, 22.0, 20.0) with current magnitude alpha = 2500.0
        let true_pos = [12.0f32, 22.0, 20.0];
        let true_alpha = 2500.0f32;

        let ch_ids = vec![0, 1, 2, 3, 4, 5];
        let mut amps = Vec::new();
        for &ch in &ch_ids {
            let s = layout.get_site(ch).unwrap();
            let dx = true_pos[0] - s.position.x_um;
            let dy = true_pos[1] - s.position.y_um;
            let dz = true_pos[2] - s.position.z_um;
            let r = (dx * dx + dy * dy + dz * dz + 1.0).sqrt();
            amps.push(true_alpha / r);
        }

        let (est_pos, _est_alpha) = localize_spike_monopolar(&ch_ids, &amps, &layout, 50);
        assert!(
            (est_pos[0] - true_pos[0]).abs() < 1.5,
            "x: {} vs {}",
            est_pos[0],
            true_pos[0]
        );
        assert!(
            (est_pos[1] - true_pos[1]).abs() < 1.5,
            "y: {} vs {}",
            est_pos[1],
            true_pos[1]
        );
        assert!(
            (est_pos[2] - true_pos[2]).abs() < 2.0,
            "z: {} vs {}",
            est_pos[2],
            true_pos[2]
        );
    }
}
