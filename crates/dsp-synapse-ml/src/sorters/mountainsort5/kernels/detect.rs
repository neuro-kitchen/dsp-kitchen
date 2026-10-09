//! Kernels of [`crate::sorters::mountainsort5::detect`].

use cubecl::prelude::*;

/// Padding of a neighbourhood row (as [`dsp_synapse::features::ChannelNeighbourhoods`]).
pub const NO_CHANNEL: u32 = u32::MAX;

/// `x` as MountainSort compares it: detections are minima of `x` (`mode` 0, sign −1), of `−x`
/// (1, sign +1) or of `−|x|` (2, sign 0).
#[cube]
fn signed<F: Float>(x: F, mode: u32) -> F {
    let mut v = x;
    if mode == 1u32 {
        v = -x;
    }
    if mode == 2u32 {
        v = -F::abs(x);
    }
    v
}

/// One unit per candidate `i` (sample `cand_samples[i]` of channel `cand_channels[i]`): `keep[i] =
/// 0` when any sample of a neighbourhood channel (`table`, `[channels, max_neighbours]` padded
/// with [`NO_CHANNEL`]) within `±time_radius` and inside `lo..hi` is strictly lower, else 1. Values
/// are compared in the sign's sense (`x`, `−x` or `−|x|` by `mode`). Dispatched via
/// [`dsp_core::compute::LaunchGeometry::elementwise`].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn locally_exclusive_kernel<F: Float>(
    trace: &[F],
    cand_samples: &[u32],
    cand_channels: &[u32],
    table: &[u32],
    keep: &mut [u32],
    n_cand: u32,
    samples: u32,
    max_neighbours: u32,
    time_radius: u32,
    lo: u32,
    hi: u32,
    mode: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n_cand {
        let t = cand_samples[i as usize];
        let ch = cand_channels[i as usize];
        let v = signed::<F>(trace[(ch * samples + t) as usize], mode);
        let mut t0 = lo;
        if t > lo + time_radius {
            t0 = t - time_radius;
        }
        let t1 = u32::min(t + time_radius + 1u32, hi);
        let mut ok = 1u32;
        // Loops with explicit breaks: a short-circuit `&&` reading memory in a loop condition
        // fails SPIR-V validation (CubeCL 0.11)
        let mut k = 0u32;
        loop {
            if k >= max_neighbours || ok == 0u32 {
                break;
            }
            let nb = table[(ch * max_neighbours + k) as usize];
            if nb == NO_CHANNEL {
                break;
            }
            let mut tt = t0;
            while tt < t1 {
                if signed::<F>(trace[(nb * samples + tt) as usize], mode) < v {
                    ok = 0u32;
                }
                tt += 1u32;
            }
            k += 1u32;
        }
        keep[i as usize] = ok;
    }
}
