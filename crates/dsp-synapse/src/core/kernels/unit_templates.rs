//! Kernel of [`crate::core::template::UnitTemplateAccumulator`].

use cubecl::prelude::*;

/// One unit per `(unit u, channel c, sample t)` slot of `[units, channels, width]`: continues the
/// slot's Welford mean and `M₂` over the window's spikes of unit `u`
/// (`spike_order[unit_offsets[u]..unit_offsets[u + 1]]`), reading `data[c, sample − n_before + t]`
/// (`[channels, samples]`; the nearest edge sample past either end). `prev_counts[u]` spikes were
/// accumulated before, so the state carries over windows exactly.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_unit_templates_kernel<F: Float>(
    data: &[F],
    peak_samples: &[u32],
    spike_order: &[u32],
    unit_offsets: &[u32],
    prev_counts: &[u32],
    mean: &mut [F],
    m2: &mut [F],
    n_units: u32,
    channels: u32,
    samples: u32,
    n_before: u32,
    width: u32,
) {
    let q = ABSOLUTE_POS as u32;
    if q < n_units * channels * width {
        let t = q % width;
        let c = (q / width) % channels;
        let u = q / (width * channels);
        let start = unit_offsets[u as usize];
        let end = unit_offsets[(u + 1u32) as usize];
        let mut mu = mean[q as usize];
        let mut acc = m2[q as usize];
        let mut n = prev_counts[u as usize];
        let mut j = start;
        while j < end {
            let at = peak_samples[spike_order[j as usize] as usize] as i32 - n_before as i32 + t as i32;
            let mut s = 0u32;
            if at > 0i32 {
                s = u32::min(at as u32, samples - 1u32);
            }
            let x = data[(c * samples + s) as usize];
            n += 1u32;
            let delta = x - mu;
            mu += delta / F::cast_from(n);
            acc += delta * (x - mu);
            j += 1u32;
        }
        mean[q as usize] = mu;
        m2[q as usize] = acc;
    }
}
