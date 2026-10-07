use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Drift-run kriging of a `[channels, samples]` chunk: sample `t` lies in run `r`
/// (`run_starts[r] ≤ t < run_starts[r + 1]`), whose weight set is `slot = run_slots[r]`;
/// `output[o, t] = Σ_k values[slot, o, k] · input[indices[slot, o, k], t]` (ELLPACK rows of
/// `width` entries, padding weight 0). One unit per `(output channel, sample)`
/// ([`dsp_core::compute::LaunchGeometry::channels_samples`]). Runs are few per chunk (drift
/// changes slowly), so they are scanned linearly.
#[cube(launch)]
pub fn kriging_runs_kernel<F: Float>(
    input: &[F],
    values: &[F],
    indices: &[u32],
    run_starts: &[u32],
    run_slots: &[u32],
    output: &mut [F],
    num_channels: u32,
    num_samples: u32,
    width: u32,
    num_runs: u32,
) {
    let t = sample_position();
    let out_ch = channel_position();
    if t < num_samples && out_ch < num_channels {
        let mut r = 0u32;
        while r + 1u32 < num_runs && run_starts[(r + 1u32) as usize] <= t {
            r += 1u32;
        }
        let row = (run_slots[r as usize] * num_channels + out_ch) * width;
        let mut acc = F::new(0.0f32);
        let mut k = 0u32;
        while k < width {
            let in_ch = indices[(row + k) as usize];
            acc += values[(row + k) as usize] * input[(in_ch * num_samples + t) as usize];
            k += 1u32;
        }
        output[(out_ch * num_samples + t) as usize] = acc;
    }
}
