use cubecl::prelude::*;

use crate::math::windows::{BLACKMAN_HARRIS_A0, BLACKMAN_HARRIS_A1, BLACKMAN_HARRIS_A2, BLACKMAN_HARRIS_A3, SINC_ZERO};

/// Un-normalized tap weight at `τ` for a window of half-width `half_width` (device form of the
/// host weight in [`fractional_delay_taps`]).
#[cube]
pub fn windowed_sinc_weight<F: Float>(tau: F, half_width: F) -> F {
    let pi = F::new(core::f32::consts::PI);
    let one = F::new(1.0f32);
    let norm = F::clamp(tau / half_width, F::new(-1.0f32), one);
    let two_pi_u = pi * (norm + one);
    let window = F::new(BLACKMAN_HARRIS_A0) - F::new(BLACKMAN_HARRIS_A1) * F::cos(two_pi_u)
        + F::new(BLACKMAN_HARRIS_A2) * F::cos(F::new(2.0f32) * two_pi_u)
        - F::new(BLACKMAN_HARRIS_A3) * F::cos(F::new(3.0f32) * two_pi_u);
    let mut s = one;
    if F::abs(tau) >= F::new(SINC_ZERO) {
        let pix = pi * tau;
        s = F::sin(pix) / pix;
    }
    s * window
}

/// One unit per shift: writes the `2·radius + 1` normalized weights of `shifts[i]` to
/// `taps[i · (2·radius + 1) ..]`.
#[cube(launch)]
pub fn fractional_delay_taps_kernel<F: Float>(shifts: &[F], taps: &mut [F], count: u32, #[comptime] radius: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let n = 2 * radius + 1;
        let shift = shifts[i as usize];
        let half_width = F::cast_from(radius);
        let base = i * n;
        let mut sum = F::new(0.0f32);
        #[unroll]
        for m in 0..n {
            let w = windowed_sinc_weight::<F>(F::cast_from(m) - half_width - shift, half_width);
            taps[(base + m) as usize] = w;
            sum += w;
        }
        #[unroll]
        for m in 0..n {
            let w = taps[(base + m) as usize];
            taps[(base + m) as usize] = w / sum;
        }
    }
}
