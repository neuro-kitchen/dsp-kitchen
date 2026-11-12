use cubecl::prelude::*;
use dspk_core::compute::element_position;
use dspk_ops::core::DspFloat;

/// The kriging kernel of a drifted probe: `kyx[i, j] = exp(−‖(x_i, y_i − shift_i) − (x_j, y_j)‖² ·
/// scale)`, `scale = 1 / (2σ²)`. One unit per `(i, j)`.
#[cube(launch)]
pub fn kriging_kernel<F: DspFloat>(x: &[F], y: &[F], shifts: &[F], kyx: &mut [F], channels: u32, scale: F) {
    let unit = element_position() as u32;
    if unit < channels * channels {
        let i = unit / channels;
        let j = unit - i * channels;
        let dx = x[i as usize] - x[j as usize];
        let dy = y[i as usize] - shifts[i as usize] - y[j as usize];
        kyx[unit as usize] = F::exp(F::new(0.0f32) - (dx * dx + dy * dy) * scale);
    }
}
