pub mod design;
pub mod iir;
pub mod fir;
pub mod non_linear;
pub mod template;

// Convenient flat re-exports for common workflows
pub use design::{FilterBand, FilterDesign, FilterError, FilterMode, FilterSpec, FilterStart, Section, Sos};
pub use iir::{DeviceFilter, PassLayout, execute_filter};
pub use fir::{
    execute_fir, execute_fir_centered,
    execute_gaussian_smooth, gaussian_kernel_1d, gaussian_smooth_1d, FIR_DEFAULT_EDGE, GAUSSIAN_DEFAULT_EDGE,
};
pub use non_linear::{execute_median, execute_median_9p, execute_teager_kaiser};
pub use template::{TemplateFilter, subtract_template_1d, subtract_template_multichannel};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{buffer, EdgeMode};
    use cubecl::prelude::*;

    /// Host reference: running median with zero padding (`scipy.signal.medfilt`).
    fn medfilt(x: &[f32], width: usize) -> Vec<f32> {
        let r = width as isize / 2;
        (0..x.len() as isize)
            .map(|t| {
                let mut w: Vec<f32> = (t - r..=t + r).map(|p| if p < 0 || p >= x.len() as isize { 0.0 } else { x[p as usize] }).collect();
                w.sort_by(f32::total_cmp);
                w[w.len() / 2]
            })
            .collect()
    }

    fn median_widths(client: &Client) {
        let samples = 40;
        let x: Vec<f32> = (0..samples).map(|i| ((i * 37 % 11) as f32) - 5.0 + if i == 20 { 500.0 } else { 0.0 }).collect();
        let input = buffer::upload(client, &x);
        for width in [1usize, 3, 7, 9, 15] {
            let output = buffer::empty::<f32>(client, samples);
            execute_median::<f32>(client, &input, &output, 1, samples, width, EdgeMode::Zeros);
            assert_eq!(buffer::download::<f32>(client, output), medfilt(&x, width), "{} width {width}", client.name());
        }
    }
    runtime_test!(test_median_matches_medfilt, median_widths);
}
