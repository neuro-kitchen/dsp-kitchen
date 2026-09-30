pub mod design;
pub mod iir;
pub mod fir;
pub mod non_linear;
pub mod template;

// Convenient flat re-exports for common workflows
pub use design::{FilterBand, FilterDesign, FilterError, FilterMode, FilterSpec, Section, Sos};
pub use iir::{DeviceFilter, execute_filter};
pub use fir::execute_fir;
pub use non_linear::{execute_median_9p, execute_teager_kaiser};
pub use template::{TemplateFilter, subtract_template_1d, subtract_template_multichannel};

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::prelude::*;

    #[test]
    fn test_median_filter_9p_wgpu() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let channels = 1;
        let samples = 20;
        let total = channels * samples;

        let mut input_data = vec![10.0f32; total];
        // Inject a single-sample transient spike at sample 10
        input_data[10] = 500.0;

        let input_bytes = f32::as_bytes(&input_data);
        let input_handle = client.create_from_slice(input_bytes);
        let output_handle = client.empty(total * core::mem::size_of::<f32>());

        execute_median_9p::<WgpuRuntime>(
            &client,
            &input_handle,
            &output_handle,
            channels,
            samples,
        );

        let output_bytes = client.read_one_unchecked(output_handle);
        let output_slice = f32::from_bytes(&output_bytes);

        // The 500.0 spike should be eliminated by the 9-point median filter!
        assert_eq!(output_slice[10], 10.0);
    }
}
