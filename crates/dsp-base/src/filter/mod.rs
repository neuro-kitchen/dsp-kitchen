pub mod iir;
pub mod fir;
pub mod non_linear;

// Convenient flat re-exports for common workflows
pub use iir::{
    BiquadCoeffs, execute_biquad, execute_cascaded_biquad_4th,
    design_notch_coeffs, execute_notch,
    BandpassCoeffs, design_butterworth_bandpass_4th, execute_bandpass,
};
pub use fir::execute_fir;
pub use non_linear::execute_median_9p;

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::prelude::*;

    #[test]
    fn test_notch_filter_wgpu() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let channels = 2;
        let samples = 500;
        let total = channels * samples;

        let coeffs = design_notch_coeffs(60.0, 30000.0, 30.0);

        let input_data = vec![1.0f32; total];
        let input_bytes = f32::as_bytes(&input_data);
        let input_handle = client.create_from_slice(input_bytes);
        let output_handle = client.empty(total * core::mem::size_of::<f32>());

        execute_notch::<WgpuRuntime>(
            &client,
            &input_handle,
            &output_handle,
            coeffs,
            channels,
            samples,
            false,
        );

        let output_bytes = client.read_one_unchecked(output_handle);
        let output_slice = f32::from_bytes(&output_bytes);

        assert_eq!(output_slice.len(), total);
        assert!(!output_slice[0].is_nan());
    }

    #[test]
    fn test_bandpass_filter_wgpu() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let channels = 4;
        let samples = 1000;
        let total = channels * samples;

        let bp_coeffs = design_butterworth_bandpass_4th(300.0, 6000.0, 30000.0);

        let input_data = vec![1.0f32; total];
        let input_bytes = f32::as_bytes(&input_data);
        let input_handle = client.create_from_slice(input_bytes);
        let output_handle = client.empty(total * core::mem::size_of::<f32>());

        execute_bandpass::<WgpuRuntime>(
            &client,
            &input_handle,
            &output_handle,
            bp_coeffs,
            channels,
            samples,
            false,
        );

        let output_bytes = client.read_one_unchecked(output_handle);
        let output_slice = f32::from_bytes(&output_bytes);

        assert_eq!(output_slice.len(), total);
        assert!(!output_slice[0].is_nan());
    }

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
            false,
        );

        let output_bytes = client.read_one_unchecked(output_handle);
        let output_slice = f32::from_bytes(&output_bytes);

        // The 500.0 spike should be eliminated by the 9-point median filter!
        assert_eq!(output_slice[10], 10.0);
    }
}
