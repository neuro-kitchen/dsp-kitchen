pub mod kernels;
pub mod scaling;
pub mod baseline;
pub mod energy;

pub use scaling::execute_scaling;
pub use baseline::execute_baseline_subtract;
pub use energy::execute_neo;

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::prelude::*;

    #[test]
    fn test_scaling_wgpu_kernel() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let total = input_data.len();

        let input_bytes = f32::as_bytes(&input_data);
        let input_handle = client.create_from_slice(input_bytes);
        let output_handle = client.empty(total * core::mem::size_of::<f32>());

        execute_scaling::<WgpuRuntime>(
            &client,
            &input_handle,
            &output_handle,
            total,
            2.5,
            10.0,
            false,
        );

        let output_bytes = client.read_one_unchecked(output_handle);
        let output_slice = f32::from_bytes(&output_bytes);

        assert_eq!(output_slice[0], 1.0 * 2.5 + 10.0);
        assert_eq!(output_slice[1], 2.0 * 2.5 + 10.0);
        assert_eq!(output_slice[2], 3.0 * 2.5 + 10.0);
        assert_eq!(output_slice[3], 4.0 * 2.5 + 10.0);
    }
}
