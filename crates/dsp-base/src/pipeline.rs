use cubecl::prelude::*;
use crate::filter::{
    design_notch_coeffs, execute_notch,
    design_butterworth_bandpass_4th, execute_bandpass,
    execute_median_9p,
};
use crate::spatial::execute_direct_car;
use crate::math::execute_scaling;

/// Individual processing stage within an in-VRAM DSP pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Linear scaling and offset: $y = \alpha x + \beta$
    Scale { alpha: f32, beta: f32 },
    /// Constant baseline subtraction: $y = x - \text{baseline}$
    SubtractBaseline { baseline_uv: f32 },
    /// 2nd-order narrow-band notch filter (e.g. 50 Hz or 60 Hz electrical hum)
    Notch { freq_hz: f64, q: f64 },
    /// 4th-order cascaded Butterworth bandpass filter (e.g. 300–6,000 Hz AP band)
    Bandpass { low_hz: f64, high_hz: f64 },
    /// Common Average Referencing across all channels
    CommonAverageReference,
    /// 9-point branchless sorting network median filter
    Median9p,
}

/// In-VRAM DSP Pipeline Engine.
/// Chains an arbitrary sequence of filter and math stages directly in device memory
/// using double-buffered (ping-pong) VRAM allocation, guaranteeing zero host PCIe round-trips.
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn with_stages(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    pub fn add(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Executes all configured stages in sequence on `input_handle` in VRAM.
    /// Uses two ping-pong buffers so memory overhead is strictly $2 \times$ the buffer size regardless of stage count.
    pub fn execute<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input_handle: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
        sample_rate: f64,
        is_cpu: bool,
    ) -> cubecl::server::Handle {
        if self.stages.is_empty() {
            return input_handle.clone();
        }

        let buffer_bytes = channels * samples * std::mem::size_of::<f32>();
        let buf_ping = client.empty(buffer_bytes);
        let buf_pong = client.empty(buffer_bytes);

        let mut current_in = input_handle;
        let mut current_out = &buf_ping;
        let mut is_ping_out = true;

        for stage in &self.stages {
            match stage {
                PipelineStage::Scale { alpha, beta } => {
                    execute_scaling::<R>(
                        client,
                        current_in,
                        current_out,
                        channels * samples,
                        *alpha,
                        *beta,
                        is_cpu,
                    );
                }
                PipelineStage::SubtractBaseline { baseline_uv } => {
                    execute_scaling::<R>(
                        client,
                        current_in,
                        current_out,
                        channels * samples,
                        1.0,
                        -*baseline_uv,
                        is_cpu,
                    );
                }
                PipelineStage::Notch { freq_hz, q } => {
                    let coeffs = design_notch_coeffs(*freq_hz, sample_rate, *q);
                    execute_notch::<R>(
                        client,
                        current_in,
                        current_out,
                        coeffs,
                        channels,
                        samples,
                        is_cpu,
                    );
                }
                PipelineStage::Bandpass { low_hz, high_hz } => {
                    let coeffs = design_butterworth_bandpass_4th(*low_hz, *high_hz, sample_rate);
                    execute_bandpass::<R>(
                        client,
                        current_in,
                        current_out,
                        coeffs,
                        channels,
                        samples,
                        is_cpu,
                    );
                }
                PipelineStage::CommonAverageReference => {
                    execute_direct_car::<R>(
                        client,
                        current_in,
                        current_out,
                        channels,
                        samples,
                        is_cpu,
                    );
                }
                PipelineStage::Median9p => {
                    execute_median_9p::<R>(
                        client,
                        current_in,
                        current_out,
                        channels,
                        samples,
                        is_cpu,
                    );
                }
            }

            // Ping-pong buffer swap
            if is_ping_out {
                current_in = &buf_ping;
                current_out = &buf_pong;
                is_ping_out = false;
            } else {
                current_in = &buf_pong;
                current_out = &buf_ping;
                is_ping_out = true;
            }
        }

        current_in.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn test_pipeline_chained_wgpu() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let channels = 32;
        let samples = 500;
        let total = channels * samples;

        let input_data = vec![100.0f32; total];
        let input_bytes = f32::as_bytes(&input_data);
        let in_handle = client.create_from_slice(input_bytes);

        // Build a 3-stage pipeline: Scale -> CAR -> Notch
        let mut pipeline = Pipeline::new();
        pipeline
            .add(PipelineStage::Scale { alpha: 0.195, beta: 0.0 })
            .add(PipelineStage::CommonAverageReference)
            .add(PipelineStage::Notch { freq_hz: 60.0, q: 30.0 });

        assert_eq!(pipeline.len(), 3);

        let out_handle = pipeline.execute::<WgpuRuntime>(
            &client,
            &in_handle,
            channels,
            samples,
            30000.0,
            false,
        );

        let out_bytes = client.read_one_unchecked(out_handle);
        let out_slice = f32::from_bytes(&out_bytes);

        assert_eq!(out_slice.len(), total);
        assert!(!out_slice[0].is_nan());
    }
}
