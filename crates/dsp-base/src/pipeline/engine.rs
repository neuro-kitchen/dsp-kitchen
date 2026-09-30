use cubecl::prelude::*;
use super::stage::PipelineStage;
use crate::filter::{
    design_notch_coeffs, execute_notch,
    design_butterworth_bandpass_4th, execute_bandpass,
    execute_median_9p, execute_teager_kaiser,
};
use crate::spatial::execute_direct_car;
use crate::math::{execute_scaling, execute_clamp};

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

    /// Computes the total filter settling / boundary margin (in samples) required by all stages
    /// in this pipeline at `sample_rate` Hz.
    pub fn settling_samples(&self, sample_rate: f64) -> usize {
        let mut max_iir = 0usize;
        let mut fir_stencil = 0usize;
        for stage in &self.stages {
            match stage {
                PipelineStage::Median9p | PipelineStage::TeagerKaiser => {
                    fir_stencil += stage.settling_samples(sample_rate);
                }
                _ => {
                    max_iir = max_iir.max(stage.settling_samples(sample_rate));
                }
            }
        }
        max_iir + fir_stencil
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
        self.execute_with_buffers::<R>(
            client,
            input_handle,
            &buf_ping,
            &buf_pong,
            channels,
            samples,
            sample_rate,
            is_cpu,
        )
    }

    /// Executes all configured stages using caller-provided persistent `buf_ping` and `buf_pong` VRAM handles.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_buffers<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input_handle: &cubecl::server::Handle,
        buf_ping: &cubecl::server::Handle,
        buf_pong: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
        sample_rate: f64,
        is_cpu: bool,
    ) -> cubecl::server::Handle {
        if self.stages.is_empty() {
            return input_handle.clone();
        }

        let mut current_in = input_handle;
        let mut current_out = buf_ping;
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
                PipelineStage::Clamp { min, max } => {
                    execute_clamp::<R>(
                        client,
                        current_in,
                        current_out,
                        channels * samples,
                        *min,
                        *max,
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
                PipelineStage::TeagerKaiser => {
                    execute_teager_kaiser::<R>(
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
                current_in = buf_ping;
                current_out = buf_pong;
                is_ping_out = false;
            } else {
                current_in = buf_pong;
                current_out = buf_ping;
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
