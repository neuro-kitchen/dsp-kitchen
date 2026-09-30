//! Persistent VRAM workspace for streaming multi-chunk execution without repeated GPU allocations.

use cubecl::prelude::*;
use super::engine::Pipeline;

/// Pre-allocated VRAM ping-pong workspace for streaming chunks through a [`Pipeline`].
///
/// Reuses the same `buf_ping` and `buf_pong` device handles across all chunks of identical
/// shape `(channels, samples)`, only reallocating if the tail chunk has a different length.
pub struct PipelineWorkspace<R: Runtime> {
    client: ComputeClient<R>,
    pipeline: Pipeline,
    channels: usize,
    allocated_samples: usize,
    sample_rate: f64,
    is_cpu: bool,
    buf_ping: cubecl::server::Handle,
    buf_pong: cubecl::server::Handle,
}

impl<R: Runtime> PipelineWorkspace<R> {
    /// Creates a new persistent VRAM workspace pre-allocated for `(channels, initial_samples)`.
    pub fn new(
        client: ComputeClient<R>,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
        is_cpu: bool,
    ) -> Self {
        let bytes = (channels * initial_samples * std::mem::size_of::<f32>()).max(4);
        let buf_ping = client.empty(bytes);
        let buf_pong = client.empty(bytes);
        Self {
            client,
            pipeline,
            channels,
            allocated_samples: initial_samples,
            sample_rate,
            is_cpu,
            buf_ping,
            buf_pong,
        }
    }

    /// Reference to the underlying [`ComputeClient`].
    #[inline]
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Reference to the underlying [`Pipeline`].
    #[inline]
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Processes a `[channels, samples]` chunk in-VRAM using the pre-allocated ping-pong buffers
    /// and returns the resulting GPU [`cubecl::server::Handle`] **without** downloading it to host RAM.
    pub fn process_chunk_in_vram(&mut self, input: &[f32], samples: usize) -> cubecl::server::Handle {
        assert_eq!(input.len(), self.channels * samples);

        let in_bytes = f32::as_bytes(input);
        let in_handle = self.client.create_from_slice(in_bytes);

        if self.pipeline.is_empty() {
            return in_handle;
        }

        if samples != self.allocated_samples {
            let bytes = (self.channels * samples * std::mem::size_of::<f32>()).max(4);
            self.buf_ping = self.client.empty(bytes);
            self.buf_pong = self.client.empty(bytes);
            self.allocated_samples = samples;
        }

        self.pipeline.execute_with_buffers::<R>(
            &self.client,
            &in_handle,
            &self.buf_ping,
            &self.buf_pong,
            self.channels,
            samples,
            self.sample_rate,
            self.is_cpu,
        )
    }

    /// Processes a `[channels, samples]` chunk in-VRAM using the pre-allocated ping-pong buffers
    /// and writes the filtered output into `output` (length `channels * samples`).
    pub fn process_chunk(&mut self, input: &[f32], samples: usize, output: &mut [f32]) {
        assert_eq!(input.len(), self.channels * samples);
        assert_eq!(output.len(), self.channels * samples);

        if self.pipeline.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        let out_handle = self.process_chunk_in_vram(input, samples);
        let out_bytes = self.client.read_one_unchecked(out_handle);
        let out_slice = f32::from_bytes(&out_bytes);
        output.copy_from_slice(&out_slice[..output.len()]);
    }
}
