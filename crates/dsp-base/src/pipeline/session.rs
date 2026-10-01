//! Persistent VRAM workspace for streaming multi-chunk execution without repeated GPU allocations.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::engine::Pipeline;
use super::stage::PipelineStage;
use crate::filter::design::FilterError;
use crate::filter::iir::DeviceFilter;
use crate::filter::{execute_fir_centered, execute_median_9p, execute_teager_kaiser, gaussian_kernel_1d};
use crate::math::{execute_clamp, execute_scaling, execute_unpack_stored, stored_words};
use dsp_core::{DspError, DspResult, SampleFormat};
use crate::spatial::{execute_direct_car, execute_spatial_matrix_multiply};

/// How consecutive chunks relate to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    /// Every chunk is processed on its own (halo windows, random access). Filters start from the
    /// steady state of the chunk's first sample; forward-backward filters odd-pad both edges. The
    /// caller supplies halos of at least [`Pipeline::settling`].
    Independent,
    /// Chunks are consecutive pieces of one stream (live input, sequential reads). Filter section
    /// state is carried from chunk to chunk, so the output is exact with no halo. Only forward
    /// filters are allowed. Stencil stages (`Median9p`, `TeagerKaiser`) still see each chunk's
    /// edges on their own.
    Stateful,
}

enum Planned {
    Stage(PipelineStage),
    Filter { filter: DeviceFilter, state: Handle },
    SpatialMatrix { weights: Handle },
    CenteredFir { taps: Handle, radius: usize },
}

/// Pre-allocated VRAM workspace for running a [`Pipeline`] over many chunks.
///
/// Filter designs are made and uploaded once. Buffers only grow: a shorter tail chunk reuses them.
pub struct PipelineWorkspace<R: Runtime> {
    client: ComputeClient<R>,
    pipeline: Pipeline,
    plan: Vec<Planned>,
    mode: ChunkMode,
    started: bool,
    channels: usize,
    capacity_samples: usize,
    buf_ping: Handle,
    buf_pong: Handle,
    scratch: Handle,
    scratch_floats: usize,
    /// Per-channel gain and offset for stored chunks, and the buffer they are unpacked into.
    stored_scaling: Option<(Handle, Handle)>,
    unpacked: Option<Handle>,
}

impl<R: Runtime> PipelineWorkspace<R> {
    /// Workspace for independent chunks (see [`ChunkMode::Independent`]).
    pub fn new(
        client: ComputeClient<R>,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, FilterError> {
        Self::with_mode(client, pipeline, channels, initial_samples, sample_rate, ChunkMode::Independent)
    }

    /// Workspace for a continuous stream (see [`ChunkMode::Stateful`]); rejects forward-backward
    /// filters.
    pub fn new_stateful(
        client: ComputeClient<R>,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, FilterError> {
        Self::with_mode(client, pipeline, channels, initial_samples, sample_rate, ChunkMode::Stateful)
    }

    fn with_mode(
        client: ComputeClient<R>,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
        mode: ChunkMode,
    ) -> Result<Self, FilterError> {
        let mut plan = Vec::with_capacity(pipeline.len());
        for stage in pipeline.stages() {
            match stage {
                PipelineStage::Filter(spec) => {
                    let filter = DeviceFilter::new(&client, spec, sample_rate)?;
                    if mode == ChunkMode::Stateful && filter.mode() != crate::filter::FilterMode::Forward {
                        return Err(FilterError::ForwardBackwardOnLiveStream);
                    }
                    let state = client.empty((channels * filter.state_len() * 4).max(4));
                    plan.push(Planned::Filter { filter, state });
                }
                PipelineStage::SpatialWhitening(w) => {
                    assert_eq!(w.num_channels, channels, "SpatialWhitening channel mismatch");
                    let weights = client.create_from_slice(f32::as_bytes(&w.matrix));
                    plan.push(Planned::SpatialMatrix { weights });
                }
                PipelineStage::SurfaceLaplacian(lap) => {
                    assert_eq!(lap.num_channels, channels, "SurfaceLaplacian channel mismatch");
                    let weights = client.create_from_slice(f32::as_bytes(&lap.matrix));
                    plan.push(Planned::SpatialMatrix { weights });
                }
                PipelineStage::GaussianSmooth { sigma_samples } => {
                    let (taps_vec, radius) = gaussian_kernel_1d(*sigma_samples, 3.0);
                    let taps = client.create_from_slice(f32::as_bytes(&taps_vec));
                    plan.push(Planned::CenteredFir { taps, radius });
                }
                other => plan.push(Planned::Stage(other.clone())),
            }
        }
        let bytes = (channels * initial_samples * 4).max(4);
        let buf_ping = client.empty(bytes);
        let buf_pong = client.empty(bytes);
        let scratch = client.empty(4);
        let mut workspace = Self {
            client,
            pipeline,
            plan,
            mode,
            started: false,
            channels,
            capacity_samples: initial_samples,
            buf_ping,
            buf_pong,
            scratch,
            scratch_floats: 1,
            stored_scaling: None,
            unpacked: None,
        };
        workspace.reserve(initial_samples);
        Ok(workspace)
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

    #[inline]
    pub fn mode(&self) -> ChunkMode {
        self.mode
    }

    /// Forgets carried filter state; the next stateful chunk starts from its own steady state.
    pub fn reset(&mut self) {
        self.started = false;
    }

    fn reserve(&mut self, samples: usize) {
        if samples > self.capacity_samples {
            let bytes = (self.channels * samples * 4).max(4);
            self.buf_ping = self.client.empty(bytes);
            self.buf_pong = self.client.empty(bytes);
            self.unpacked = None;
            self.capacity_samples = samples;
        }
        let need = self
            .plan
            .iter()
            .map(|p| match p {
                Planned::Filter { filter, .. } => filter.scratch_len(self.channels, samples),
                Planned::Stage(_) | Planned::SpatialMatrix { .. } | Planned::CenteredFir { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        if need > self.scratch_floats {
            self.scratch = self.client.empty(need * 4);
            self.scratch_floats = need;
        }
    }

    /// Runs the pipeline on a `[channels, samples]` device buffer and returns a handle to the
    /// result. The handle views one of the workspace's buffers: read it before the next call.
    pub fn process_handle(&mut self, input: &Handle, samples: usize) -> Handle {
        if self.plan.is_empty() {
            return input.clone();
        }
        self.reserve(samples);
        let (channels, total) = (self.channels, self.channels * samples);
        let first = !self.started;
        self.started = true;

        let mut current_in = input.clone();
        let mut use_ping = true;
        for planned in &self.plan {
            let out = if use_ping { self.buf_ping.clone() } else { self.buf_pong.clone() };
            let client = &self.client;
            match planned {
                Planned::Filter { filter, state } => match self.mode {
                    ChunkMode::Independent => {
                        filter.apply(client, &current_in, &out, &self.scratch, state, channels, samples)
                    }
                    ChunkMode::Stateful => filter
                        .apply_stateful(client, &current_in, &out, state, channels, samples, first)
                        .expect("stateful workspaces only hold forward filters"),
                },
                Planned::SpatialMatrix { weights } => {
                    execute_spatial_matrix_multiply::<R>(client, &current_in, weights, &out, channels, samples);
                }
                Planned::CenteredFir { taps, radius } => {
                    execute_fir_centered::<R>(client, &current_in, &out, taps, channels, samples, *radius);
                }
                Planned::Stage(stage) => match stage {
                    PipelineStage::Scale { alpha, beta } => {
                        execute_scaling::<R>(client, &current_in, &out, total, *alpha, *beta)
                    }
                    PipelineStage::SubtractBaseline { baseline_uv } => {
                        execute_scaling::<R>(client, &current_in, &out, total, 1.0, -*baseline_uv)
                    }
                    PipelineStage::Clamp { min, max } => {
                        execute_clamp::<R>(client, &current_in, &out, total, *min, *max)
                    }
                    PipelineStage::CommonAverageReference => {
                        execute_direct_car::<R>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::Median9p => {
                        execute_median_9p::<R>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::TeagerKaiser => {
                        execute_teager_kaiser::<R>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::Filter(_)
                    | PipelineStage::SpatialWhitening(_)
                    | PipelineStage::SurfaceLaplacian(_)
                    | PipelineStage::GaussianSmooth { .. } => {
                        unreachable!("pre-planned stage")
                    }
                },
            }
            current_in = out;
            use_ping = !use_ping;
        }

        let unused = ((self.capacity_samples - samples) * channels * 4) as u64;
        current_in.offset_end(unused)
    }

    /// Uploads a `[channels, samples]` host chunk, runs the pipeline and returns the device result
    /// **without** downloading it (see [`Self::process_handle`]).
    pub fn process_chunk_in_vram(&mut self, input: &[f32], samples: usize) -> Handle {
        assert_eq!(input.len(), self.channels * samples);
        let in_handle = self.client.create_from_slice(f32::as_bytes(input));
        self.process_handle(&in_handle, samples)
    }

    /// Sets the per-channel gain and offset (µV per stored unit) used by
    /// [`Self::process_stored_chunk_in_vram`].
    pub fn set_stored_scaling(&mut self, gains: &[f32], offsets: &[f32]) {
        assert_eq!(gains.len(), self.channels);
        assert_eq!(offsets.len(), self.channels);
        self.stored_scaling =
            Some((self.client.create_from_slice(f32::as_bytes(gains)), self.client.create_from_slice(f32::as_bytes(offsets))));
    }

    /// Uploads a `[channels, samples]` chunk of stored values (`format`, little-endian, as
    /// [`dsp_core::RecordingSource::read_stored`] returns it), scales it to µV on the device and
    /// runs the pipeline. Integer recordings move `format.bytes()` per sample instead of 4. Needs
    /// [`Self::set_stored_scaling`]; see [`Self::process_handle`] for the returned handle.
    pub fn process_stored_chunk_in_vram(&mut self, stored: &[u8], format: SampleFormat, samples: usize) -> DspResult<Handle> {
        if stored.len() != self.channels * samples * format.bytes() {
            return Err(DspError::ShapeMismatch { expected: vec![self.channels, samples, format.bytes()], actual: vec![stored.len()] });
        }
        let (gains, offsets) = self
            .stored_scaling
            .clone()
            .ok_or_else(|| DspError::InvalidConfig("set_stored_scaling before processing stored chunks".into()))?;
        self.reserve(samples);
        let unpacked = self
            .unpacked
            .get_or_insert_with(|| self.client.empty((self.channels * self.capacity_samples * 4).max(4)))
            .clone();
        let words = self.client.create_from_slice(u32::as_bytes(&stored_words(stored)));
        execute_unpack_stored::<R>(&self.client, &words, format, &gains, &offsets, &unpacked, self.channels, samples)?;
        let unused = ((self.capacity_samples - samples) * self.channels * 4) as u64;
        Ok(self.process_handle(&unpacked.offset_end(unused), samples))
    }

    /// Runs the pipeline on a `[channels, samples]` host chunk and writes the result to `output`.
    pub fn process_chunk(&mut self, input: &[f32], samples: usize, output: &mut [f32]) {
        assert_eq!(input.len(), self.channels * samples);
        assert_eq!(output.len(), self.channels * samples);

        if self.plan.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        let out_handle = self.process_chunk_in_vram(input, samples);
        let out_bytes = self.client.read_one_unchecked(out_handle);
        let out_slice = f32::from_bytes(&out_bytes);
        output.copy_from_slice(&out_slice[..output.len()]);
    }
}
