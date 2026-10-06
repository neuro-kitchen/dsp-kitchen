//! Persistent device workspace for streaming multi-chunk execution without repeated allocations.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::engine::Pipeline;
use super::stage::PipelineStage;
use crate::core::{buffer, cast, cast_f32, DspFloat, EdgeMode};
use crate::filter::design::FilterError;
use crate::filter::fir::gaussian::GAUSSIAN_TRUNCATE;
use crate::filter::iir::DeviceFilter;
use crate::filter::{execute_fir_centered, execute_median, execute_teager_kaiser, gaussian_kernel_1d, FilterMode};
use crate::math::{execute_clamp, execute_scaling, execute_unpack_stored, upload_stored};
use crate::spatial::{execute_direct_car, DeviceSpatialMatrix};
use dsp_core::{DspError, DspResult, SampleFormat};

/// How consecutive chunks relate to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    /// Every chunk is processed on its own (halo windows, random access). Forward filters start per
    /// their `FilterStart` (default at rest), forward-backward filters odd-pad both edges and start from the steady state of their
    /// first sample (scipy `sosfilt` / `sosfiltfilt`). The caller supplies halos of at least
    /// [`Pipeline::settling`].
    Independent,
    /// Chunks are consecutive pieces of one stream (live input, sequential reads). Filter section
    /// state is carried from chunk to chunk (the first chunk starts per `FilterStart`), so the output is exact
    /// with no halo. Only forward filters are allowed. Stencil stages (`Median`, `TeagerKaiser`,
    /// `GaussianSmooth`) still see each chunk's edges on their own.
    Stateful,
}

enum Planned<F: DspFloat> {
    Stage(PipelineStage),
    Filter { filter: DeviceFilter<F>, state: Handle },
    SpatialMatrix(DeviceSpatialMatrix),
    CenteredFir { taps: Handle, radius: usize, edge: EdgeMode },
}

/// Pre-allocated device workspace for running a [`Pipeline`] over many chunks of `F` values.
///
/// Filter designs are made and uploaded once. Buffers only grow: a shorter tail chunk reuses them.
pub struct PipelineWorkspace<R: Runtime, F: DspFloat = f32> {
    client: ComputeClient<R>,
    pipeline: Pipeline,
    plan: Vec<Planned<F>>,
    mode: ChunkMode,
    started: bool,
    channels: usize,
    capacity_samples: usize,
    buf_ping: Handle,
    buf_pong: Handle,
    scratch: Handle,
    scratch_len: usize,
    /// Per-channel gain and offset for stored chunks, and the buffer they are unpacked into.
    stored_scaling: Option<(Handle, Handle)>,
    unpacked: Option<Handle>,
}

impl<R: Runtime, F: DspFloat> PipelineWorkspace<R, F> {
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
                    let filter = DeviceFilter::<F>::new(&client, spec, sample_rate)?;
                    if mode == ChunkMode::Stateful && filter.mode() != FilterMode::Forward {
                        return Err(FilterError::ForwardBackwardOnLiveStream);
                    }
                    let state = buffer::empty::<R, F>(&client, channels * filter.state_len());
                    plan.push(Planned::Filter { filter, state });
                }
                PipelineStage::SpatialWhitening(w) => {
                    assert_eq!(w.num_channels, channels, "SpatialWhitening channel mismatch");
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<R, F>(&client, &w.matrix, channels)));
                }
                PipelineStage::SurfaceLaplacian(lap) => {
                    assert_eq!(lap.num_channels, channels, "SurfaceLaplacian channel mismatch");
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<R, F>(&client, &lap.matrix, channels)));
                }
                PipelineStage::GaussianSmooth { sigma_samples, edge } => {
                    let (taps, radius) = gaussian_kernel_1d(*sigma_samples, GAUSSIAN_TRUNCATE);
                    plan.push(Planned::CenteredFir { taps: buffer::upload(&client, &cast_f32::<F>(&taps)), radius, edge: *edge });
                }
                other => plan.push(Planned::Stage(other.clone())),
            }
        }
        let mut workspace = Self {
            buf_ping: buffer::empty::<R, F>(&client, channels * initial_samples),
            buf_pong: buffer::empty::<R, F>(&client, channels * initial_samples),
            scratch: buffer::empty::<R, F>(&client, 1),
            client,
            pipeline,
            plan,
            mode,
            started: false,
            channels,
            capacity_samples: initial_samples,
            scratch_len: 1,
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

    /// Forgets carried filter state; the next stateful chunk starts per its `FilterStart`.
    pub fn reset(&mut self) {
        self.started = false;
    }

    fn reserve(&mut self, samples: usize) {
        if samples > self.capacity_samples {
            self.buf_ping = buffer::empty::<R, F>(&self.client, self.channels * samples);
            self.buf_pong = buffer::empty::<R, F>(&self.client, self.channels * samples);
            self.unpacked = None;
            self.capacity_samples = samples;
        }
        let need = self
            .plan
            .iter()
            .map(|p| match p {
                Planned::Filter { filter, .. } => filter.scratch_len(self.channels, samples),
                Planned::Stage(_) | Planned::SpatialMatrix(_) | Planned::CenteredFir { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        if need > self.scratch_len {
            self.scratch = buffer::empty::<R, F>(&self.client, need);
            self.scratch_len = need;
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
                Planned::SpatialMatrix(matrix) => matrix.apply::<R, F>(client, &current_in, &out, channels, samples),
                Planned::CenteredFir { taps, radius, edge } => {
                    execute_fir_centered::<R, F>(client, &current_in, &out, taps, channels, samples, *radius, *edge);
                }
                Planned::Stage(stage) => match stage {
                    PipelineStage::Scale { alpha, beta } => {
                        execute_scaling::<R, F>(client, &current_in, &out, total, cast(*alpha as f64), cast(*beta as f64))
                    }
                    PipelineStage::SubtractBaseline { baseline } => {
                        execute_scaling::<R, F>(client, &current_in, &out, total, cast(1.0), cast(-*baseline as f64))
                    }
                    PipelineStage::Clamp { min, max } => {
                        execute_clamp::<R, F>(client, &current_in, &out, total, cast(*min as f64), cast(*max as f64))
                    }
                    PipelineStage::CommonAverageReference => {
                        execute_direct_car::<R, F>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::Median { width, edge } => {
                        execute_median::<R, F>(client, &current_in, &out, channels, samples, *width, *edge)
                    }
                    PipelineStage::TeagerKaiser { edge } => {
                        execute_teager_kaiser::<R, F>(client, &current_in, &out, channels, samples, *edge)
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

        buffer::truncate::<F>(current_in, (self.capacity_samples - samples) * channels)
    }

    /// Uploads a `[channels, samples]` host chunk, runs the pipeline and returns the device result
    /// **without** downloading it (see [`Self::process_handle`]).
    pub fn process_chunk_in_vram(&mut self, input: &[F], samples: usize) -> Handle {
        assert_eq!(input.len(), self.channels * samples);
        let in_handle = buffer::upload(&self.client, input);
        self.process_handle(&in_handle, samples)
    }

    /// Sets the per-channel gain and offset (scaled unit per stored unit) used by
    /// [`Self::process_stored_chunk_in_vram`].
    pub fn set_stored_scaling(&mut self, gains: &[f32], offsets: &[f32]) {
        assert_eq!(gains.len(), self.channels);
        assert_eq!(offsets.len(), self.channels);
        self.stored_scaling =
            Some((buffer::upload(&self.client, &cast_f32::<F>(gains)), buffer::upload(&self.client, &cast_f32::<F>(offsets))));
    }

    /// Uploads a `[channels, samples]` chunk of stored values (`format`, little-endian, as
    /// [`dsp_core::RecordingSource::read_stored`] returns it), scales it on the device and runs the
    /// pipeline. Integer recordings move `format.bytes()` per sample instead of `size_of::<F>()`.
    /// Needs [`Self::set_stored_scaling`]; see [`Self::process_handle`] for the returned handle.
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
            .get_or_insert_with(|| buffer::empty::<R, F>(&self.client, self.channels * self.capacity_samples))
            .clone();
        let words = upload_stored(&self.client, stored);
        execute_unpack_stored::<R, F>(&self.client, &words, format, &gains, &offsets, &unpacked, self.channels, samples)?;
        let unpacked = buffer::truncate::<F>(unpacked, (self.capacity_samples - samples) * self.channels);
        Ok(self.process_handle(&unpacked, samples))
    }

    /// Runs the pipeline on a `[channels, samples]` host chunk and writes the result to `output`.
    pub fn process_chunk(&mut self, input: &[F], samples: usize, output: &mut [F]) {
        assert_eq!(input.len(), self.channels * samples);
        assert_eq!(output.len(), self.channels * samples);

        if self.plan.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        let out_handle = self.process_chunk_in_vram(input, samples);
        output.copy_from_slice(&buffer::download_prefix::<R, F>(&self.client, out_handle, output.len()));
    }
}

