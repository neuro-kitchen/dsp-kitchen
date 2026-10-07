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
use crate::math::{execute_clamp, execute_scaling, execute_unpack_stored, stored_word_bytes, write_stored_owned};
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
pub struct PipelineWorkspace<F: DspFloat = f32> {
    client: Client,
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
    /// Persistent upload buffers (`F` chunks; stored words with their byte capacity), written in
    /// place every chunk instead of allocating a new device buffer per chunk.
    input: Option<Handle>,
    stored_input: Option<(Handle, usize)>,
}

impl<F: DspFloat> PipelineWorkspace<F> {
    /// Workspace for independent chunks (see [`ChunkMode::Independent`]).
    pub fn new(
        client: Client,
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
        client: Client,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, FilterError> {
        Self::with_mode(client, pipeline, channels, initial_samples, sample_rate, ChunkMode::Stateful)
    }

    fn with_mode(
        client: Client,
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
                    let state = buffer::empty::<F>(&client, channels * filter.state_len());
                    plan.push(Planned::Filter { filter, state });
                }
                PipelineStage::SpatialWhitening(w) => {
                    assert_eq!(w.num_channels, channels, "SpatialWhitening channel mismatch");
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<F>(&client, &w.matrix, channels)));
                }
                PipelineStage::SurfaceLaplacian(lap) => {
                    assert_eq!(lap.num_channels, channels, "SurfaceLaplacian channel mismatch");
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<F>(&client, &lap.matrix, channels)));
                }
                PipelineStage::GaussianSmooth { sigma_samples, edge } => {
                    let (taps, radius) = gaussian_kernel_1d(*sigma_samples, GAUSSIAN_TRUNCATE);
                    plan.push(Planned::CenteredFir { taps: buffer::upload(&client, &cast_f32::<F>(&taps)), radius, edge: *edge });
                }
                other => plan.push(Planned::Stage(other.clone())),
            }
        }
        let mut workspace = Self {
            buf_ping: buffer::empty::<F>(&client, channels * initial_samples),
            buf_pong: buffer::empty::<F>(&client, channels * initial_samples),
            scratch: buffer::empty::<F>(&client, 1),
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
            input: None,
            stored_input: None,
        };
        workspace.reserve(initial_samples);
        Ok(workspace)
    }

    /// Reference to the underlying [`Client`].
    #[inline]
    pub fn client(&self) -> &Client {
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
            self.buf_ping = buffer::empty::<F>(&self.client, self.channels * samples);
            self.buf_pong = buffer::empty::<F>(&self.client, self.channels * samples);
            self.unpacked = None;
            self.input = None;
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
            self.scratch = buffer::empty::<F>(&self.client, need);
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
                Planned::SpatialMatrix(matrix) => matrix.apply::<F>(client, &current_in, &out, channels, samples),
                Planned::CenteredFir { taps, radius, edge } => {
                    execute_fir_centered::<F>(client, &current_in, &out, taps, channels, samples, *radius, *edge);
                }
                Planned::Stage(stage) => match stage {
                    PipelineStage::Scale { alpha, beta } => {
                        execute_scaling::<F>(client, &current_in, &out, total, cast(*alpha as f64), cast(*beta as f64))
                    }
                    PipelineStage::SubtractBaseline { baseline } => {
                        execute_scaling::<F>(client, &current_in, &out, total, cast(1.0), cast(-*baseline as f64))
                    }
                    PipelineStage::Clamp { min, max } => {
                        execute_clamp::<F>(client, &current_in, &out, total, cast(*min as f64), cast(*max as f64))
                    }
                    PipelineStage::CommonAverageReference => {
                        execute_direct_car::<F>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::Median { width, edge } => {
                        execute_median::<F>(client, &current_in, &out, channels, samples, *width, *edge)
                    }
                    PipelineStage::TeagerKaiser { edge } => {
                        execute_teager_kaiser::<F>(client, &current_in, &out, channels, samples, *edge)
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
        self.process_owned_chunk_in_vram(input.to_vec(), samples)
    }

    /// [`Self::process_chunk_in_vram`] taking ownership of the chunk: it moves into the upload
    /// without a copy on this thread. The upload writes a persistent device buffer in place.
    pub fn process_owned_chunk_in_vram(&mut self, input: Vec<F>, samples: usize) -> Handle {
        assert_eq!(input.len(), self.channels * samples);
        self.reserve(samples);
        let capacity = self.channels * self.capacity_samples;
        let len = input.len();
        let in_handle = self.input.get_or_insert_with(|| buffer::empty::<F>(&self.client, capacity)).clone();
        buffer::write_owned(&self.client, &in_handle, input);
        let in_handle = buffer::truncate::<F>(in_handle, capacity - len);
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
        self.process_owned_stored_chunk_in_vram(stored.to_vec(), format, samples)
    }

    /// [`Self::process_stored_chunk_in_vram`] taking ownership of the stored bytes: they move into
    /// the upload without a copy on this thread (padded in place when they do not fill whole
    /// 32-bit words).
    pub fn process_owned_stored_chunk_in_vram(&mut self, stored: Vec<u8>, format: SampleFormat, samples: usize) -> DspResult<Handle> {
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
            .get_or_insert_with(|| buffer::empty::<F>(&self.client, self.channels * self.capacity_samples))
            .clone();
        let need = stored_word_bytes(stored.len());
        let words = match &self.stored_input {
            Some((handle, bytes)) if *bytes >= need => handle.clone(),
            _ => {
                let handle = self.client.empty(need);
                self.stored_input = Some((handle.clone(), need));
                handle
            }
        };
        write_stored_owned(&self.client, &words, stored);
        execute_unpack_stored::<F>(&self.client, &words, format, &gains, &offsets, &unpacked, self.channels, samples)?;
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
        output.copy_from_slice(&buffer::download_prefix::<F>(&self.client, out_handle, output.len()));
    }
}

