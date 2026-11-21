//! Persistent device workspace for streaming multi-chunk execution without repeated allocations.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::engine::Pipeline;
use super::stage::PipelineStage;
use crate::core::{buffer, cast, cast_f32, DspFloat};
use crate::filter::design::OpsError;
use crate::filter::fir::gaussian::GAUSSIAN_TRUNCATE;
use crate::filter::iir::DeviceFilter;
use crate::filter::{execute_median, execute_teager_kaiser, gaussian_kernel_1d, FilterMode, PreparedFir};
use crate::math::{execute_clamp, execute_scaling, execute_unpack_stored, stored_word_bytes, write_stored_owned};
use crate::spatial::{execute_common_median, execute_direct_car, DeviceSpatialMatrix};
use dspk_core::{DspError, SampleFormat};

use crate::error::OpsResult;

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
    Fir(PreparedFir),
    /// The channel order, uploaded once.
    Reorder { rows: Handle },
    /// Consecutive pointwise stages fused into one kernel (two or more; one runs as it is).
    Pointwise(crate::math::PointwiseRun),
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
    ///
    /// # Errors
    ///
    /// [`OpsError`] when a filter stage's design is invalid at `sample_rate`.
    pub fn new(
        client: Client,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, OpsError> {
        Self::with_mode(client, pipeline, channels, initial_samples, sample_rate, ChunkMode::Independent)
    }

    /// Workspace for a continuous stream (see [`ChunkMode::Stateful`]); rejects forward-backward
    /// filters.
    ///
    /// # Errors
    ///
    /// [`OpsError`] when a filter stage's design is invalid at `sample_rate`, and
    /// [`OpsError::ForwardBackwardOnLiveStream`] for a zero-phase filter (it needs future samples).
    pub fn new_stateful(
        client: Client,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, OpsError> {
        Self::with_mode(client, pipeline, channels, initial_samples, sample_rate, ChunkMode::Stateful)
    }

    fn with_mode(
        client: Client,
        pipeline: Pipeline,
        channels: usize,
        initial_samples: usize,
        sample_rate: f64,
        mode: ChunkMode,
    ) -> Result<Self, OpsError> {
        let mut plan = Vec::with_capacity(pipeline.len());
        for stage in pipeline.stages() {
            match stage {
                PipelineStage::Filter(spec) => {
                    let filter = DeviceFilter::<F>::new(&client, spec, sample_rate)?;
                    if mode == ChunkMode::Stateful && filter.mode() != FilterMode::Forward {
                        return Err(OpsError::ForwardBackwardOnLiveStream);
                    }
                    let state = buffer::empty::<F>(&client, channels * filter.state_len());
                    plan.push(Planned::Filter { filter, state });
                }
                PipelineStage::SpatialWhitening(w) => {
                    if w.num_channels != channels {
                        return Err(OpsError::SpatialChannelMismatch { stage: "spatial whitening", expected: w.num_channels, actual: channels });
                    }
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<F>(&client, &w.matrix, channels)));
                }
                PipelineStage::SurfaceLaplacian(lap) => {
                    if lap.num_channels != channels {
                        return Err(OpsError::SpatialChannelMismatch { stage: "the surface Laplacian", expected: lap.num_channels, actual: channels });
                    }
                    plan.push(Planned::SpatialMatrix(DeviceSpatialMatrix::upload::<F>(&client, &lap.matrix, channels)));
                }
                PipelineStage::Fir { taps, centered, edge } => {
                    assert!(!taps.is_empty(), "FIR stage without taps");
                    let handle = buffer::upload(&client, &cast_f32::<F>(taps));
                    if *centered {
                        assert!(taps.len() % 2 == 1, "centered FIR needs an odd number of taps, got {}", taps.len());
                        plan.push(Planned::Fir(PreparedFir::centered::<F>(&client, handle, taps.len() / 2, *edge)));
                    } else {
                        plan.push(Planned::Fir(PreparedFir::causal::<F>(&client, handle, taps.len(), *edge)));
                    }
                }
                PipelineStage::GaussianSmooth { sigma_samples, edge } => {
                    let (taps, radius) = gaussian_kernel_1d(*sigma_samples, GAUSSIAN_TRUNCATE);
                    let handle = buffer::upload(&client, &cast_f32::<F>(&taps));
                    plan.push(Planned::Fir(PreparedFir::centered::<F>(&client, handle, radius, *edge)));
                }
                PipelineStage::ReorderChannels { order } => {
                    let mut seen = vec![false; channels];
                    for &c in order {
                        if c >= channels || std::mem::replace(&mut seen[c], true) {
                            return Err(OpsError::InvalidChannelOrder(format!("{order:?} is not an order of {channels} channels (each once)")));
                        }
                    }
                    if order.len() != channels {
                        return Err(OpsError::InvalidChannelOrder(format!("{order:?} is not an order of {channels} channels (each once)")));
                    }
                    let rows: Vec<u32> = order.iter().map(|&c| c as u32).collect();
                    plan.push(Planned::Reorder { rows: buffer::upload(&client, &rows) });
                }
                other => plan.push(Planned::Stage(other.clone())),
            }
        }
        // Runs of two or more pointwise stages become one kernel (each sample read and written once
        // for the run instead of once per stage)
        let mut fused: Vec<Planned<F>> = Vec::with_capacity(plan.len());
        let mut run: Vec<(u32, f64, f64)> = Vec::new();
        let flush = |run: &mut Vec<(u32, f64, f64)>, stages: &mut Vec<PipelineStage>, fused: &mut Vec<Planned<F>>, client: &Client| {
            if run.len() >= 2 {
                fused.push(Planned::Pointwise(crate::math::PointwiseRun::upload::<F>(client, run)));
            } else {
                fused.extend(stages.drain(..).map(Planned::Stage));
            }
            run.clear();
            stages.clear();
        };
        let mut run_stages: Vec<PipelineStage> = Vec::new();
        for planned in plan {
            match &planned {
                Planned::Stage(stage) if crate::math::pointwise_op(stage).is_some() => {
                    run.push(crate::math::pointwise_op(stage).expect("a pointwise stage"));
                    run_stages.push(stage.clone());
                }
                _ => {
                    flush(&mut run, &mut run_stages, &mut fused, &client);
                    fused.push(planned);
                }
            }
        }
        flush(&mut run, &mut run_stages, &mut fused, &client);
        let plan = fused;
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

    /// Number of channels this workspace processes.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Bytes this workspace keeps on the device (its ping-pong and scratch buffers, the upload and
    /// unpack buffers once made, filter state): what a memory budget charges it.
    pub fn device_bytes(&self) -> u64 {
        let f = size_of::<F>() as u64;
        let rows = (self.channels * self.capacity_samples) as u64 * f;
        let state: u64 = self.plan.iter().map(|p| if let Planned::Filter { filter, .. } = p { (self.channels * filter.state_len()) as u64 * f } else { 0 }).sum();
        2 * rows
            + self.scratch_len as u64 * f
            + if self.unpacked.is_some() { rows } else { 0 }
            + if self.input.is_some() { rows } else { 0 }
            + self.stored_input.as_ref().map_or(0, |(_, bytes)| *bytes as u64)
            + state
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
                Planned::Stage(_) | Planned::SpatialMatrix(_) | Planned::Fir(_) | Planned::Reorder { .. } | Planned::Pointwise(_) => 0,
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
        self.process_handle_with(input, samples, |_| {})
    }

    /// Kernels the pipeline runs per chunk, in order (a fused run of pointwise stages counts once).
    pub fn launches(&self) -> usize {
        self.plan.len()
    }

    /// [`Self::process_handle`], calling `after(i)` once each planned step `i` (of
    /// [`Self::launches`]) is launched: a measuring caller syncs there to time each step.
    pub fn process_handle_with(&mut self, input: &Handle, samples: usize, mut after: impl FnMut(usize)) -> Handle {
        if self.plan.is_empty() {
            return input.clone();
        }
        self.reserve(samples);
        let (channels, total) = (self.channels, self.channels * samples);
        let first = !self.started;
        self.started = true;

        let mut current_in = input.clone();
        let mut use_ping = true;
        for (step, planned) in self.plan.iter().enumerate() {
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
                Planned::Fir(fir) => {
                    fir.execute::<F>(client, &current_in, &out, channels, samples);
                }
                Planned::Reorder { rows } => {
                    crate::core::layout::execute_gather_rows::<F>(client, &current_in, rows, &out, channels, channels, samples);
                }
                Planned::Pointwise(run) => run.apply::<F>(client, &current_in, &out, total),
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
                    PipelineStage::Elementwise(f) => crate::math::execute_elementwise::<F>(client, &current_in, &out, total, *f),
                    PipelineStage::CommonAverageReference => {
                        execute_direct_car::<F>(client, &current_in, &out, channels, samples)
                    }
                    PipelineStage::CommonMedianReference => {
                        execute_common_median::<F>(client, &current_in, &out, channels, samples)
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
                    | PipelineStage::GaussianSmooth { .. }
                    | PipelineStage::ReorderChannels { .. }
                    | PipelineStage::Fir { .. } => {
                        unreachable!("pre-planned stage")
                    }
                },
            }
            current_in = out;
            use_ping = !use_ping;
            after(step);
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
    /// [`dspk_core::RecordingSource::read_stored`] returns it), scales it on the device and runs the
    /// pipeline. Integer recordings move `format.bytes()` per sample instead of `size_of::<F>()`.
    /// Needs [`Self::set_stored_scaling`]; see [`Self::process_handle`] for the returned handle.
    ///
    /// # Errors
    ///
    /// [`DspError::ShapeMismatch`] when `stored` is not `channels × samples × format.bytes()` long;
    /// [`DspError::InvalidConfig`] before [`Self::set_stored_scaling`]; [`DspError::UnsupportedFormat`]
    /// for `float64` storage.
    pub fn process_stored_chunk_in_vram(&mut self, stored: &[u8], format: SampleFormat, samples: usize) -> OpsResult<Handle> {
        self.process_owned_stored_chunk_in_vram(stored.to_vec(), format, samples)
    }

    /// [`Self::process_stored_chunk_in_vram`] taking ownership of the stored bytes: they move into
    /// the upload without a copy on this thread (padded in place when they do not fill whole
    /// 32-bit words).
    ///
    /// # Errors
    ///
    /// As [`Self::process_stored_chunk_in_vram`].
    pub fn process_owned_stored_chunk_in_vram(&mut self, stored: Vec<u8>, format: SampleFormat, samples: usize) -> OpsResult<Handle> {
        if stored.len() != self.channels * samples * format.bytes() {
            return Err(DspError::ShapeMismatch { expected: vec![self.channels, samples, format.bytes()], actual: vec![stored.len()] }.into());
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

    /// Uploads a `[channels, samples]` chunk of stored values and scales it on the device, into a
    /// buffer of its own (not one of the workspace's): an input kept for later chains of the same
    /// window (a parameter changed, an op turned on or off) run with [`Self::process_handle`]
    /// without reading and uploading the window again.
    ///
    /// # Errors
    ///
    /// As [`Self::process_stored_chunk_in_vram`].
    pub fn upload_stored_owned(&mut self, stored: Vec<u8>, format: SampleFormat, samples: usize) -> OpsResult<Handle> {
        if stored.len() != self.channels * samples * format.bytes() {
            return Err(DspError::ShapeMismatch { expected: vec![self.channels, samples, format.bytes()], actual: vec![stored.len()] }.into());
        }
        let (gains, offsets) = self
            .stored_scaling
            .clone()
            .ok_or_else(|| DspError::InvalidConfig("set_stored_scaling before processing stored chunks".into()))?;
        let words = self.client.empty(stored_word_bytes(stored.len()));
        write_stored_owned(&self.client, &words, stored);
        // Sized to the workspace's capacity (a caller that sizes workspaces in classes gets inputs
        // the device's pool reuses across window sizes); the rows are `samples` long
        let unpacked = buffer::empty::<F>(&self.client, self.channels * self.capacity_samples.max(samples));
        execute_unpack_stored::<F>(&self.client, &words, format, &gains, &offsets, &unpacked, self.channels, samples)?;
        Ok(unpacked)
    }

    /// Records the pipeline over `input` (`[channels, samples]` on the device) as a CubeCL graph,
    /// replayed as one submission (CUDA, HIP: a hardware graph; wgpu: a software graph). For work
    /// of one shape repeated (same-size windows, live blocks): write the next input into `input` in
    /// place between replays. Runs the pipeline once to warm it (kernels compiled, buffers sized),
    /// then once recorded. Independent chunks only (a stateful workspace's carried state changes
    /// what a recording would replay). Nothing else may use the device while it records: another
    /// thread's launches or reads on the same client land in the recording or fail.
    ///
    /// # Errors
    ///
    /// When the runtime has no graph support (the CPU runtime) or the recording is refused (an
    /// allocation or a host read inside it): run [`Self::process_handle`] instead.
    pub fn capture(&mut self, input: &Handle, samples: usize) -> Result<CapturedPipeline, String> {
        if self.mode != ChunkMode::Independent {
            return Err("only independent chunks are captured".into());
        }
        self.client.graph_prepare().map_err(|e| format!("graph capture: {e:?}"))?;
        // Warm: compiles the kernels and sizes every buffer, so the recording allocates nothing
        let _ = self.process_handle(input, samples);
        self.client.start_capture().map_err(|e| format!("graph capture: {e:?}"))?;
        let output = self.process_handle(input, samples);
        let graph = self.client.stop_capture().map_err(|e| format!("graph capture: {e:?}"))?;
        Ok(CapturedPipeline { graph, output })
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

/// A pipeline recorded by [`PipelineWorkspace::capture`]: replay it after writing the next input
/// into the buffer it was recorded over; the result lands in [`Self::output`].
pub struct CapturedPipeline {
    graph: cubecl::client::Graph,
    output: Handle,
}

impl CapturedPipeline {
    /// Where every replay writes its result (a view of the workspace's buffers: read it before the
    /// next replay).
    pub fn output(&self) -> &Handle {
        &self.output
    }

    /// Runs the recorded launches again.
    ///
    /// # Safety
    ///
    /// The input buffer it was recorded over and the workspace that recorded it must still exist,
    /// unshared, until the replay's work completes (CubeCL replays against the recorded buffers
    /// without checking them).
    pub unsafe fn replay(&self) -> Result<(), String> {
        // SAFETY: the caller keeps the recorded buffers alive (see above)
        unsafe { self.graph.replay() }.map_err(|e| format!("graph replay: {e:?}"))
    }
}
