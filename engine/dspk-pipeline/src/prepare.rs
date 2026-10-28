//! Everything decided before the device is touched: the stages, the opened input, the channels and
//! range, the mode and halos, the window schedule. [`plan`] reports it ([`Plan`]); the executor
//! (`run.rs`) runs it.

use std::path::{Path, PathBuf};

use dspk_core::compute::device_elements;
use dspk_core::{ChunkSchedule, DspResult, RecordingSource, SampleFormat, SignalUnit};
use dspk_io::MergedRecording;
use dspk_ops::core::EdgeMode;
use dspk_ops::filter::FilterMode;
use dspk_ops::resampler::poly::RESAMPLE_POLY_HALF_LEN_PER_RATE;
use dspk_ops::resampler::{resample_poly_len, FirWindow, ResampleFilter};
use dspk_ops::{Pipeline, PipelineStage};

use crate::spec::{ExecMode, Input, Node, OutputFormat, PipelineSpec, Upload};
use crate::PipelineError;

/// How the chain runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutedMode {
    /// Windows in order with filter state carried: equal to processing the whole range at once.
    Stream,
    /// Independent windows with `left` / `right` halos of context.
    Windows { left: usize, right: usize },
}

/// What running one input would do, without touching the device.
#[derive(Debug, Clone)]
pub struct Plan {
    pub input: String,
    pub output: PathBuf,
    pub mode: ExecutedMode,
    /// `true` when stored integers would be uploaded and scaled on the device.
    pub stored_upload: bool,
    pub channels: usize,
    /// Input samples per channel processed (`stop - start`).
    pub input_samples: u64,
    pub input_rate_hz: f64,
    /// Output samples per channel and sample rate (differ from the input's after `resample`).
    pub output_samples: u64,
    pub output_rate_hz: f64,
    pub windows: usize,
    /// Samples per channel of the largest window, halos included.
    pub max_window_samples: usize,
    /// Bytes read from the input and uploaded, over all windows (halos included).
    pub bytes_read: u64,
    pub bytes_uploaded: u64,
}

/// A trailing `resample` node, reduced to coprime factors.
pub(crate) struct Resample {
    pub(crate) up: usize,
    pub(crate) down: usize,
    pub(crate) filter: ResampleFilter,
    pub(crate) edge: EdgeMode,
    /// Input samples the resampling filter reaches on each side of an output sample.
    pub(crate) reach: usize,
}

impl Resample {
    fn new(up: usize, down: usize, beta: f64, edge: EdgeMode) -> Self {
        let g = gcd(up, down);
        let (up, down) = (up / g, down / g);
        let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
        Self { up, down, filter: ResampleFilter::Window(FirWindow::Kaiser { beta }), edge, reach: half_len.div_ceil(up) + 1 }
    }
}

/// One input, ready to run.
pub(crate) struct Prepared {
    pub(crate) source: Box<dyn RecordingSource>,
    pub(crate) pipeline: Pipeline,
    pub(crate) resample: Option<Resample>,
    pub(crate) channels: Vec<usize>,
    pub(crate) range: std::ops::Range<u64>,
    pub(crate) unit: SignalUnit,
    pub(crate) sample_rate: f64,
    pub(crate) stored_format: SampleFormat,
    pub(crate) stored_upload: bool,
    pub(crate) mode: ExecutedMode,
    pub(crate) chunk: usize,
    pub(crate) schedule: ChunkSchedule,
    pub(crate) out_rate: f64,
    pub(crate) out_samples: u64,
    pub(crate) output: PathBuf,
    pub(crate) format: OutputFormat,
    pub(crate) input_label: String,
}

impl Prepared {
    pub(crate) fn plan(&self) -> Plan {
        let nch = self.channels.len() as u64;
        let read: u64 = self.schedule.windows().iter().map(|w| w.read_len() as u64).sum::<u64>() * nch;
        let per_sample_up = if self.stored_upload { self.stored_format.bytes() } else { 4 } as u64;
        Plan {
            input: self.input_label.clone(),
            output: self.output.clone(),
            mode: self.mode,
            stored_upload: self.stored_upload,
            channels: self.channels.len(),
            input_samples: self.range.end - self.range.start,
            input_rate_hz: self.sample_rate,
            output_samples: self.out_samples,
            output_rate_hz: self.out_rate,
            windows: self.schedule.len(),
            max_window_samples: self.schedule.max_read_samples(),
            bytes_read: read * self.stored_format.bytes() as u64,
            bytes_uploaded: read * per_sample_up,
        }
    }
}

/// What [`crate::run`] would do for each input of `spec`, opening the inputs (metadata reads only)
/// with `dspk_io::open` and never touching the device.
pub fn plan(spec: &PipelineSpec, base_dir: &Path) -> Result<Vec<Plan>, PipelineError> {
    plan_with(spec, base_dir, default_open)
}

/// [`plan`] with recordings opened by `open`.
pub fn plan_with(
    spec: &PipelineSpec,
    base_dir: &Path,
    open: impl Fn(&Path, Option<&str>) -> DspResult<Box<dyn RecordingSource>>,
) -> Result<Vec<Plan>, PipelineError> {
    spec.expand_inputs(base_dir)?.iter().map(|one| prepare(one, base_dir, &open).map(|p| p.plan())).collect()
}

/// Opens a recording with `dspk_io` (generic formats): `source` picks a source within the file.
pub fn default_open(path: &Path, source: Option<&str>) -> DspResult<Box<dyn RecordingSource>> {
    match source {
        Some(id) => dspk_io::open_source(path, id),
        None => dspk_io::open(path),
    }
}

/// One input of an expanded spec (a `path`, or a single merge), checked and scheduled.
pub(crate) fn prepare(
    spec: &PipelineSpec,
    base_dir: &Path,
    open: &impl Fn(&Path, Option<&str>) -> DspResult<Box<dyn RecordingSource>>,
) -> Result<Prepared, PipelineError> {
    let chain = spec.chain()?;
    let Node::Read { path: input, inputs, source, channels: picked, start, stop } = chain.read.1 else {
        unreachable!("chain starts with read")
    };
    let Node::Write { path: output, format } = chain.write.1 else { unreachable!("chain ends with write") };

    let mut stages = Vec::with_capacity(chain.stages.len());
    let mut resample = None;
    for (i, (id, node)) in chain.stages.iter().enumerate() {
        let stage = node.stage().map_err(|e| PipelineError::Invalid(format!("node {id:?}: {e}")))?;
        match (node, stage) {
            (Node::Resample { up, down, beta, edge }, _) => {
                if i + 1 != chain.stages.len() {
                    return Err(PipelineError::Invalid(format!("node {id:?}: resample must be the last node before write")));
                }
                resample = Some(Resample::new(*up, *down, *beta, (*edge).into()));
            }
            (_, Some(stage)) => stages.push(stage),
            (_, None) => unreachable!("every other processing node has a stage"),
        }
    }
    let pipeline = Pipeline::with_stages(stages);

    let (source, input_label): (Box<dyn RecordingSource>, String) = match (input, inputs.as_deref()) {
        (Some(path), None) => (open(&base_dir.join(path), source.as_deref())?, path.display().to_string()),
        (None, Some([Input::Merge { merge }])) => {
            let parts = merge
                .iter()
                .map(|p| Ok((open(&base_dir.join(&p.path), p.source.as_deref())?, p.channels.clone())))
                .collect::<DspResult<Vec<_>>>()?;
            let label = merge.iter().map(|p| p.path.display().to_string()).collect::<Vec<_>>().join(" + ");
            (Box::new(MergedRecording::new(parts)?), label)
        }
        _ => unreachable!("expand_inputs leaves one path or one merge"),
    };
    let info = source.info();
    let (total, sample_rate, stored_format) = (info.samples, info.sample_rate_hz(), info.format);
    let channels = select_channels(picked.as_deref(), info.channel_count())?;
    let range = select_range(*start, *stop, total)?;
    let unit = info.channels[channels[0]].unit.clone();
    pipeline.validate(sample_rate).map_err(|e| PipelineError::Invalid(e.to_string()))?;

    let mut mode = resolve_mode(spec.execution.mode, &pipeline, sample_rate)?;
    let stored_upload = resolve_upload(spec.execution.upload, stored_format)?;
    let mut chunk = spec.execution.chunk_samples.max(1);
    let (mut out_rate, mut out_samples) = (sample_rate, range.end - range.start);
    if let Some(r) = &resample {
        // Independent windows aligned to `down`, with the stages' context plus the resampling
        // filter's reach: each output sample sees the input a whole-recording run would
        if spec.execution.mode == ExecMode::Stream {
            return Err(PipelineError::Invalid("execution.mode = \"stream\" is not possible with resample: use \"windows\" or \"auto\"".into()));
        }
        if range.start % r.down as u64 != 0 {
            return Err(PipelineError::Invalid(format!("read: start {} must be a multiple of the resample's down ({})", range.start, r.down)));
        }
        let (left, right) = pipeline.settling(sample_rate).map_err(|e| PipelineError::Invalid(e.to_string()))?;
        let align = |x: usize| x.div_ceil(r.down) * r.down;
        mode = ExecutedMode::Windows { left: align(left + r.reach), right: align(right + r.reach) };
        chunk = align(chunk);
        out_rate = sample_rate * r.up as f64 / r.down as f64;
        out_samples = resample_poly_len((range.end - range.start) as usize, r.up, r.down) as u64;
    }
    let (left, right) = match mode {
        ExecutedMode::Stream => (0, 0),
        ExecutedMode::Windows { left, right } => (left, right),
    };
    let schedule = ChunkSchedule::new(range.clone(), chunk as u64, left as u64, right as u64, total);
    device_elements("pipeline window [channels, samples]", &[channels.len(), schedule.max_read_samples()])?;

    Ok(Prepared {
        source,
        pipeline,
        resample,
        channels,
        range,
        unit,
        sample_rate,
        stored_format,
        stored_upload,
        mode,
        chunk,
        schedule,
        out_rate,
        out_samples,
        output: base_dir.join(output),
        format: *format,
        input_label,
    })
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `picked` (in its order, no repeats), or every channel.
fn select_channels(picked: Option<&[usize]>, count: usize) -> Result<Vec<usize>, PipelineError> {
    let Some(picked) = picked else { return Ok((0..count).collect()) };
    if picked.is_empty() {
        return Err(PipelineError::Invalid("read: channels is empty".into()));
    }
    let mut seen = vec![false; count];
    for &c in picked {
        if c >= count {
            return Err(PipelineError::Invalid(format!("read: channel {c} does not exist (the recording has {count})")));
        }
        if std::mem::replace(&mut seen[c], true) {
            return Err(PipelineError::Invalid(format!("read: channel {c} is listed twice")));
        }
    }
    Ok(picked.to_vec())
}

fn select_range(start: Option<u64>, stop: Option<u64>, total: u64) -> Result<std::ops::Range<u64>, PipelineError> {
    let (start, stop) = (start.unwrap_or(0), stop.unwrap_or(total));
    if start >= stop || stop > total {
        return Err(PipelineError::Invalid(format!("read: samples {start}..{stop} are not a non-empty range within 0..{total}")));
    }
    Ok(start..stop)
}

/// Stored upload works for integer storage; float32 gains nothing; float64 has no device unpacking.
fn resolve_upload(requested: Upload, format: SampleFormat) -> Result<bool, PipelineError> {
    let integer = !matches!(format, SampleFormat::F32 | SampleFormat::F64);
    match requested {
        Upload::Auto => Ok(integer),
        Upload::Scaled => Ok(false),
        Upload::Stored if format == SampleFormat::F64 => {
            Err(PipelineError::Invalid("execution.upload = \"stored\": float64 recordings cannot be scaled on the device".into()))
        }
        Upload::Stored => Ok(true),
    }
}

/// `stream` needs every filter forward (zero-phase needs future samples) and no stencil stage
/// (median, Gaussian, Teager–Kaiser, FIR read neighbours across window edges and carry no state;
/// for FIR, window mode with its halo is exact).
fn resolve_mode(requested: ExecMode, pipeline: &Pipeline, sample_rate: f64) -> Result<ExecutedMode, PipelineError> {
    let blocker = pipeline.stages().iter().find_map(|stage| match stage {
        PipelineStage::Filter(spec) if spec.mode == FilterMode::ForwardBackward => Some("a zero-phase (forward_backward) filter"),
        PipelineStage::Median { .. } => Some("a median stage"),
        PipelineStage::Fir { .. } => Some("a fir stage"),
        PipelineStage::GaussianSmooth { .. } => Some("a gaussian_smooth stage"),
        PipelineStage::TeagerKaiser { .. } => Some("a teager_kaiser stage"),
        _ => None,
    });
    let windows = || {
        let (left, right) = pipeline.settling(sample_rate).map_err(|e| PipelineError::Invalid(e.to_string()))?;
        Ok(ExecutedMode::Windows { left, right })
    };
    match (requested, blocker) {
        (ExecMode::Stream, Some(what)) => Err(PipelineError::Invalid(format!(
            "execution.mode = \"stream\" is not possible with {what}: use \"windows\" or \"auto\""
        ))),
        (ExecMode::Stream, None) | (ExecMode::Auto, None) => Ok(ExecutedMode::Stream),
        (ExecMode::Windows, _) | (ExecMode::Auto, Some(_)) => windows(),
    }
}
