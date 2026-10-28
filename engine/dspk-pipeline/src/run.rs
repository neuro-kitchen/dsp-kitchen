//! The executor: a recording through the chain's stages on the device, window by window, into the
//! output file.
//!
//! Every window crosses to the device once and back once ([`PipelineWorkspace`]: filters designed
//! and uploaded once, device buffers reused). Integer recordings cross as their stored integers and
//! are scaled on the device. The next window is read while the current one runs ([`WindowLoader`]).
//! Results are appended to a raw float32 time-major file; Zarr output is converted from it at the
//! end. [`DataMovement`] counts the bytes of each step.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dspk_core::compute::device_elements;
use dspk_core::{ChunkSchedule, ComputeTarget, DspResult, HaloWindow, MemoryOrder, RecordingSource, SampleFormat, WindowLoader};
use dspk_io::{RawParams, RawRecording};
use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_ops::core::{buffer, EdgeMode};
use dspk_ops::resampler::poly::RESAMPLE_POLY_HALF_LEN_PER_RATE;
use dspk_ops::resampler::{resample_poly, resample_poly_len, FirWindow, ResampleFilter};
use dspk_ops::filter::FilterMode;
use dspk_ops::math::stored_word_bytes;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};

use crate::spec::{ExecMode, Node, OutputFormat, PipelineSpec, Upload};
use crate::PipelineError;

/// Appended to the output path for the canonical spec written next to it (provenance).
pub const PROVENANCE_SUFFIX: &str = ".pipeline.json";

/// How the chain was run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutedMode {
    /// Windows in order with filter state carried: equal to processing the whole range at once.
    Stream,
    /// Independent windows with `left` / `right` halos of context.
    Windows { left: usize, right: usize },
}

/// Bytes moved by a run, step by step (window halos included: they are read, uploaded and
/// downloaded with their window).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DataMovement {
    /// Stored bytes read from the input recording.
    pub read: u64,
    /// Bytes uploaded to the device.
    pub uploaded: u64,
    /// Bytes downloaded from the device.
    pub downloaded: u64,
    /// Bytes written to the output (float32).
    pub written: u64,
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub mode: ExecutedMode,
    pub runtime: &'static str,
    /// `true` when stored integers were uploaded and scaled on the device.
    pub stored_upload: bool,
    pub channels: usize,
    /// Samples per channel written (`stop - start`, resampled when the chain ends with `resample`).
    pub samples: u64,
    pub sample_rate_hz: f64,
    pub windows: usize,
    pub bytes: DataMovement,
    pub output: PathBuf,
    /// The canonical spec written next to the output.
    pub provenance: PathBuf,
    pub seconds: f64,
}

/// Runs `spec`, resolving relative paths against `base_dir`, opening the input with
/// `dspk_io::open` (generic formats).
pub fn run(spec: &PipelineSpec, base_dir: &Path) -> Result<RunReport, PipelineError> {
    run_with(spec, base_dir, |path, source| match source {
        Some(id) => dspk_io::open_source(path, id),
        None => dspk_io::open(path),
    })
}

/// [`run`] with the input opened by `open` (e.g. a crate that adds formats, such as
/// `dspk-synapse-io`).
pub fn run_with(
    spec: &PipelineSpec,
    base_dir: &Path,
    open: impl Fn(&Path, Option<&str>) -> DspResult<Box<dyn RecordingSource>>,
) -> Result<RunReport, PipelineError> {
    let started = Instant::now();
    let chain = spec.chain()?;
    let Node::Read { path: input, source, channels: picked, start, stop } = chain.read.1 else { unreachable!("chain starts with read") };
    let Node::Write { path: output, format } = chain.write.1 else { unreachable!("chain ends with write") };
    let (input, output) = (base_dir.join(input), base_dir.join(output));

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

    let source = open(&input, source.as_deref())?;
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
    let nch = channels.len();
    device_elements("pipeline window [channels, samples]", &[nch, schedule.max_read_samples()])?;

    let target = match &spec.execution.runtime {
        Some(name) => ComputeTarget::parse(name).and_then(ComputeTarget::checked),
        None => ComputeTarget::from_env(),
    }
    .map_err(|e| PipelineError::Compute(e.to_string()))?;
    let client = target.client().map_err(|e| PipelineError::Compute(e.to_string()))?;
    let capacity = schedule.max_read_samples().max(1);
    let mut workspace = match mode {
        ExecutedMode::Stream => PipelineWorkspace::<f32>::new_stateful(client, pipeline, nch, capacity, sample_rate),
        ExecutedMode::Windows { .. } => PipelineWorkspace::<f32>::new(client, pipeline, nch, capacity, sample_rate),
    }
    .map_err(|e| PipelineError::Invalid(e.to_string()))?;
    if stored_upload {
        let gains: Vec<f32> = channels.iter().map(|&c| info.channels[c].gain).collect();
        let offsets: Vec<f32> = channels.iter().map(|&c| info.channels[c].offset).collect();
        workspace.set_stored_scaling(&gains, &offsets);
    }

    // Raw float32, time-major; Zarr is converted from it at the end
    let raw_path = match format {
        OutputFormat::Raw => output.clone(),
        OutputFormat::Zarr => output.with_extension("dspk-tmp.bin"),
    };
    if let Some(dir) = raw_path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| PipelineError::Io(format!("{}: {e}", dir.display())))?;
    }
    let file = File::create(&raw_path).map_err(|e| PipelineError::Io(format!("{}: {e}", raw_path.display())))?;
    let mut writer = BufWriter::new(file);

    let mut bytes = DataMovement::default();
    let mut frames = Vec::new();
    // Columns `cols` of a processed `[channels, row_len]` block, channel-major → time-major,
    // appended to the output
    let mut emit = |rows: &[f32], row_len: usize, cols: std::ops::Range<usize>, bytes: &mut DataMovement| -> DspResult<()> {
        frames.clear();
        frames.reserve(cols.len() * nch * 4);
        for t in cols {
            for c in 0..nch {
                frames.extend_from_slice(&rows[c * row_len + t].to_le_bytes());
            }
        }
        writer.write_all(&frames)?;
        bytes.written += frames.len() as u64;
        Ok(())
    };
    // Downloads a processed window (resampled on the device first, when the chain ends with one)
    // and appends the samples it owns
    let origin = range.start;
    let mut finish = |win: &HaloWindow, handle: Handle, client: &Client, bytes: &mut DataMovement| -> DspResult<()> {
        let n = win.read_len();
        let (rows, row_len, cols) = match &resample {
            None => (buffer::download_prefix::<f32>(client, handle, nch * n), n, win.valid_local.clone()),
            Some(r) => {
                let out_len = resample_poly_len(n, r.up, r.down);
                let out = buffer::empty::<f32>(client, nch * out_len);
                resample_poly::<f32>(client, &handle, &out, nch, n, r.up, r.down, &r.filter, r.edge);
                // Output j of the window is input `read_start + j · down / up`
                let first = ((win.valid_global.start - win.read_global.start) as usize) * r.up / r.down;
                let to_end = resample_poly_len((win.valid_global.end - origin) as usize, r.up, r.down);
                let start = ((win.valid_global.start - origin) as usize) * r.up / r.down;
                let count = to_end - start;
                (buffer::download_prefix::<f32>(client, out, nch * out_len), out_len, first..(first + count).min(out_len))
            }
        };
        bytes.downloaded += (rows.len() * 4) as u64;
        emit(&rows, row_len, cols, bytes)
    };
    let loader = WindowLoader::with_channels(source.as_ref(), channels.clone());
    let read_bytes = |win: &HaloWindow| (nch * win.read_len() * stored_format.bytes()) as u64;
    if stored_upload {
        loader.stream_stored_owned_while(schedule.windows(), |win, stored| {
            bytes.read += read_bytes(win);
            bytes.uploaded += stored_word_bytes(stored.len()) as u64;
            let handle = workspace.process_owned_stored_chunk_in_vram(stored, stored_format, win.read_len())?;
            finish(win, handle, workspace.client(), &mut bytes)?;
            Ok(true)
        })?;
    } else {
        loader.stream(schedule.windows(), |win, samples| {
            bytes.read += read_bytes(win);
            bytes.uploaded += (samples.len() * 4) as u64;
            let handle = workspace.process_chunk_in_vram(samples, win.read_len());
            finish(win, handle, workspace.client(), &mut bytes)
        })?;
    }
    drop(finish);
    drop(emit);
    writer.flush().map_err(|e| PipelineError::Io(format!("{}: {e}", raw_path.display())))?;
    drop(writer);

    let samples = out_samples;
    let mut params = RawParams::new(nch, out_rate, SampleFormat::F32, MemoryOrder::TimeMajor);
    params.unit = unit;
    params.samples = Some(samples);
    params.write_sidecar(&raw_path)?;

    if *format == OutputFormat::Zarr {
        let raw = RawRecording::open(&raw_path)?;
        dspk_io::write_zarr(&raw, &output, chunk, |_, _| {})?;
        drop(raw);
        for p in [raw_path.clone(), RawParams::sidecar_path(&raw_path)] {
            std::fs::remove_file(&p).map_err(|e| PipelineError::Io(format!("{}: {e}", p.display())))?;
        }
    }

    let provenance = provenance_path(&output);
    std::fs::write(&provenance, spec.to_json()).map_err(|e| PipelineError::Io(format!("{}: {e}", provenance.display())))?;

    Ok(RunReport {
        mode,
        runtime: target.name(),
        stored_upload,
        channels: nch,
        samples,
        sample_rate_hz: out_rate,
        windows: schedule.len(),
        bytes,
        output,
        provenance,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn provenance_path(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(PROVENANCE_SUFFIX);
    output.with_file_name(name)
}

/// A trailing `resample` node, reduced to coprime factors.
struct Resample {
    up: usize,
    down: usize,
    filter: ResampleFilter,
    edge: EdgeMode,
    /// Input samples the resampling filter reaches on each side of an output sample.
    reach: usize,
}

impl Resample {
    fn new(up: usize, down: usize, beta: f64, edge: EdgeMode) -> Self {
        let g = gcd(up, down);
        let (up, down) = (up / g, down / g);
        let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
        Self { up, down, filter: ResampleFilter::Window(FirWindow::Kaiser { beta }), edge, reach: half_len.div_ceil(up) + 1 }
    }
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
