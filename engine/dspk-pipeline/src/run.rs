//! The executor: a recording through the chain's stages on the device, window by window, into the
//! output file.
//!
//! Every window crosses to the device once and back once ([`PipelineWorkspace`]: filters designed
//! and uploaded once, device buffers reused). The next window is read while the current one runs
//! ([`WindowLoader`]). Results are appended to a raw float32 time-major file; Zarr output is
//! converted from it at the end.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dspk_core::compute::device_elements;
use dspk_core::{ChunkSchedule, ComputeTarget, DspResult, MemoryOrder, RecordingSource, SampleFormat, WindowLoader};
use dspk_io::{RawParams, RawRecording};
use dspk_ops::filter::FilterMode;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};

use crate::spec::{ExecMode, Node, OutputFormat, PipelineSpec};
use crate::PipelineError;

/// Appended to the output path for the canonical spec written next to it (provenance).
pub const PROVENANCE_SUFFIX: &str = ".pipeline.json";

/// How the chain was run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutedMode {
    /// Windows in order with filter state carried: equal to processing the whole recording at once.
    Stream,
    /// Independent windows with `left` / `right` halos of context.
    Windows { left: usize, right: usize },
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub mode: ExecutedMode,
    pub runtime: &'static str,
    pub channels: usize,
    pub samples: u64,
    pub sample_rate_hz: f64,
    pub windows: usize,
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
    let Node::Read { path: input, source } = chain.read.1 else { unreachable!("chain starts with read") };
    let Node::Write { path: output, format } = chain.write.1 else { unreachable!("chain ends with write") };
    let (input, output) = (base_dir.join(input), base_dir.join(output));

    let mut stages = Vec::with_capacity(chain.stages.len());
    for (id, node) in &chain.stages {
        let stage = node.stage().map_err(|e| PipelineError::Invalid(format!("node {id:?}: {e}")))?;
        stages.push(stage.expect("processing nodes have a stage"));
    }
    let pipeline = Pipeline::with_stages(stages);

    let source = open(&input, source.as_deref())?;
    let info = source.info();
    let (channels, total, sample_rate) = (info.channel_count(), info.samples, info.sample_rate_hz());
    let unit = info.channels.first().map(|c| c.unit.clone()).unwrap_or_default();
    pipeline.validate(sample_rate).map_err(|e| PipelineError::Invalid(e.to_string()))?;

    let mode = resolve_mode(spec.execution.mode, &pipeline, sample_rate)?;
    let chunk = spec.execution.chunk_samples.max(1);
    let (left, right) = match mode {
        ExecutedMode::Stream => (0, 0),
        ExecutedMode::Windows { left, right } => (left, right),
    };
    let schedule = ChunkSchedule::full_recording(total, chunk as u64, left as u64, right as u64);
    device_elements("pipeline window [channels, samples]", &[channels, schedule.max_read_samples()])?;

    let target = match &spec.execution.runtime {
        Some(name) => ComputeTarget::parse(name).and_then(ComputeTarget::checked),
        None => ComputeTarget::from_env(),
    }
    .map_err(|e| PipelineError::Compute(e.to_string()))?;
    let client = target.client().map_err(|e| PipelineError::Compute(e.to_string()))?;
    let capacity = schedule.max_read_samples().max(1);
    let mut workspace = match mode {
        ExecutedMode::Stream => PipelineWorkspace::<f32>::new_stateful(client, pipeline, channels, capacity, sample_rate),
        ExecutedMode::Windows { .. } => PipelineWorkspace::<f32>::new(client, pipeline, channels, capacity, sample_rate),
    }
    .map_err(|e| PipelineError::Invalid(e.to_string()))?;

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

    let mut processed = Vec::new();
    let mut frames = Vec::new();
    WindowLoader::new(source.as_ref()).stream(schedule.windows(), |win, samples| {
        let n = win.read_len();
        processed.resize(channels * n, 0.0f32);
        workspace.process_chunk(samples, n, &mut processed);
        // The window's own samples, channel-major → time-major
        let valid = win.valid_local.clone();
        frames.clear();
        frames.reserve(valid.len() * channels * 4);
        for t in valid {
            for c in 0..channels {
                frames.extend_from_slice(&processed[c * n + t].to_le_bytes());
            }
        }
        writer.write_all(&frames)?;
        Ok(())
    })?;
    writer.flush().map_err(|e| PipelineError::Io(format!("{}: {e}", raw_path.display())))?;
    drop(writer);

    let mut params = RawParams::new(channels, sample_rate, SampleFormat::F32, MemoryOrder::TimeMajor);
    params.unit = unit;
    params.samples = Some(total);
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
        channels,
        samples: total,
        sample_rate_hz: sample_rate,
        windows: schedule.len(),
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

/// `stream` needs every filter forward (zero-phase needs future samples) and no stencil stage
/// (median, Gaussian, Teager–Kaiser read neighbours across window edges and carry no state).
fn resolve_mode(requested: ExecMode, pipeline: &Pipeline, sample_rate: f64) -> Result<ExecutedMode, PipelineError> {
    let blocker = pipeline.stages().iter().find_map(|stage| match stage {
        PipelineStage::Filter(spec) if spec.mode == FilterMode::ForwardBackward => Some("a zero-phase (forward_backward) filter"),
        PipelineStage::Median { .. } => Some("a median stage"),
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
