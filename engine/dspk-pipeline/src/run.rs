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

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_core::{ComputeTarget, DspResult, HaloWindow, MemoryOrder, RecordingSource, SampleFormat, WindowLoader};
use dspk_io::{RawParams, RawRecording};
use dspk_ops::core::buffer;
use dspk_ops::math::stored_word_bytes;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::resampler::{resample_poly, resample_poly_len};

use crate::prepare::{default_open, prepare, ExecutedMode, Prepared};
use crate::spec::{OutputFormat, PipelineSpec};
use crate::PipelineError;

/// Appended to the output path for the canonical spec written next to it (provenance).
pub const PROVENANCE_SUFFIX: &str = ".pipeline.json";

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

/// Where a run is, passed to the observer after each window.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// Input being processed (0-based) and the number of inputs.
    pub input: usize,
    pub inputs: usize,
    /// Windows of this input done, and its total.
    pub windows_done: usize,
    pub windows: usize,
    pub bytes: DataMovement,
}

/// Runs `spec` once per input, resolving relative paths against `base_dir`, opening recordings
/// with `dspk_io::open` (generic formats). One report per input, in order.
pub fn run(spec: &PipelineSpec, base_dir: &Path) -> Result<Vec<RunReport>, PipelineError> {
    run_with(spec, base_dir, default_open, |_| true)
}

/// [`run`] with recordings opened by `open` (e.g. a crate that adds formats, such as
/// `dspk-synapse-io`) and `observe` called after each window: returning `false` cancels the run
/// ([`PipelineError::Cancelled`]; the unfinished output is removed, finished inputs stay).
pub fn run_with(
    spec: &PipelineSpec,
    base_dir: &Path,
    open: impl Fn(&Path, Option<&str>) -> DspResult<Box<dyn RecordingSource>>,
    mut observe: impl FnMut(&Progress) -> bool,
) -> Result<Vec<RunReport>, PipelineError> {
    let specs = spec.expand_inputs(base_dir)?;
    let inputs = specs.len();
    let mut reports = Vec::with_capacity(inputs);
    for (input, one) in specs.iter().enumerate() {
        let prepared = prepare(one, base_dir, &open)?;
        reports.push(execute(one, prepared, |windows_done, windows, bytes| {
            observe(&Progress { input, inputs, windows_done, windows, bytes })
        })?);
    }
    Ok(reports)
}

/// Runs one prepared input on the device.
fn execute(
    spec: &PipelineSpec,
    p: Prepared,
    mut observe: impl FnMut(usize, usize, DataMovement) -> bool,
) -> Result<RunReport, PipelineError> {
    let started = Instant::now();
    let Prepared {
        source, pipeline, resample, channels, range, unit, sample_rate, stored_format, stored_upload, mode, chunk, schedule,
        out_rate, out_samples, output, format, ..
    } = p;
    let info = source.info();
    let nch = channels.len();
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
    let total_windows = schedule.len();
    let mut done = 0;
    let mut cancelled = false;
    let read_bytes = |win: &HaloWindow| (nch * win.read_len() * stored_format.bytes()) as u64;
    if stored_upload {
        loader.stream_stored_owned_while(schedule.windows(), |win, stored| {
            bytes.read += read_bytes(win);
            bytes.uploaded += stored_word_bytes(stored.len()) as u64;
            let handle = workspace.process_owned_stored_chunk_in_vram(stored, stored_format, win.read_len())?;
            finish(win, handle, workspace.client(), &mut bytes)?;
            done += 1;
            cancelled = !observe(done, total_windows, bytes);
            Ok(!cancelled)
        })?;
    } else {
        loader.stream_while(schedule.windows(), |win, samples| {
            bytes.read += read_bytes(win);
            bytes.uploaded += (samples.len() * 4) as u64;
            let handle = workspace.process_chunk_in_vram(samples, win.read_len());
            finish(win, handle, workspace.client(), &mut bytes)?;
            done += 1;
            cancelled = !observe(done, total_windows, bytes);
            Ok(!cancelled)
        })?;
    }
    drop(finish);
    drop(emit);
    writer.flush().map_err(|e| PipelineError::Io(format!("{}: {e}", raw_path.display())))?;
    drop(writer);
    if cancelled {
        let _ = std::fs::remove_file(&raw_path);
        return Err(PipelineError::Cancelled);
    }

    let samples = out_samples;
    let mut params = RawParams::new(nch, out_rate, SampleFormat::F32, MemoryOrder::TimeMajor);
    params.unit = unit;
    params.samples = Some(samples);
    params.write_sidecar(&raw_path)?;

    if format == OutputFormat::Zarr {
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
