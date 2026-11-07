//! The executor: a recording through the chain's stages on the device, window by window, into the
//! output file.
//!
//! Every window crosses to the device once and back once ([`PipelineWorkspace`]: filters designed
//! and uploaded once, device buffers reused). Integer recordings cross as their stored integers and
//! are scaled on the device. Three threads overlap: the next window is read while the current one
//! runs ([`WindowLoader`]), and a writer thread takes downloaded windows through a bounded queue
//! ([`WRITE_QUEUE_WINDOWS`]) and writes them: raw float32 time-major, or Zarr chunks directly
//! ([`ZarrWriter`]). [`DataMovement`] counts the bytes of each step, [`StageTimings`] the time.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::time::{Duration, Instant};

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_core::{ComputeTarget, DspError, DspResult, HaloWindow, MemoryOrder, RecordingSource, SampleFormat, WindowLoader};
use dspk_io::{RawParams, ZarrWriter};
use dspk_ops::core::buffer;
use dspk_ops::core::layout::execute_gather_rows;
use dspk_ops::math::stored_word_bytes;
use dspk_ops::pipeline::{Pipeline, PipelineWorkspace};
use dspk_ops::resampler::{execute_resample_poly, resample_poly_len};

use crate::prepare::{default_open, prepare, ExecutedMode, Prepared};
use crate::spec::{OutputFormat, PipelineSpec};
use crate::PipelineError;

/// Appended to the output path for the canonical spec written next to it (provenance).
pub const PROVENANCE_SUFFIX: &str = ".pipeline.json";

/// Samples per parallel task when raw output frames are built.
const ENCODE_SAMPLES_PER_TASK: usize = 4096;

/// Downloaded windows waiting for the writer thread, at most: enough to absorb a slow write, and
/// a bound on the memory they hold (each is `channels × window` float32 values).
pub const WRITE_QUEUE_WINDOWS: usize = 2;

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

/// Where a run spent its time, in seconds. Reading, the device and the writer run on three
/// threads at once, so these add up to more than [`RunReport::seconds`]; the largest of
/// `read_wait`, `device` and `encode + disk` is the bottleneck.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    /// Device thread waiting for the next window from the reader (read-ahead thread): reading is
    /// the bottleneck when this is large.
    pub read_wait: f64,
    /// Upload, every stage, resampling and download, on the device thread.
    pub device: f64,
    /// Device thread waiting for room in the writer's queue: writing is the bottleneck when this is
    /// large.
    pub write_wait: f64,
    /// Writer thread: channel-major windows into the output layout (raw: time-major frames;
    /// Zarr: chunk buffers).
    pub encode: f64,
    /// Writer thread: handing bytes to the file system (raw writes, Zarr chunk stores).
    pub disk: f64,
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
    pub timings: StageTimings,
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
/// `dspk_synapse::io`) and `observe` called after each window: returning `false` cancels the run
/// ([`PipelineError::Cancelled`]; the unfinished output is removed, finished inputs stay).
pub fn run_with(
    spec: &PipelineSpec,
    base_dir: &Path,
    open: impl Fn(&Path, Option<&str>) -> DspResult<Box<dyn RecordingSource>>,
    mut observe: impl FnMut(&Progress) -> bool,
) -> Result<Vec<RunReport>, PipelineError> {
    if spec.writes() > 1 {
        // A branched spec: every branch from one read of the recording
        let mut ones = Vec::new();
        let mut prepared = Vec::new();
        for branch in spec.branches()? {
            let mut expanded = branch.expand_inputs(base_dir)?;
            if expanded.len() != 1 {
                return Err(PipelineError::Invalid("a branched spec reads one recording (no globs or input lists)".into()));
            }
            let one = expanded.remove(0);
            prepared.push(prepare(&one, base_dir, &open)?);
            ones.push(one);
        }
        return execute_fanout(&ones, prepared, |windows_done, windows, bytes| observe(&Progress { input: 0, inputs: 1, windows_done, windows, bytes }));
    }
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
    let target = match &spec.execution.runtime {
        Some(name) => ComputeTarget::parse(name).and_then(ComputeTarget::checked),
        None => ComputeTarget::from_env(),
    }
    .map_err(|e| PipelineError::Compute(e.to_string()))?;
    let mut p = p;
    p.fit(target)?;
    let Prepared {
        source, pipeline, resample, channels, range, unit, sample_rate, stored_format, stored_upload, mode, schedule,
        out_rate, out_samples, output, format, zarr, names, keep, ..
    } = p;
    let info = source.info();
    let nch = channels.len();
    // Channels written: the kept ones, gathered on the device before the download
    let out_nch = names.len();
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

    if let Some(dir) = output.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| PipelineError::Io(format!("{}: {e}", dir.display())))?;
    }
    let sink = match format {
        OutputFormat::Raw => {
            let file = File::create(&output).map_err(|e| PipelineError::Io(format!("{}: {e}", output.display())))?;
            Sink::Raw { file: BufWriter::new(file), frames: Vec::new(), parallel: zarr.parallel }
        }
        OutputFormat::Zarr => {
            let mut w = ZarrWriter::create(&output, out_nch, out_samples, out_rate, &unit, &zarr)?;
            w.set_channel_names(&names)?;
            Sink::Zarr(w)
        }
    };

    let mut bytes = DataMovement::default();
    let mut timings = StageTimings::default();
    let loader = WindowLoader::with_channels(source.as_ref(), channels.clone());
    let total_windows = schedule.len();
    let mut done = 0;
    let mut cancelled = false;
    let read_bytes = |win: &HaloWindow| (nch * win.read_len() * stored_format.bytes()) as u64;
    let origin = range.start;

    let (sent, written) = std::thread::scope(|scope| {
        let (queue, windows) = sync_channel::<Block>(WRITE_QUEUE_WINDOWS);
        let writer = scope.spawn(move || sink.drain(windows, out_nch));
        // Owns the queue: dropping it tells the writer no more blocks come
        let send = move |block: Block, timings: &mut StageTimings| -> DspResult<()> {
            let waited = Instant::now();
            // The writer only hangs up after an error, which the join below reports
            queue.send(block).map_err(|_| writer_stopped())?;
            timings.write_wait += waited.elapsed().as_secs_f64();
            Ok(())
        };
        // Downloads a processed window (resampled on the device first, when the chain ends with
        // one) and queues the samples it owns for the writer
        let keep_rows = keep.as_ref().map(|k| buffer::upload(workspace.client(), &k.iter().map(|&c| c as u32).collect::<Vec<u32>>()));
        let finish = |win: &HaloWindow, handle: Handle, client: &Client, bytes: &mut DataMovement| -> Block {
            let n = win.read_len();
            let handle = match &keep_rows {
                Some(rows) => {
                    let kept = buffer::empty::<f32>(client, out_nch * n);
                    execute_gather_rows::<f32>(client, &handle, rows, &kept, nch, out_nch, n);
                    kept
                }
                None => handle,
            };
            let nch = out_nch;
            let block = match &resample {
                None => Block { rows: buffer::download_prefix::<f32>(client, handle, nch * n), row_len: n, cols: win.valid_local.clone() },
                Some(r) => {
                    let out_len = resample_poly_len(n, r.up, r.down);
                    let out = buffer::empty::<f32>(client, nch * out_len);
                    execute_resample_poly::<f32>(client, &handle, &out, nch, n, r.up, r.down, &r.filter, r.edge);
                    // Output j of the window is input `read_start + j · down / up`
                    let first = ((win.valid_global.start - win.read_global.start) as usize) * r.up / r.down;
                    let to_end = resample_poly_len((win.valid_global.end - origin) as usize, r.up, r.down);
                    let start = ((win.valid_global.start - origin) as usize) * r.up / r.down;
                    let count = to_end - start;
                    Block { rows: buffer::download_prefix::<f32>(client, out, nch * out_len), row_len: out_len, cols: first..(first + count).min(out_len) }
                }
            };
            bytes.downloaded += (block.rows.len() * 4) as u64;
            bytes.written += (block.cols.len() * nch * 4) as u64;
            block
        };
        let mut idle = Instant::now();
        let streamed = if stored_upload {
            loader.stream_stored_owned_while(schedule.windows(), |win, stored| {
                let started = Instant::now();
                timings.read_wait += (started - idle).as_secs_f64();
                bytes.read += read_bytes(win);
                bytes.uploaded += stored_word_bytes(stored.len()) as u64;
                let handle = workspace.process_owned_stored_chunk_in_vram(stored, stored_format, win.read_len())?;
                let block = finish(win, handle, workspace.client(), &mut bytes);
                timings.device += started.elapsed().as_secs_f64();
                send(block, &mut timings)?;
                done += 1;
                cancelled = !observe(done, total_windows, bytes);
                idle = Instant::now();
                Ok(!cancelled)
            })
        } else {
            loader.stream_while(schedule.windows(), |win, samples| {
                let started = Instant::now();
                timings.read_wait += (started - idle).as_secs_f64();
                bytes.read += read_bytes(win);
                bytes.uploaded += (samples.len() * 4) as u64;
                let handle = workspace.process_chunk_in_vram(samples, win.read_len());
                let block = finish(win, handle, workspace.client(), &mut bytes);
                timings.device += started.elapsed().as_secs_f64();
                send(block, &mut timings)?;
                done += 1;
                cancelled = !observe(done, total_windows, bytes);
                idle = Instant::now();
                Ok(!cancelled)
            })
        };
        drop(send);
        // The writer's own error explains a failed send; report it first
        let written = writer.join().unwrap_or_else(|_| Err(DspError::Io("the output writer panicked".into())));
        (streamed, written)
    });
    let cleanup = || match format {
        OutputFormat::Raw => drop(std::fs::remove_file(&output)),
        OutputFormat::Zarr => drop(std::fs::remove_dir_all(&output)),
    };
    // A cancelled run leaves the output incomplete (a Zarr writer then reports missing samples)
    if cancelled {
        cleanup();
        return Err(PipelineError::Cancelled);
    }
    let stats = match (sent, written) {
        (Ok(()), Ok(stats)) => stats,
        // A failed send is explained by the writer's own error; any other error came first, and the
        // writer then stopped short of the end because of it
        (Err(e), Err(writer)) if e == writer_stopped() => {
            cleanup();
            return Err(writer.into());
        }
        (Err(e), _) | (Ok(()), Err(e)) => {
            cleanup();
            return Err(e.into());
        }
    };
    timings.encode = stats.encode.as_secs_f64();
    timings.disk = stats.disk.as_secs_f64();

    let samples = out_samples;
    if format == OutputFormat::Raw {
        let mut params = RawParams::new(out_nch, out_rate, SampleFormat::F32, MemoryOrder::TimeMajor);
        params.unit = unit;
        params.samples = Some(samples);
        params.write_sidecar(&output)?;
    }

    let provenance = provenance_path(&output);
    std::fs::write(&provenance, spec.to_json()).map_err(|e| PipelineError::Io(format!("{}: {e}", provenance.display())))?;

    Ok(RunReport {
        mode,
        runtime: target.name(),
        stored_upload,
        channels: out_nch,
        samples,
        sample_rate_hz: out_rate,
        windows: schedule.len(),
        bytes,
        output,
        provenance,
        seconds: started.elapsed().as_secs_f64(),
        timings,
    })
}

/// How many stages every branch starts with in common: the same nodes of the spec (so the same
/// stages), counted from the read.
fn shared_stages(prepared: &[Prepared]) -> usize {
    let Some(first) = prepared.first() else { return 0 };
    let mut n = 0;
    while n < first.stages.len() && prepared.iter().all(|p| p.stages.get(n).is_some_and(|(id, _)| *id == first.stages[n].0)) {
        n += 1;
    }
    n
}

/// The branches of a branched spec (each prepared as its own chain), from one read of the
/// recording: each window is read once and goes through every branch's device workspace to its
/// own writer. All branches run in window mode with the largest halo of any (or all in stream
/// mode when every branch can); resampling branches are not supported here.
fn execute_fanout(
    specs: &[PipelineSpec],
    prepared: Vec<Prepared>,
    mut observe: impl FnMut(usize, usize, DataMovement) -> bool,
) -> Result<Vec<RunReport>, PipelineError> {
    let started = Instant::now();
    // Whitening stages fitted first, on the device the branches run on
    let fit_target = match &specs[0].execution.runtime {
        Some(name) => ComputeTarget::parse(name).and_then(ComputeTarget::checked),
        None => ComputeTarget::from_env(),
    }
    .map_err(|e| PipelineError::Compute(e.to_string()))?;
    let mut prepared = prepared;
    for p in &mut prepared {
        p.fit(fit_target)?;
    }
    let invalid = |m: &str| Err(PipelineError::Invalid(m.to_string()));
    let first = &prepared[0];
    if prepared.iter().any(|p| p.channels != first.channels || p.range != first.range || p.stored_upload != first.stored_upload || p.stored_format != first.stored_format) {
        return invalid("the branches of one spec read the same recording the same way (channels, range, upload)");
    }
    if prepared.iter().any(|p| p.resample.is_some()) {
        return invalid("resample in a branched spec: write that branch with its own spec");
    }
    // One schedule for all: stream when every branch streams, else windows with the largest halo
    let stream = prepared.iter().all(|p| p.mode == ExecutedMode::Stream);
    let mut halo = (0usize, 0usize);
    if !stream {
        for p in &prepared {
            let (l, r) = match p.mode {
                ExecutedMode::Windows { left, right } => (left, right),
                ExecutedMode::Stream => p.pipeline.settling(p.sample_rate).map_err(|e| PipelineError::Invalid(e.to_string()))?,
            };
            halo = (halo.0.max(l), halo.1.max(r));
        }
    }
    let mode = if stream { ExecutedMode::Stream } else { ExecutedMode::Windows { left: halo.0, right: halo.1 } };
    let (channels, range, stored_upload, stored_format) = (first.channels.clone(), first.range.clone(), first.stored_upload, first.stored_format);
    let info = first.source.info();
    let nch = channels.len();
    let schedule = dspk_core::ChunkSchedule::new(range.clone(), specs[0].execution.chunk_samples.max(1) as u64, halo.0 as u64, halo.1 as u64, info.samples);
    let target = match &specs[0].execution.runtime {
        Some(name) => ComputeTarget::parse(name).and_then(ComputeTarget::checked),
        None => ComputeTarget::from_env(),
    }
    .map_err(|e| PipelineError::Compute(e.to_string()))?;
    let capacity = schedule.max_read_samples().max(1);
    // A workspace for `pipeline`; `takes_input`: it receives the read windows (the stored values'
    // scaling is its job), else the device output of the shared stages
    let make = |pipeline: Pipeline, takes_input: bool| -> Result<PipelineWorkspace<f32>, PipelineError> {
        let client = target.client().map_err(|e| PipelineError::Compute(e.to_string()))?;
        let mut ws = if stream {
            PipelineWorkspace::<f32>::new_stateful(client, pipeline, nch, capacity, first.sample_rate)
        } else {
            PipelineWorkspace::<f32>::new(client, pipeline, nch, capacity, first.sample_rate)
        }
        .map_err(|e| PipelineError::Invalid(e.to_string()))?;
        if stored_upload && takes_input {
            let gains: Vec<f32> = channels.iter().map(|&c| info.channels[c].gain).collect();
            let offsets: Vec<f32> = channels.iter().map(|&c| info.channels[c].offset).collect();
            ws.set_stored_scaling(&gains, &offsets);
        }
        Ok(ws)
    };
    // The stages every branch starts with (the same nodes of the spec) run once per window; each
    // branch then runs only its own stages, on the shared result in device memory
    let shared = shared_stages(&prepared);
    let mut prefix = match shared {
        0 => None,
        n => Some(make(Pipeline::with_stages(first.stages[..n].iter().map(|(_, s)| s.clone()).collect()), true)?),
    };
    let mut workspaces = Vec::with_capacity(prepared.len());
    let mut sinks = Vec::with_capacity(prepared.len());
    for p in &prepared {
        let pipeline = if shared > 0 { Pipeline::with_stages(p.stages[shared..].iter().map(|(_, s)| s.clone()).collect()) } else { p.pipeline.clone() };
        workspaces.push(make(pipeline, prefix.is_none())?);
        if let Some(dir) = p.output.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| PipelineError::Io(format!("{}: {e}", dir.display())))?;
        }
        sinks.push(match p.format {
            OutputFormat::Raw => {
                let file = File::create(&p.output).map_err(|e| PipelineError::Io(format!("{}: {e}", p.output.display())))?;
                Sink::Raw { file: BufWriter::new(file), frames: Vec::new(), parallel: p.zarr.parallel }
            }
            OutputFormat::Zarr => {
                let mut w = ZarrWriter::create(&p.output, p.names.len(), p.out_samples, p.out_rate, &p.unit, &p.zarr)?;
                w.set_channel_names(&p.names)?;
                Sink::Zarr(w)
            }
        });
    }

    let keep_rows: Vec<Option<Handle>> = prepared
        .iter()
        .zip(&workspaces)
        .map(|(p, ws)| p.keep.as_ref().map(|k| buffer::upload(ws.client(), &k.iter().map(|&c| c as u32).collect::<Vec<u32>>())))
        .collect();
    let mut bytes = DataMovement::default();
    let mut timings = StageTimings::default();
    let loader = WindowLoader::with_channels(first.source.as_ref(), channels.clone());
    let total_windows = schedule.len();
    let mut done = 0;
    let mut cancelled = false;
    let (sent, written) = std::thread::scope(|scope| {
        let mut queues = Vec::new();
        let mut writers = Vec::new();
        for (sink, p) in sinks.into_iter().zip(&prepared) {
            let (queue, blocks) = sync_channel::<Block>(WRITE_QUEUE_WINDOWS);
            queues.push(queue);
            let out_nch = p.names.len();
            writers.push(scope.spawn(move || sink.drain(blocks, out_nch)));
        }
        let mut idle = Instant::now();
        let mut each = |win: &HaloWindow, input: Input<'_>, bytes: &mut DataMovement, timings: &mut StageTimings| -> DspResult<bool> {
            let t = Instant::now();
            timings.read_wait += (t - idle).as_secs_f64();
            bytes.read += (nch * win.read_len() * stored_format.bytes()) as u64;
            let n = win.read_len();
            let last = workspaces.len() - 1;
            let mut owned = match &input {
                Input::Stored(v) => Some(v.clone()),
                Input::Values(_) => None,
            };
            // The shared stages, once: their result stays on the device for every branch
            let shared_out = match prefix.as_mut() {
                Some(pw) => Some(match &input {
                    Input::Stored(_) => {
                        let stored = owned.take().unwrap_or_default();
                        bytes.uploaded += stored_word_bytes(stored.len()) as u64;
                        pw.process_owned_stored_chunk_in_vram(stored, stored_format, n)?
                    }
                    Input::Values(samples) => {
                        bytes.uploaded += (samples.len() * 4) as u64;
                        pw.process_chunk_in_vram(samples, n)
                    }
                }),
                None => None,
            };
            for (i, ws) in workspaces.iter_mut().enumerate() {
                let handle = if let Some(shared) = &shared_out {
                    ws.process_handle(shared, n)
                } else {
                    match &input {
                    Input::Stored(_) => {
                        // Each branch takes the stored values (the last one takes them as they are)
                        let stored = if i == last { owned.take().unwrap_or_default() } else { owned.clone().unwrap_or_default() };
                        bytes.uploaded += stored_word_bytes(stored.len()) as u64;
                        ws.process_owned_stored_chunk_in_vram(stored, stored_format, n)?
                    }
                    Input::Values(samples) => {
                        bytes.uploaded += (samples.len() * 4) as u64;
                        ws.process_chunk_in_vram(samples, n)
                    }
                    }
                };
                // A branch that writes fewer channels gathers them on the device first
                let out_nch = prepared[i].names.len();
                let handle = match &keep_rows[i] {
                    Some(rows) => {
                        let kept = buffer::empty::<f32>(ws.client(), out_nch * n);
                        execute_gather_rows::<f32>(ws.client(), &handle, rows, &kept, nch, out_nch, n);
                        kept
                    }
                    None => handle,
                };
                let rows = buffer::download_prefix::<f32>(ws.client(), handle, out_nch * n);
                bytes.downloaded += (rows.len() * 4) as u64;
                bytes.written += (win.valid_local.len() * out_nch * 4) as u64;
                let waited = Instant::now();
                queues[i].send(Block { rows, row_len: n, cols: win.valid_local.clone() }).map_err(|_| writer_stopped())?;
                timings.write_wait += waited.elapsed().as_secs_f64();
            }
            timings.device += t.elapsed().as_secs_f64();
            done += 1;
            cancelled = !observe(done, total_windows, *bytes);
            idle = Instant::now();
            Ok(!cancelled)
        };
        let streamed = if stored_upload {
            loader.stream_stored_owned_while(schedule.windows(), |win, stored| each(win, Input::Stored(stored), &mut bytes, &mut timings))
        } else {
            loader.stream_while(schedule.windows(), |win, samples| each(win, Input::Values(samples), &mut bytes, &mut timings))
        };
        drop(queues);
        let mut written = Vec::new();
        for w in writers {
            written.push(w.join().unwrap_or_else(|_| Err(DspError::Io("the output writer panicked".into()))));
        }
        (streamed, written)
    });
    let cleanup = || {
        for p in &prepared {
            match p.format {
                OutputFormat::Raw => drop(std::fs::remove_file(&p.output)),
                OutputFormat::Zarr => drop(std::fs::remove_dir_all(&p.output)),
            }
        }
    };
    if cancelled {
        cleanup();
        return Err(PipelineError::Cancelled);
    }
    let mut stats = Vec::new();
    for w in written {
        match w {
            Ok(s) => stats.push(s),
            Err(e) => {
                cleanup();
                return Err(e.into());
            }
        }
    }
    if let Err(e) = sent {
        cleanup();
        return Err(e.into());
    }
    timings.encode = stats.iter().map(|s| s.encode.as_secs_f64()).sum();
    timings.disk = stats.iter().map(|s| s.disk.as_secs_f64()).sum();
    let mut reports = Vec::new();
    for (p, one) in prepared.iter().zip(specs) {
        if p.format == OutputFormat::Raw {
            let mut params = RawParams::new(p.names.len(), p.out_rate, SampleFormat::F32, MemoryOrder::TimeMajor);
            params.unit = p.unit.clone();
            params.samples = Some(p.out_samples);
            params.write_sidecar(&p.output)?;
        }
        let provenance = provenance_path(&p.output);
        std::fs::write(&provenance, one.to_json()).map_err(|e| PipelineError::Io(format!("{}: {e}", provenance.display())))?;
        reports.push(RunReport {
            mode,
            runtime: target.name(),
            stored_upload,
            channels: p.names.len(),
            samples: p.out_samples,
            sample_rate_hz: p.out_rate,
            windows: schedule.len(),
            bytes,
            output: p.output.clone(),
            provenance,
            seconds: started.elapsed().as_secs_f64(),
            timings,
        });
    }
    Ok(reports)
}

/// A window as read: stored values (bytes) or scaled values.
enum Input<'a> {
    Stored(Vec<u8>),
    Values(&'a [f32]),
}

fn writer_stopped() -> DspError {
    DspError::Io("the output writer stopped".into())
}

/// Columns `cols` of a processed `[channels, row_len]` window, for the writer.
struct Block {
    rows: Vec<f32>,
    row_len: usize,
    cols: std::ops::Range<usize>,
}

/// Where the writer thread puts blocks.
enum Sink {
    /// Raw float32, time-major (frames of all channels, sample after sample).
    Raw { file: BufWriter<File>, frames: Vec<u8>, parallel: bool },
    Zarr(ZarrWriter),
}

/// Time the writer thread spent on its two steps.
#[derive(Default)]
struct WriterStats {
    encode: Duration,
    disk: Duration,
}

impl Sink {
    /// Writes every block from `blocks` until the sender hangs up, then completes the file.
    fn drain(mut self, blocks: Receiver<Block>, nch: usize) -> DspResult<WriterStats> {
        let mut stats = WriterStats::default();
        for Block { rows, row_len, cols } in blocks {
            match &mut self {
                Sink::Raw { file, frames, parallel } => {
                    // Frames (all channels of one sample), blocks of samples per task: in
                    // parallel by default
                    use rayon::prelude::*;
                    let clock = Instant::now();
                    let frame = nch * size_of::<f32>();
                    frames.resize(cols.len() * frame, 0);
                    let first = cols.start;
                    let encode = |(block, out): (usize, &mut [u8])| {
                        for (j, f) in out.chunks_exact_mut(frame).enumerate() {
                            let t = first + block * ENCODE_SAMPLES_PER_TASK + j;
                            for (c, v) in f.chunks_exact_mut(size_of::<f32>()).enumerate() {
                                v.copy_from_slice(&rows[c * row_len + t].to_le_bytes());
                            }
                        }
                    };
                    if *parallel {
                        frames.par_chunks_mut(frame * ENCODE_SAMPLES_PER_TASK).enumerate().for_each(encode);
                    } else {
                        frames.chunks_mut(frame * ENCODE_SAMPLES_PER_TASK).enumerate().for_each(encode);
                    }
                    stats.encode += clock.elapsed();
                    let clock = Instant::now();
                    file.write_all(frames)?;
                    stats.disk += clock.elapsed();
                }
                // Copying into chunk buffers is cheap next to compressing and storing them (in
                // parallel, inside `push`): counted as disk
                Sink::Zarr(writer) => {
                    let clock = Instant::now();
                    writer.push(&rows, row_len, cols)?;
                    stats.disk += clock.elapsed();
                }
            }
        }
        let clock = Instant::now();
        match self {
            Sink::Raw { mut file, .. } => file.flush()?,
            Sink::Zarr(writer) => writer.finish()?,
        }
        stats.disk += clock.elapsed();
        Ok(stats)
    }
}

fn provenance_path(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(PROVENANCE_SUFFIX);
    output.with_file_name(name)
}
