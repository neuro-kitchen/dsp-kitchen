//! `benchmark`: device timings. Every timed region is synchronised with the compute client
//! (`dsp_core::compute::bench`), so kernel time is execution time, not enqueueing.
//!
//! - `pipeline`: the production path, a `PipelineWorkspace` (upload, stages, download).
//! - `sweep`: the same stages across channel counts.
//! - `suite`: every kernel family and streaming detection, with a JSON report.
//!
//! Signals come from dsp-io's `SyntheticRecording`; detection settings from
//! `StreamingDetectionConfig::default()`; matching settings from dsp-synapse's defaults.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, Subcommand};
use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::filter::non_linear::{MEDIAN9_RADIUS, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dsp_base::filter::{execute_fir, execute_median, execute_teager_kaiser, DeviceFilter, FilterMode, FilterSpec, FIR_DEFAULT_EDGE};
use dsp_base::math::{execute_channel_noise_std, execute_clamp, execute_scaling, execute_unpack_stored, stored_words};
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::execute_direct_car;
use dsp_core::compute::bench::{sync, time_device};
use dsp_core::compute::{ComputeTarget, ComputeTask};
use dsp_core::{MemoryOrder, MemoryRecording, RecordingSource, SampleFormat};
use dsp_io::neuro::probe::{precompute_knn_table, Position3D, SensorLayout, SensorSite};
use dsp_io::{SyntheticParams, SyntheticRecording};
use dsp_synapse::sorting::matching_pursuit::{
    DEFAULT_MAX_AMPLITUDE_SCALE, DEFAULT_MAX_PASSES, DEFAULT_MIN_AMPLITUDE_SCALE, DEFAULT_MIN_EXPLAINED_ENERGY_UV2,
};
use dsp_synapse::{
    deduplicate_spikes_spatial, execute_detect_spikes_in_vram, execute_extract_sinc_in_vram, execute_reduce_templates_in_vram,
    match_spikes_matching_pursuit, SpikeSpacing, StreamingDetectionConfig, StreamingDetector,
};
use serde::Serialize;

/// Defaults: a Neuropixels-sized probe for one second at 30 kHz.
const DEFAULT_CHANNELS: usize = 384;
const DEFAULT_SAMPLES: usize = 30_000;
const DEFAULT_SAMPLE_RATE_HZ: f64 = 30_000.0;
const DEFAULT_ITERATIONS: usize = 5;
/// Live-pipeline chain: µV per int16 step of a typical headstage, power-line notch.
const STEP_UV: f32 = 0.195;
const LINE_HZ: f64 = 60.0;
const NOTCH_Q: f64 = 30.0;
/// Spike band of extracellular recordings (SpikeInterface's default band-pass).
const SPIKE_BAND_HZ: (f64, f64) = (300.0, 6_000.0);
/// Moving-average FIR length timed by the suite.
const FIR_TAPS: usize = 32;
/// Channel counts of `sweep`.
const SWEEP_CHANNELS: [usize; 8] = [1, 4, 16, 32, 64, 128, 384, 1024];
/// Synthetic probe for the suite: two columns, this pitch (µm).
const PROBE_COLUMN_PITCH_UM: f32 = 32.0;
const PROBE_ROW_PITCH_UM: f32 = 20.0;
const PROBE_COLUMNS: usize = 2;
/// One simulated unit per this many channels, at most [`MAX_UNITS`].
const CHANNELS_PER_UNIT: usize = 4;
const MAX_UNITS: usize = 64;
/// Matching pursuit is timed on at most this many channels and seconds.
const MATCHING_CHANNELS: usize = 32;
const MATCHING_SEC: f64 = 1.0;
/// Streaming detection is timed on at most this many seconds.
const DETECTION_MAX_SEC: f64 = 10.0;
const MS_PER_S: f64 = 1e3;
const PER_MEGA: f64 = 1e6;

#[derive(Args, Debug, Clone)]
pub struct Shape {
    /// Channels
    #[arg(short, long, default_value_t = DEFAULT_CHANNELS)]
    channels: usize,
    /// Samples per channel
    #[arg(short, long, default_value_t = DEFAULT_SAMPLES)]
    samples: usize,
    /// Sample rate (Hz): sets the filters and the real-time factor
    #[arg(short = 'r', long, default_value_t = DEFAULT_SAMPLE_RATE_HZ)]
    sample_rate: f64,
    /// Timed repetitions (the median is reported)
    #[arg(short, long, default_value_t = DEFAULT_ITERATIONS)]
    iterations: usize,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// A PipelineWorkspace (scale → notch → TKEO): upload, stages, download
    Pipeline {
        #[command(flatten)]
        shape: Shape,
        /// Upload int16 stored samples (scaled on the device) instead of f32
        #[arg(long)]
        int16: bool,
        /// Write the output once as raw binary + sidecar (timed separately)
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// The pipeline's stages across channel counts
    Sweep {
        #[command(flatten)]
        shape: Shape,
    },
    /// Every kernel family and streaming detection; writes a JSON report
    Suite {
        #[command(flatten)]
        shape: Shape,
        /// Folder for the report (`<unix time>-<runtime>.json`)
        #[arg(long)]
        report_dir: PathBuf,
    },
}

#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    mode: Mode,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * MS_PER_S
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Samples per second of input, in millions.
fn msamples_per_s(values: usize, t: Duration) -> f64 {
    values as f64 / t.as_secs_f64() / PER_MEGA
}

/// Channel-major samples of a synthetic recording (noise, line noise, drifting units).
fn synthetic(channels: usize, samples: usize, sample_rate: f64) -> anyhow::Result<(SyntheticRecording, Vec<f32>)> {
    let source = SyntheticRecording::new(SyntheticParams {
        channels,
        sample_rate_hz: sample_rate,
        duration_sec: samples as f64 / sample_rate,
        units: (channels / CHANNELS_PER_UNIT).clamp(1, MAX_UNITS),
        ..Default::default()
    })?;
    let n = source.info().samples.min(samples as u64);
    let mut data = vec![0.0f32; channels * n as usize];
    source.read(&(0..channels).collect::<Vec<_>>(), 0..n, &mut data)?;
    Ok((source, data))
}

/// The live chain: scale stored steps to µV, remove the power line, emphasise spikes.
fn live_chain() -> Pipeline {
    Pipeline::with_stages(vec![PipelineStage::Scale { alpha: STEP_UV, beta: 0.0 }, PipelineStage::notch(LINE_HZ, NOTCH_Q), PipelineStage::teager_kaiser()])
}

pub fn run(target: ComputeTarget, args: &BenchmarkArgs) -> anyhow::Result<()> {
    struct Task<'a>(&'a Mode);
    impl ComputeTask for Task<'_> {
        type Output = anyhow::Result<()>;
        fn run(self, client: Client) -> Self::Output {
            match self.0 {
                Mode::Pipeline { shape, int16, save } => pipeline_on(client, shape, *int16, save.as_deref()),
                Mode::Sweep { shape } => sweep_on(client, shape),
                Mode::Suite { shape, report_dir } => suite_on(client, shape, report_dir),
            }
        }
    }
    target.run(Task(&args.mode))?
}

// ------------------------------------------------------------------------------------------------
// Pipeline
// ------------------------------------------------------------------------------------------------

fn pipeline_on(client: Client, shape: &Shape, int16: bool, save: Option<&Path>) -> anyhow::Result<()> {
    let Shape { channels, samples, sample_rate, iterations } = *shape;
    let (_, values) = synthetic(channels, samples, sample_rate)?;
    let samples = values.len() / channels.max(1);
    let total = channels * samples;
    let mut workspace = PipelineWorkspace::<f32>::new(client.clone(), live_chain(), channels, samples, sample_rate)?;
    // int16 stored steps of the same signal; the workspace scales them on the device
    let stored: Vec<u8> = values.iter().flat_map(|v| ((v / STEP_UV).round() as i16).to_le_bytes()).collect();
    if int16 {
        workspace.set_stored_scaling(&vec![STEP_UV; channels], &vec![0.0; channels]);
    }
    let upload_bytes = if int16 { stored.len() } else { total * std::mem::size_of::<f32>() };
    println!("Pipeline: {channels} ch × {samples} samples at {sample_rate} Hz, {} upload ({upload_bytes} bytes), {}", if int16 { "int16" } else { "f32" }, client.name());
    println!("Stages: {:?}", workspace.pipeline().stages());

    let run_once = |workspace: &mut PipelineWorkspace<f32>| -> anyhow::Result<Handle> {
        Ok(if int16 { workspace.process_stored_chunk_in_vram(&stored, SampleFormat::I16, samples)? } else { workspace.process_chunk_in_vram(&values, samples) })
    };
    // Warm-up: compilation and autotuning
    run_once(&mut workspace)?;
    sync(&client);

    let (mut end_to_end, mut download) = (Vec::new(), Vec::new());
    let mut last = Vec::new();
    for _ in 0..iterations.max(1) {
        let t0 = Instant::now();
        let out = run_once(&mut workspace)?;
        sync(&client);
        let t1 = Instant::now();
        last = client.read_one_unchecked(out).to_vec();
        let t2 = Instant::now();
        end_to_end.push(t1 - t0);
        download.push(t2 - t1);
    }
    let processed = median(end_to_end);
    println!("  upload + stages  {:>9.3} ms   {:.0} Msamples/s, {:.1}× real time", ms(processed), msamples_per_s(total, processed), samples as f64 / sample_rate / processed.as_secs_f64());
    println!("  download         {:>9.3} ms", ms(median(download)));

    if let Some(path) = save {
        let output = f32::from_bytes(&last).to_vec();
        let recording = MemoryRecording::new("benchmark output", output, channels, sample_rate)?;
        let t = Instant::now();
        dsp_io::write_raw(&recording, path, SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, samples, |_, _| {})?;
        println!("  save             {:>9.3} ms → {} (+ sidecar)", ms(t.elapsed()), path.display());
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Sweep
// ------------------------------------------------------------------------------------------------

fn sweep_on(client: Client, shape: &Shape) -> anyhow::Result<()> {
    let Shape { samples, sample_rate, iterations, .. } = *shape;
    println!("Channel sweep: live chain on {samples} samples per channel at {sample_rate} Hz, {}", client.name());
    println!("{:>8} | {:>12} | {:>10} | {:>12} | {:>9}", "channels", "values", "stages ms", "Msamples/s", "realtime");
    for channels in SWEEP_CHANNELS {
        let total = channels * samples;
        let input = client.create_from_slice(f32::as_bytes(&vec![0.0f32; total]));
        let mut workspace = PipelineWorkspace::<f32>::new(client.clone(), live_chain(), channels, samples, sample_rate)?;
        let t = time_device(&client, iterations, || {
            workspace.process_handle(&input, samples);
        });
        println!("{channels:>8} | {total:>12} | {:>10.3} | {:>12.0} | {:>8.1}×", ms(t), msamples_per_s(total, t), samples as f64 / sample_rate / t.as_secs_f64());
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Suite
// ------------------------------------------------------------------------------------------------

#[derive(Serialize)]
struct SuiteRow {
    kernel: String,
    ms: f64,
    /// Input samples processed per second (millions), when meaningful.
    msamples_per_s: Option<f64>,
    note: String,
}

#[derive(Serialize)]
struct Hardware {
    runtime: String,
    plane_size_max: u32,
    max_units_per_cube: u32,
    num_cpu_cores: Option<u32>,
    num_streaming_multiprocessors: Option<u32>,
}

#[derive(Serialize)]
struct DetectionRow {
    channels: usize,
    seconds: f64,
    /// First run, including kernel compilation and autotuning.
    cold_wall_s: f64,
    /// Second run on the same client (steady state).
    wall_s: f64,
    msamples_per_s: f64,
    real_time_factor: f64,
    spikes: u64,
}

#[derive(Serialize)]
struct SuiteReport {
    unix_time: u64,
    hardware: Hardware,
    channels: usize,
    samples: usize,
    sample_rate_hz: f64,
    iterations: usize,
    kernels: Vec<SuiteRow>,
    streaming_detection: DetectionRow,
}

/// A two-column layout for any channel count.
fn columns_probe(channels: usize) -> SensorLayout {
    let sites = (0..channels)
        .map(|c| {
            let (col, row) = (c % PROBE_COLUMNS, c / PROBE_COLUMNS);
            SensorSite::new(c, Position3D::new(col as f32 * PROBE_COLUMN_PITCH_UM, row as f32 * PROBE_ROW_PITCH_UM, 0.0), 0)
        })
        .collect();
    SensorLayout::new("benchmark columns", sites)
}

fn suite_on(client: Client, shape: &Shape, report_dir: &Path) -> anyhow::Result<()> {
    let Shape { channels, samples, sample_rate: fs, iterations } = *shape;
    let hw = &client.properties().hardware;
    let hardware = Hardware {
        runtime: client.name().to_string(),
        plane_size_max: hw.plane_size_max,
        max_units_per_cube: hw.max_units_per_cube,
        num_cpu_cores: hw.num_cpu_cores,
        num_streaming_multiprocessors: hw.num_streaming_multiprocessors,
    };
    let (source, host) = synthetic(channels, samples, fs)?;
    let samples = host.len() / channels.max(1);
    let total = channels * samples;
    println!("Kernel suite: {channels} ch × {samples} samples ({:.1} s at {fs} Hz), median of {iterations}, {}", samples as f64 / fs, hardware.runtime);

    let mut rows: Vec<SuiteRow> = Vec::new();
    let mut add = |kernel: &str, t: Duration, values: Option<usize>, note: String| {
        let row = SuiteRow { kernel: kernel.into(), ms: ms(t), msamples_per_s: values.map(|n| msamples_per_s(n, t)), note };
        match row.msamples_per_s {
            Some(r) => println!("  {:<30} {:>10.3} ms  {:>9.0} Msamples/s  {}", row.kernel, row.ms, r, row.note),
            None => println!("  {:<30} {:>10.3} ms  {:>20}  {}", row.kernel, row.ms, "", row.note),
        }
        rows.push(row);
    };
    let input = client.create_from_slice(f32::as_bytes(&host));
    let output = client.empty(total * std::mem::size_of::<f32>());

    add("scale", time_device(&client, iterations, || execute_scaling::<f32>(&client, &input, &output, total, STEP_UV, 0.0)), Some(total), String::new());
    let (lo, hi) = host.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    add("clamp", time_device(&client, iterations, || execute_clamp::<f32>(&client, &input, &output, total, lo / 2.0, hi / 2.0)), Some(total), "to half the range".into());
    add("common reference (direct)", time_device(&client, iterations, || execute_direct_car::<f32>(&client, &input, &output, channels, samples)), Some(total), String::new());
    let median_width = 2 * MEDIAN9_RADIUS + 1;
    add("median", time_device(&client, iterations, || execute_median::<f32>(&client, &input, &output, channels, samples, median_width, MEDIAN_DEFAULT_EDGE)), Some(total), format!("width {median_width}"));
    add("TKEO", time_device(&client, iterations, || execute_teager_kaiser::<f32>(&client, &input, &output, channels, samples, TEAGER_KAISER_DEFAULT_EDGE)), Some(total), String::new());
    let taps = client.create_from_slice(f32::as_bytes(&[1.0f32 / FIR_TAPS as f32; FIR_TAPS]));
    add("FIR", time_device(&client, iterations, || execute_fir::<f32>(&client, &input, &output, &taps, channels, samples, FIR_TAPS, FIR_DEFAULT_EDGE)), Some(total), format!("{FIR_TAPS} taps"));

    for (name, mode) in [("band-pass forward", FilterMode::Forward), ("band-pass forward-backward", FilterMode::ForwardBackward)] {
        let filter = DeviceFilter::<f32>::new(&client, &FilterSpec::bandpass(SPIKE_BAND_HZ.0, SPIKE_BAND_HZ.1).with_mode(mode), fs)?;
        let scratch = client.empty((filter.scratch_len(channels, samples) * std::mem::size_of::<f32>()).max(std::mem::size_of::<f32>()));
        let state = client.empty((channels * filter.state_len() * std::mem::size_of::<f32>()).max(std::mem::size_of::<f32>()));
        let t = time_device(&client, iterations, || filter.apply(&client, &input, &output, &scratch, &state, channels, samples));
        add(name, t, Some(total), format!("{}–{} Hz, autotuned blocks", SPIKE_BAND_HZ.0, SPIKE_BAND_HZ.1));
    }

    // int16 stored samples → µV on the device (half the upload of f32)
    let stored: Vec<u8> = host.iter().flat_map(|v| ((v / STEP_UV).round() as i16).to_le_bytes()).collect();
    let words = client.create_from_slice(u32::as_bytes(&stored_words(&stored)));
    let gains = client.create_from_slice(f32::as_bytes(&vec![STEP_UV; channels]));
    let offsets = client.create_from_slice(f32::as_bytes(&vec![0.0f32; channels]));
    let t = time_device(&client, iterations, || {
        execute_unpack_stored::<f32>(&client, &words, SampleFormat::I16, &gains, &offsets, &output, channels, samples).expect("int16 unpack");
    });
    add("unpack int16 → µV", t, Some(total), String::new());

    // Detection, deduplication, extraction and templates, with the streaming defaults
    let config = StreamingDetectionConfig::default();
    let sigmas = execute_channel_noise_std::<f32>(&client, &input, channels, samples, 0..samples);
    let heights: Vec<f32> = sigmas.iter().map(|s| (*s as f32) * config.threshold_factor).collect();
    let heights = client.create_from_slice(f32::as_bytes(&heights));
    let refractory = config.refractory_samples(fs);
    let spacing = SpikeSpacing { refractory_samples: refractory, rule: config.distance_rule };
    let (pre, post) = (config.pre_samples(fs), config.post_samples(fs));
    let emit: Range<usize> = pre..samples.saturating_sub(post);
    let mut events = Vec::new();
    let t = time_device(&client, iterations, || {
        events = execute_detect_spikes_in_vram(&client, &input, &heights, channels, samples, emit.clone(), 0, config.polarity, spacing);
    });
    add("threshold detection", t, Some(total), format!("{} crossings", events.len()));

    let probe = columns_probe(channels);
    let k = config.k_neighbors.min(channels);
    let spikes = deduplicate_spikes_spatial(&events, &probe, config.spatial_radius_um, refractory as u64);
    let knn = client.create_from_slice(u32::as_bytes(&precompute_knn_table(&probe, channels, k)));
    let mut extracted = None;
    let t = time_device(&client, iterations, || {
        extracted = execute_extract_sinc_in_vram::<f32>(&client, &input, &knn, channels, samples, &spikes, k, pre, post, config.apply_sinc_shift);
    });
    add("sinc snippet extraction", t, None, format!("{} spikes × {k} ch × {} samples", spikes.len(), pre + post));
    if let Some(ex) = &extracted {
        let t = time_device(&client, iterations, || {
            execute_reduce_templates_in_vram(&client, &ex.snippets, &ex.primaries, channels, k, pre + post);
        });
        add("template reduction", t, None, format!("{} spikes", ex.primaries.len()));
    }

    // Streaming detection end to end on the synthetic recording
    let detection_sec = (samples as f64 / fs).min(DETECTION_MAX_SEC);
    let pipeline = Pipeline::with_stages(vec![PipelineStage::bandpass(SPIKE_BAND_HZ.0, SPIKE_BAND_HZ.1)]);
    let detector = StreamingDetector::new(config.clone());
    let start = Instant::now();
    detector.run_on(client.clone(), &source, &pipeline, &probe)?;
    let cold = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let result = detector.run_on(client.clone(), &source, &pipeline, &probe)?;
    let wall = start.elapsed().as_secs_f64();
    let detected_values = source.info().samples as f64 * channels as f64;
    let streaming_detection = DetectionRow {
        channels,
        seconds: detection_sec,
        cold_wall_s: cold,
        wall_s: wall,
        msamples_per_s: detected_values / wall / PER_MEGA,
        real_time_factor: source.info().duration_sec() / wall,
        spikes: result.total_dedup_spikes,
    };
    println!(
        "  {:<30} {:>10.3} s   {:>9.0} Msamples/s  {:.1}× real time, {} spikes (first run {:.2} s with compilation)",
        "streaming detection", wall, streaming_detection.msamples_per_s, streaming_detection.real_time_factor, streaming_detection.spikes, cold
    );

    // Matching pursuit with the templates detection found, on the first channels
    let templates: Vec<_> = result.channel_templates.iter().flatten().filter(|t| t.channel_ids.iter().all(|&c| c < MATCHING_CHANNELS)).cloned().collect();
    let (mp_channels, mp_samples) = (channels.min(MATCHING_CHANNELS), samples.min((MATCHING_SEC * fs) as usize));
    if !templates.is_empty() {
        let data: Vec<f32> = (0..mp_channels).flat_map(|c| host[c * samples..c * samples + mp_samples].to_vec()).collect();
        let mut matched = 0;
        let t = time_device(&client, iterations, || {
            matched = match_spikes_matching_pursuit(
                &client,
                &data,
                mp_channels,
                mp_samples,
                &templates,
                DEFAULT_MIN_AMPLITUDE_SCALE,
                DEFAULT_MAX_AMPLITUDE_SCALE,
                DEFAULT_MIN_EXPLAINED_ENERGY_UV2,
                DEFAULT_MAX_PASSES,
            )
            .len();
        });
        add("matching pursuit", t, Some(mp_channels * mp_samples), format!("{mp_channels} ch × {mp_samples} samples, {} templates, {matched} matches", templates.len()));
    }

    let unix_time = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let report = SuiteReport { unix_time, hardware, channels, samples, sample_rate_hz: fs, iterations, kernels: rows, streaming_detection };
    fs::create_dir_all(report_dir)?;
    let file = report_dir.join(format!("{unix_time}-{}.json", client.name()));
    fs::write(&file, serde_json::to_string_pretty(&report)?)?;
    println!("Report written to {}", file.display());
    Ok(())
}
