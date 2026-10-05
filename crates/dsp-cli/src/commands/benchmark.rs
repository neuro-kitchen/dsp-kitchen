//! Device benchmarks. Every timed region is synchronised with the compute client
//! (`dsp_core::compute::bench`), so kernel time is execution time, not enqueueing.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::filter::{DeviceFilter, FilterMode, FilterSpec, execute_fir, execute_median_9p, execute_teager_kaiser};
use dsp_base::math::{execute_clamp, execute_scaling, execute_unpack_stored, stored_words};
use dsp_base::pipeline::{Pipeline, PipelineStage};
use dsp_base::spatial::execute_direct_car;
use dsp_core::compute::bench::{sync, time_device};
use dsp_core::compute::{ComputeTarget, ComputeTask};
use dsp_core::layout::{Position3D, SensorLayout, SensorSite};
use dsp_core::{RecordingSource, SampleFormat};
use dsp_synapse::detection::deduplicate_spikes_spatial;
use dsp_synapse::{
    StreamingSortConfig, StreamingSpikeRunner, WaveformTemplate, execute_detect_spikes_in_vram,
    execute_extract_sinc_in_vram, execute_reduce_templates_in_vram, match_spikes_omp_on, precompute_knn_table,
};
use serde::Serialize;

const FS: f64 = 30_000.0;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

// ------------------------------------------------------------------------------------------------
// Pipeline: upload → kernels → download, each region synchronised
// ------------------------------------------------------------------------------------------------

pub fn run_benchmark_pipeline(
    target: ComputeTarget,
    channels: usize,
    samples: usize,
    iterations: usize,
    save: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    struct Task<'a>(usize, usize, usize, bool, &'a str, &'a Path);
    impl ComputeTask for Task<'_> {
        type Output = anyhow::Result<()>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            benchmark_pipeline_on(client, self.0, self.1, self.2, self.3, self.4, self.5)
        }
    }
    target.run(Task(channels, samples, iterations, save, format, output_path))?
}

/// Scale → 60 Hz notch → TKEO on `[channels, samples]`, the live-pipeline chain.
fn chain<R: Runtime>(client: &ComputeClient<R>, notch: &DeviceFilter, state: &Handle, a: &Handle, b: &Handle, channels: usize, samples: usize) {
    execute_scaling::<R>(client, a, b, channels * samples, 0.195, 0.0);
    notch.apply(client, b, a, state, state, channels, samples);
    execute_teager_kaiser::<R>(client, a, b, channels, samples);
}

fn benchmark_pipeline_on<R: Runtime>(
    client: ComputeClient<R>,
    channels: usize,
    samples: usize,
    iterations: usize,
    save: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    let total = channels * samples;
    println!("Pipeline benchmark: {channels} ch × {samples} samples, {iterations} iterations, {}", R::name(&client));
    println!("Chain: scale → 60 Hz notch (forward) → TKEO; every region is device-synchronised");

    let input = vec![10.0f32; total];
    let notch = notch_filter(&client)?;
    let state = client.empty(channels * notch.state_len() * 4);
    let scratch = client.empty(total * 4);

    // Warm-up: kernel compilation and autotuning, then drain the queue
    let warm = client.create_from_slice(f32::as_bytes(&input));
    chain(&client, &notch, &state, &warm, &scratch, channels, samples);
    sync(&client);

    let (mut up, mut kern, mut down, mut store, mut end_to_end) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..iterations.max(1) {
        let t0 = Instant::now();
        let buf = client.create_from_slice(f32::as_bytes(&input));
        sync(&client);
        let t1 = Instant::now();
        chain(&client, &notch, &state, &buf, &scratch, channels, samples);
        sync(&client);
        let t2 = Instant::now();
        let out = client.read_one_unchecked(scratch.clone());
        let t3 = Instant::now();
        if save {
            persist(&out, channels, format, output_path)?;
        }
        let t4 = Instant::now();
        up.push(t1 - t0);
        kern.push(t2 - t1);
        down.push(t3 - t2);
        store.push(t4 - t3);
        end_to_end.push(t4 - t0);
    }
    let kernel = median(kern);
    let msamples = total as f64 / kernel.as_secs_f64() / 1e6;
    println!("  upload       {:>9.3} ms", ms(median(up)));
    println!("  kernels      {:>9.3} ms   {msamples:.0} Msamples/s, {:.1}× real time", ms(kernel), samples as f64 / FS / kernel.as_secs_f64());
    println!("  download     {:>9.3} ms", ms(median(down)));
    if save {
        println!("  storage      {:>9.3} ms ({format})", ms(median(store)));
    }
    println!("  end to end   {:>9.3} ms (medians)", ms(median(end_to_end)));
    Ok(())
}

fn persist(bytes: &[u8], channels: usize, format: &str, output_path: &Path) -> anyhow::Result<()> {
    if format == "zarr" {
        let path = if output_path.extension().is_some_and(|e| e == "bin") { output_path.with_extension("zarr") } else { output_path.to_path_buf() };
        let rec = dsp_core::MemoryRecording::new("pipeline_output", f32::from_bytes(bytes).to_vec(), channels, FS)?;
        dsp_io::write_zarr(&rec, &path, dsp_io::zarr::DEFAULT_CHUNK_SAMPLES, |_, _| {})?;
    } else {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, bytes)?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Channel sweep
// ------------------------------------------------------------------------------------------------

pub fn run_benchmark_sweep(target: ComputeTarget, samples_per_channel: usize, iterations: usize) -> anyhow::Result<()> {
    struct Task(usize, usize);
    impl ComputeTask for Task {
        type Output = anyhow::Result<()>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            benchmark_sweep_on(client, self.0, self.1)
        }
    }
    target.run(Task(samples_per_channel, iterations))?
}

fn benchmark_sweep_on<R: Runtime>(client: ComputeClient<R>, samples: usize, iterations: usize) -> anyhow::Result<()> {
    println!("Channel sweep: scale → notch → TKEO, {samples} samples per channel, {}", R::name(&client));
    println!("{:>8} | {:>12} | {:>10} | {:>12} | {:>9}", "channels", "elements", "kernel ms", "Msamples/s", "realtime");
    let notch = notch_filter(&client)?;
    for channels in [1usize, 4, 16, 32, 64, 128, 384, 1024] {
        let total = channels * samples;
        let a = client.create_from_slice(f32::as_bytes(&vec![1.0f32; total]));
        let b = client.empty(total * 4);
        let state = client.empty(channels * notch.state_len() * 4);
        let t = time_device(&client, iterations, || chain(&client, &notch, &state, &a, &b, channels, samples));
        println!(
            "{channels:>8} | {total:>12} | {:>10.3} | {:>12.0} | {:>8.1}×",
            ms(t),
            total as f64 / t.as_secs_f64() / 1e6,
            samples as f64 / FS / t.as_secs_f64()
        );
    }
    Ok(())
}

/// Causal 60 Hz notch (Q 30) at 30 kHz, as used in live pipelines.
fn notch_filter<R: Runtime>(client: &ComputeClient<R>) -> anyhow::Result<DeviceFilter> {
    let spec = FilterSpec::notch(60.0, 30.0).with_mode(FilterMode::Forward);
    Ok(DeviceFilter::new(client, &spec, FS)?)
}

// ------------------------------------------------------------------------------------------------
// Kernel suite
// ------------------------------------------------------------------------------------------------

#[derive(Serialize)]
struct SuiteRow {
    kernel: String,
    ms: f64,
    /// Input samples processed per second (channels × samples / time), when meaningful.
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
struct SortRow {
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
    iterations: usize,
    kernels: Vec<SuiteRow>,
    sorter: SortRow,
}

/// Times each kernel family and the streaming sorter on `target`, prints a table and writes the
/// report as JSON into `out_dir` (`<unix time>-<runtime>.json`).
pub fn run_benchmark_suite(target: ComputeTarget, channels: usize, samples: usize, iterations: usize, out_dir: &Path) -> anyhow::Result<PathBuf> {
    struct Task(usize, usize, usize);
    impl ComputeTask for Task {
        type Output = anyhow::Result<SuiteReport>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            suite_on(client, self.0, self.1, self.2)
        }
    }
    let report = target.run(Task(channels, samples, iterations))??;
    fs::create_dir_all(out_dir)?;
    let file = out_dir.join(format!("{}-{}.json", report.unix_time, target.name()));
    fs::write(&file, serde_json::to_string_pretty(&report)?)?;
    println!("Report written to {}", file.display());
    Ok(file)
}

/// Two-column layout with 20 µm row pitch for any channel count.
fn linear_probe(channels: usize) -> SensorLayout {
    let sites = (0..channels).map(|c| SensorSite::new(c, Position3D::new((c % 2) as f32 * 32.0, (c / 2) as f32 * 20.0, 0.0), 0)).collect();
    SensorLayout::new("benchmark", sites)
}

fn suite_on<R: Runtime>(client: ComputeClient<R>, channels: usize, samples: usize, iterations: usize) -> anyhow::Result<SuiteReport> {
    let hw = &client.properties().hardware;
    let hardware = Hardware {
        runtime: R::name(&client).to_string(),
        plane_size_max: hw.plane_size_max,
        max_units_per_cube: hw.max_units_per_cube,
        num_cpu_cores: hw.num_cpu_cores,
        num_streaming_multiprocessors: hw.num_streaming_multiprocessors,
    };
    println!("Kernel suite: {channels} ch × {samples} samples ({:.1} s at 30 kHz), median of {iterations}, {}", samples as f64 / FS, hardware.runtime);
    let total = channels * samples;
    let mut rows: Vec<SuiteRow> = Vec::new();
    let mut add = |kernel: &str, t: Duration, elements: Option<usize>, note: String| {
        let row = SuiteRow { kernel: kernel.into(), ms: ms(t), msamples_per_s: elements.map(|n| n as f64 / t.as_secs_f64() / 1e6), note };
        match row.msamples_per_s {
            Some(r) => println!("  {:<30} {:>10.3} ms  {:>9.0} Msamples/s  {}", row.kernel, row.ms, r, row.note),
            None => println!("  {:<30} {:>10.3} ms  {:>20}  {}", row.kernel, row.ms, "", row.note),
        }
        rows.push(row);
    };

    // Deterministic noise with sparse negative spikes
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };
    let host: Vec<f32> = (0..total).map(|i| rnd() * 20.0 + if i % 997 == 0 { -150.0 } else { 0.0 }).collect();
    let input = client.create_from_slice(f32::as_bytes(&host));
    let output = client.empty(total * 4);

    add("scale", time_device(&client, iterations, || execute_scaling::<R>(&client, &input, &output, total, 0.5, 1.0)), Some(total), String::new());
    add("clamp", time_device(&client, iterations, || execute_clamp::<R>(&client, &input, &output, total, -50.0, 50.0)), Some(total), String::new());
    add("CAR (direct)", time_device(&client, iterations, || execute_direct_car::<R>(&client, &input, &output, channels, samples)), Some(total), String::new());
    add("median 9-point", time_device(&client, iterations, || execute_median_9p::<R>(&client, &input, &output, channels, samples)), Some(total), String::new());
    add("TKEO", time_device(&client, iterations, || execute_teager_kaiser::<R>(&client, &input, &output, channels, samples)), Some(total), String::new());
    let taps = client.create_from_slice(f32::as_bytes(&[1.0f32 / 32.0; 32]));
    add("FIR 32 taps", time_device(&client, iterations, || execute_fir::<R>(&client, &input, &output, &taps, channels, samples, 32)), Some(total), String::new());

    for (name, mode) in [("bandpass o5 forward", FilterMode::Forward), ("bandpass o5 filtfilt", FilterMode::ForwardBackward)] {
        let filter = DeviceFilter::new(&client, &FilterSpec::bandpass(300.0, 6000.0).with_mode(mode), FS)?;
        let scratch = client.empty((filter.scratch_len(channels, samples) * 4).max(4));
        let state = client.empty(channels * filter.state_len() * 4);
        let t = time_device(&client, iterations, || filter.apply(&client, &input, &output, &scratch, &state, channels, samples));
        add(name, t, Some(total), "autotuned time blocks".into());
    }

    // int16 stored samples → µV (half the upload of f32)
    let stored: Vec<u8> = host.iter().flat_map(|v| ((v * 4.0) as i16).to_le_bytes()).collect();
    let words = client.create_from_slice(u32::as_bytes(&stored_words(&stored)));
    let gains = client.create_from_slice(f32::as_bytes(&vec![0.25f32; channels]));
    let offsets = client.create_from_slice(f32::as_bytes(&vec![0.0f32; channels]));
    let t = time_device(&client, iterations, || {
        execute_unpack_stored::<R>(&client, &words, SampleFormat::I16, &gains, &offsets, &output, channels, samples).expect("int16 unpack")
    });
    add("unpack int16 → µV", t, Some(total), String::new());

    let sigmas = client.create_from_slice(f32::as_bytes(&vec![12.0f32; channels]));
    let mut events = Vec::new();
    let t = time_device(&client, iterations, || {
        events = execute_detect_spikes_in_vram::<R>(&client, &input, &sigmas, channels, samples, 10, samples - 10, 0, 5.0, 30, None);
    });
    add("threshold detection", t, Some(total), format!("{} crossings", events.len()));

    let probe = linear_probe(channels);
    let (k, pre, post) = (4usize.min(channels), 30usize, 60usize);
    let spikes = deduplicate_spikes_spatial(&events, &probe, 40.0, 30);
    let knn = client.create_from_slice(u32::as_bytes(&precompute_knn_table(&probe, channels, k)));
    let mut extracted = None;
    let t = time_device(&client, iterations, || {
        extracted = execute_extract_sinc_in_vram::<R>(&client, &input, &knn, channels, samples, &spikes, k, pre, post, true);
    });
    add("sinc snippet extraction", t, None, format!("{} spikes × {k} ch × {} samples", spikes.len(), pre + post));
    if let Some(ex) = &extracted {
        let t = time_device(&client, iterations, || {
            execute_reduce_templates_in_vram::<R>(&client, &ex.snippets, &ex.primaries, channels, k, pre + post);
        });
        add("template reduction", t, None, format!("{} spikes", ex.num_spikes()));
    }

    // OMP on one second of the first channels against a small dictionary
    let (omp_ch, omp_len, units, t_len) = (channels.min(32), samples.min(30_000), 16usize, 60usize);
    let omp_data: Vec<f32> = (0..omp_ch).flat_map(|c| host[c * samples..c * samples + omp_len].to_vec()).collect();
    let templates: Vec<WaveformTemplate> = (0..units)
        .map(|u| {
            let ids: Vec<usize> = (0..4).map(|r| (u * 2 + r) % omp_ch).collect();
            let mean: Vec<f32> = (0..4 * t_len).map(|i| -100.0 * (-((i % t_len) as f32 - 20.0).powi(2) / 20.0).exp() * (1.0 - 0.2 * (i / t_len) as f32)).collect();
            WaveformTemplate::new(ids, t_len, mean, vec![1.0; 4 * t_len])
        })
        .collect();
    let mut matched = 0;
    let t = time_device(&client, iterations, || {
        matched = match_spikes_omp_on::<R>(&client, &omp_data, omp_ch, omp_len, &templates, 0.65, 1.45, 500.0, 4).len();
    });
    add("OMP matching", t, Some(omp_ch * omp_len), format!("{omp_ch} ch × {omp_len} samples, {units} templates, {matched} matches"));

    // Streaming sorter end to end on a synthetic recording
    let sort_secs = (samples as f64 / FS).min(10.0);
    let source = dsp_io::SyntheticRecording::new(dsp_io::SyntheticParams {
        channels,
        sample_rate_hz: FS,
        duration_sec: sort_secs,
        units: (channels / 4).clamp(1, 64),
        ..Default::default()
    })?;
    // Reading the source alone, to separate its cost from the sorter's
    let all: Vec<usize> = (0..channels).collect();
    let n_src = source.info().samples;
    let mut buf = vec![0.0f32; channels * FS as usize];
    let start = Instant::now();
    let mut s0 = 0u64;
    while s0 < n_src {
        let s1 = (s0 + FS as u64).min(n_src);
        source.read(&all, s0..s1, &mut buf[..channels * (s1 - s0) as usize])?;
        s0 = s1;
    }
    let read = start.elapsed();
    add("source read (synthetic)", read, Some(channels * n_src as usize), "host-side sample generation, no processing".into());

    let pipeline = Pipeline::with_stages(vec![PipelineStage::bandpass(300.0, 6000.0)]);
    let runner = StreamingSpikeRunner::new(StreamingSortConfig::default());
    let start = Instant::now();
    runner.run_on::<R>(client.clone(), &source, &pipeline, &probe)?;
    let cold = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let result = runner.run_on::<R>(client.clone(), &source, &pipeline, &probe)?;
    let wall = start.elapsed().as_secs_f64();
    let sorted_samples = source.info().samples as f64 * channels as f64;
    let sorter = SortRow {
        channels,
        seconds: sort_secs,
        cold_wall_s: cold,
        wall_s: wall,
        msamples_per_s: sorted_samples / wall / 1e6,
        real_time_factor: sort_secs / wall,
        spikes: result.total_dedup_spikes,
    };
    println!(
        "  {:<30} {:>10.3} s   {:>9.0} Msamples/s  {:.1}× real time, {} spikes (first run {:.2} s with compilation)",
        "streaming sorter",
        sorter.wall_s,
        sorter.msamples_per_s,
        sorter.real_time_factor,
        sorter.spikes,
        sorter.cold_wall_s
    );

    let unix_time = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    Ok(SuiteReport { unix_time, hardware, channels, samples, iterations, kernels: rows, sorter })
}
