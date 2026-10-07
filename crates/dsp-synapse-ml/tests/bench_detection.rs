//! Universal-template detection over a sequence of windows at Neuropixels size (384 channels in
//! two columns, 60 000-sample windows of whitened noise with injected spikes), on the discrete GPU.
//! Each window is uploaded and detected, as in a run. Ignored by default:
//!
//! ```text
//! cargo test -p dsp-synapse-ml --release --test bench_detection -- --ignored --nocapture
//! ```

use std::time::Instant;

use cubecl::device::{WgpuDevice, WgpuDeviceKind};
use cubecl::Device;
use dsp_base::core::buffer;
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse_ml::sorters::kilosort4::{CentreOptions, TemplateCentres, UniversalDetector, UniversalTemplates};

/// Channels (two columns of a Neuropixels-like probe), their spacing, and the window length.
const CHANNELS: usize = 384;
const PITCH_UM: f32 = 20.0;
const WINDOW: usize = 60_000;
/// Windows detected per run (after one warm-up window).
const WINDOWS: usize = 8;
/// Kilosort4 defaults: samples per template, templates, PCs, threshold, peak sample.
const NT: usize = 61;
const N_TEMPLATES: usize = 6;
const N_PCS: usize = 6;
const TH_UNIVERSAL: f32 = 9.0;
const NT0MIN: usize = 20;
/// One injected spike every this many samples, on a channel that walks across the probe.
const SPIKE_EVERY: usize = 97;
/// Injected spike amplitude (whitened σ).
const SPIKE_AMPLITUDE: f32 = 15.0;

fn probe() -> SensorLayout {
    let ids: Vec<usize> = (0..CHANNELS).collect();
    let positions: Vec<[f32; 2]> = (0..CHANNELS).map(|c| [(c % 2) as f32 * 32.0, (c / 2) as f32 * PITCH_UM]).collect();
    SensorLayout::from_channel_arrays("bench-384", &ids, &positions, &vec![0; CHANNELS]).expect("probe")
}

/// Smooth troughs of different widths, rows L2-normalized (`[n, NT]`).
fn shapes(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * NT);
    for k in 0..n {
        let width = 2.0 + k as f32;
        let row: Vec<f32> = (0..NT).map(|t| -(-(((t as f32 - NT0MIN as f32) / width).powi(2))).exp()).collect();
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        out.extend(row.iter().map(|v| v / norm));
    }
    out
}

/// Whitened noise with spikes (`[CHANNELS, WINDOW]`), different per `seed`.
fn window(seed: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9u32.wrapping_add(seed as u32 * 7919);
    let mut x: Vec<f32> = (0..CHANNELS * WINDOW)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 3.4 - 1.7
        })
        .collect();
    let shape = shapes(1);
    for (i, t0) in (NT..WINDOW - NT).step_by(SPIKE_EVERY).enumerate() {
        let c = (i * 37 + seed) % CHANNELS;
        for (t, v) in shape.iter().enumerate() {
            x[c * WINDOW + t0 + t] += SPIKE_AMPLITUDE * v * NT as f32 / 8.0;
        }
    }
    x
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_universal_detection() {
    let client = Device::Wgpu(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0))).client();
    let centres = TemplateCentres::new(&probe(), &CentreOptions::default()).expect("centres");
    let templates = UniversalTemplates { nt: NT, n_pcs: N_PCS, n_templates: N_TEMPLATES, wpca: shapes(N_PCS), wtemp: shapes(N_TEMPLATES) };
    let mut detector = UniversalDetector::new(&client, CHANNELS, WINDOW, &centres, &templates, TH_UNIVERSAL, NT0MIN).expect("detector");
    let data: Vec<Vec<f32>> = (0..=WINDOWS).map(window).collect();

    // Warm-up: compilation and autotuning
    detector.detect(&buffer::upload(&client, &data[0]), WINDOW).expect("detect");
    // One persistent device buffer, written in place every window (as `PipelineWorkspace` does)
    let handle = buffer::empty::<f32>(&client, CHANNELS * WINDOW);
    let (mut host_ms, mut upload_ms, mut detect_ms, mut spikes) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for x in &data[1..] {
        // In a run the read-ahead thread fills an owned buffer; only the hand-over is timed here
        let owned = x.clone();
        let start = Instant::now();
        buffer::write_owned(&client, &handle, owned);
        host_ms += start.elapsed().as_secs_f64() * 1e3;
        dsp_core::compute::bench::sync(&client);
        let uploaded = Instant::now();
        spikes += detector.detect(&handle, WINDOW).expect("detect").len();
        upload_ms += (uploaded - start).as_secs_f64() * 1e3;
        detect_ms += uploaded.elapsed().as_secs_f64() * 1e3;
    }
    let (host_ms, upload_ms, detect_ms) = (host_ms / WINDOWS as f64, upload_ms / WINDOWS as f64, detect_ms / WINDOWS as f64);
    println!(
        "sequential | {CHANNELS} ch x {WINDOW} | {} centres | upload {upload_ms:.2} ms (host call {host_ms:.2}) + detect {detect_ms:.2} ms / window | {spikes} spikes",
        centres.n_centres()
    );
}
