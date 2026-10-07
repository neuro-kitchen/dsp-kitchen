//! Timings of the dense linear-algebra paths (covariance, second moment, dense spatial operator,
//! projection) at sorter sizes, on the discrete GPU, the integrated GPU (wgpu) and the CPU
//! runtime when compiled in. Ignored by default (minutes, mostly autotuning):
//!
//! ```text
//! cargo test -p dsp-base --release --features cpu --test bench_linalg -- --ignored --nocapture
//! ```
//!
//! The first call of each case is the warm-up (kernel compilation and autotuning); the reported
//! time is the median over [`ITERATIONS`] device-synchronised runs.

use cubecl::device::{CpuDevice, WgpuDevice, WgpuDeviceKind};
use cubecl::prelude::*;
use cubecl::Device;
use dsp_base::core::buffer;
use dsp_base::linalg::{covariance, DeviceProjection, SecondMomentAccumulator};
use dsp_base::spatial::execute_spatial_matrix_multiply;
use dsp_core::compute::bench::time_device;

/// Timed runs per case (after the warm-up).
const ITERATIONS: usize = 5;
/// Samples of one detection window (2 s at 30 kHz).
const WINDOW: usize = 60_000;
/// Halo samples on each side of a window's interior.
const HALO: usize = 1_000;
/// Channel counts: Neuropixels, HD-EMG grid, a tetrode-sized probe.
const CHANNELS: [usize; 3] = [384, 64, 4];
/// Clip length (samples) and principal components of Kilosort4's `wPCA`.
const CLIP: usize = 61;
const COMPONENTS: usize = 6;
/// Clips projected at once.
const CLIPS: usize = 100_000;

fn data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7919) % 1013) as f32 * 0.01 - 5.0).collect()
}

fn report(device: &str, case: &str, time: std::time::Duration) {
    println!("{device:>10} | {case:<34} | {:>10.3} ms", time.as_secs_f64() * 1e3);
}

fn bench(device: &str, client: &Client) {
    for c in CHANNELS {
        let x = buffer::upload(client, &data(c * WINDOW));
        let (cov, mean) = (buffer::empty::<f32>(client, c * c), buffer::empty::<f32>(client, c));
        let t = time_device(client, ITERATIONS, || covariance::<f32>(client, &x, &cov, &mean, c, WINDOW));
        report(device, &format!("covariance {c}x{WINDOW}"), t);

        let row_len = WINDOW + 2 * HALO;
        let padded = buffer::upload(client, &data(c * row_len));
        let mut acc = SecondMomentAccumulator::<f32>::new(client, c);
        let t = time_device(client, ITERATIONS, || acc.add(&padded, row_len, HALO..HALO + WINDOW));
        report(device, &format!("second moment {c}x{WINDOW} (halo)"), t);

        let w = buffer::upload(client, &data(c * c));
        let out = buffer::empty::<f32>(client, c * WINDOW);
        let t = time_device(client, ITERATIONS, || execute_spatial_matrix_multiply::<f32>(client, &x, &w, &out, c, WINDOW));
        report(device, &format!("dense spatial {c}x{c} . {c}x{WINDOW}"), t);
    }

    #[allow(unused_mut)]
    let mut proj = DeviceProjection::upload::<f32>(client, &data(CLIP * COMPONENTS), &data(CLIP), CLIP, COMPONENTS);
    let clips = buffer::upload(client, &data(CLIP * CLIPS));
    let out = buffer::empty::<f32>(client, COMPONENTS * CLIPS);
    let t = time_device(client, ITERATIONS, || proj.project::<f32>(client, &clips, &out, CLIPS));
    report(device, &format!("projection {CLIP}->{COMPONENTS} x{CLIPS}"), t);
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_dense_linalg() {
    let mut devices = vec![
        ("dGPU", Device::Wgpu(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0)))),
        ("iGPU", Device::Wgpu(WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(0)))),
    ];
    if cfg!(feature = "cpu") {
        devices.push(("cpu", Device::Cpu(CpuDevice)));
    }
    for (name, device) in devices {
        bench(name, &device.client());
    }
}
