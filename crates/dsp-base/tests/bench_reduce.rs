//! Timings of the per-channel reductions (median-|x| noise estimate) at sorter sizes, on the
//! discrete and the integrated GPU (wgpu). Ignored by default:
//!
//! ```text
//! cargo test -p dsp-base --release --test bench_reduce -- --ignored --nocapture
//! ```

use cubecl::device::{WgpuDevice, WgpuDeviceKind};
use cubecl::prelude::*;
use cubecl::Device;
use dsp_base::core::buffer;
use dsp_base::math::execute_channel_noise_std;
use dsp_core::compute::bench::time_device;

/// Timed runs per case (after the warm-up).
const ITERATIONS: usize = 5;
/// Samples of one detection window (2 s at 30 kHz).
const WINDOW: usize = 60_000;
/// Channel counts: Neuropixels, HD-EMG grid, a tetrode-sized probe.
const CHANNELS: [usize; 3] = [384, 64, 4];

fn data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7919) % 1013) as f32 * 0.01 - 5.0).collect()
}

fn bench(device: &str, client: &Client) {
    for c in CHANNELS {
        let x = buffer::upload(client, &data(c * WINDOW));
        let t = time_device(client, ITERATIONS, || {
            execute_channel_noise_std::<f32>(client, &x, c, WINDOW, 0..WINDOW);
        });
        println!("{device:>10} | median |x| {c}x{WINDOW:<22} | {:>10.3} ms", t.as_secs_f64() * 1e3);
    }
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_reductions() {
    for (name, kind) in [("dGPU", WgpuDeviceKind::DiscreteGpu(0)), ("iGPU", WgpuDeviceKind::IntegratedGpu(0))] {
        bench(name, &Device::Wgpu(WgpuDevice::new(kind)).client());
    }
}
