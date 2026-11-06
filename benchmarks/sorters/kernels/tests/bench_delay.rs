//! EMUsort channel-delay estimation: one fit window added to the estimator (envelopes + lagged
//! cross-correlation of every channel pair), at HD-EMG size on the discrete GPU. Ignored by default:
//!
//! ```text
//! cargo test -p dspk-benchmarks-sorters --profile validate --test bench_delay -- --ignored --nocapture
//! ```

use cubecl::device::{WgpuDevice, WgpuDeviceKind};
use cubecl::Device;
use dspk_ops::core::buffer;
use dspk_benchmarks::Report;
use dspk_core::compute::bench::time_device;
use dspk_sorter_kilosort::delays::ChannelDelayEstimator;

/// HD-EMG grid channels, window samples (interior plus halos), halo, and ±2 ms of lag at 30 kHz.
const CHANNELS: usize = 32;
const WINDOW: usize = 60_000;
const HALO: usize = 1_000;
const MAX_LAG: usize = 60;
/// Timed runs (after one warm-up).
const ITERATIONS: usize = 5;

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_delay_estimation() {
    let client = dspk_core::compute::open_device(Device::Wgpu(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0))));
    let row_len = WINDOW + 2 * HALO;
    let x: Vec<f32> = (0..CHANNELS * row_len).map(|i| ((i * 7919) % 1013) as f32 * 0.01 - 5.0).collect();
    let handle = buffer::upload(&client, &x);
    let mut estimator = ChannelDelayEstimator::new(&client, CHANNELS, MAX_LAG);
    let t = time_device(&client, ITERATIONS, || estimator.add(&handle, row_len, HALO..HALO + WINDOW));
    println!("delay add | {CHANNELS} ch x {WINDOW} | lags ±{MAX_LAG} | {:.2} ms / window", t.as_secs_f64() * 1e3);
    let mut record = Report::new("sorters/emusort-delays", "wgpu dGPU", &format!("one fit window added to the channel-delay estimator (lags ±{MAX_LAG}); median of {ITERATIONS} after a warm-up"));
    record.case("delay estimator, one window", &[CHANNELS, WINDOW], t.as_secs_f64() * 1e3, &[]);
    record.write_to(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results")).expect("writing the record");
}
