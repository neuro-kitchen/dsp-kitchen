//! One timing per kernel family at a recording-sized block (384 channels × 60 000 samples, a 2 s
//! Neuropixels window), as effective bandwidth: the bytes a kernel must move at least (read the
//! block once, write the result once) over its time, and that as a share of the streaming kernel
//! (`scale`: one read, one write, nothing else), the practical ceiling on this device. A kernel
//! near 100 % is memory-bound and done; one far below is where work pays. Ignored by default:
//!
//! ```text
//! cargo test -p dspk-ops --release --test bench_kernels -- --ignored --nocapture
//! DSPK_TEST_RUNTIMES=cuda cargo test ...   # one runtime
//! ```
//!
//! Timings are medians of fresh launches after a warm-up (autotune happens in the warm-up); run it
//! in a fresh process, not right after another benchmark.

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_core::compute::bench::time_device;
use dspk_core::ComputeTarget;
use dspk_ops::core::{buffer, EdgeMode};
use dspk_ops::filter::{execute_fir, execute_fir_centered, execute_filter, execute_median, execute_teager_kaiser, FilterBand, FilterMode, FilterSpec};
use dspk_ops::math::{execute_channel_mean_std, execute_channel_noise_std, execute_clamp, execute_scaling};
use dspk_ops::peaks::find_peak_candidates_on_device;
use dspk_ops::resampler::{firwin, resample_poly, resample_poly_len, FirWindow, ResampleFilter};
use dspk_ops::spatial::{execute_common_median, execute_direct_car};

/// Timed runs per kernel (after one warm-up).
const ITERATIONS: usize = 7;
const CHANNELS: usize = 384;
/// 2 s at 30 kHz.
const SAMPLES: usize = 60_000;
const SAMPLE_RATE: f64 = 30_000.0;

fn data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i.wrapping_mul(7919)) % 1013) as f32 * 0.01 - 5.0).collect()
}

struct Row {
    name: &'static str,
    ms: f64,
    /// Least bytes the kernel must move.
    bytes: f64,
}

fn bench(target: ComputeTarget, client: &Client) -> Vec<Row> {
    let n = CHANNELS * SAMPLES;
    let block = (n * 4) as f64;
    let x = buffer::upload(client, &data(n));
    let out = buffer::empty::<f32>(client, n);
    let mut rows = Vec::new();
    let mut time = |name: &'static str, bytes: f64, f: &mut dyn FnMut()| {
        let t = time_device(client, ITERATIONS, f);
        rows.push(Row { name, ms: t.as_secs_f64() * 1e3, bytes });
    };
    let rw = 2.0 * block;

    time("scale (ceiling)", rw, &mut || execute_scaling::<f32>(client, &x, &out, n, 2.0, 1.0));
    time("clamp", rw, &mut || execute_clamp::<f32>(client, &x, &out, n, -1.0, 1.0));
    time("car (mean)", rw, &mut || execute_direct_car::<f32>(client, &x, &out, CHANNELS, SAMPLES));
    time("cmr (median)", rw, &mut || execute_common_median::<f32>(client, &x, &out, CHANNELS, SAMPLES));
    for (name, spec) in [
        ("iir bandpass 5, forward", FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)).with_mode(FilterMode::Forward)),
        ("iir bandpass 5, zero-phase", FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)).with_mode(FilterMode::ForwardBackward)),
    ] {
        time(name, rw, &mut || execute_filter::<f32>(client, &spec, SAMPLE_RATE, &x, &out, CHANNELS, SAMPLES).unwrap());
    }
    let taps_host: Vec<f32> = firwin(63, 0.2, FirWindow::Hamming).into_iter().map(|v| v as f32).collect();
    let taps = buffer::upload(client, &taps_host);
    time("fir 63 taps, causal", rw, &mut || execute_fir::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, 63, EdgeMode::Reflect));
    time("fir 63 taps, centred", rw, &mut || execute_fir_centered::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, 31, EdgeMode::Reflect));
    time("median width 5", rw, &mut || execute_median::<f32>(client, &x, &out, CHANNELS, SAMPLES, 5, EdgeMode::Reflect));
    time("median width 31", rw, &mut || execute_median::<f32>(client, &x, &out, CHANNELS, SAMPLES, 31, EdgeMode::Reflect));
    time("teager-kaiser", rw, &mut || execute_teager_kaiser::<f32>(client, &x, &out, CHANNELS, SAMPLES, EdgeMode::Reflect));
    let resampled = buffer::empty::<f32>(client, CHANNELS * resample_poly_len(SAMPLES, 1, 3));
    let window = ResampleFilter::Window(FirWindow::Kaiser { beta: 5.0 });
    time("resample_poly 1/3", block * (1.0 + 1.0 / 3.0), &mut || {
        resample_poly::<f32>(client, &x, &resampled, CHANNELS, SAMPLES, 1, 3, &window, dspk_ops::resampler::RESAMPLE_POLY_DEFAULT_EDGE);
    });
    let (mean, std) = (buffer::empty::<f32>(client, CHANNELS), buffer::empty::<f32>(client, CHANNELS));
    time("channel mean/std", block, &mut || execute_channel_mean_std::<f32>(client, &x, &mean, &std, CHANNELS, SAMPLES));
    time("channel noise (median |x|)", block, &mut || {
        execute_channel_noise_std::<f32>(client, &x, CHANNELS, SAMPLES, 0..SAMPLES);
    });
    let heights: Handle = buffer::upload(client, &vec![3.0f32; CHANNELS]);
    time("peak candidates", block, &mut || {
        let _ = find_peak_candidates_on_device::<f32>(client, &x, &heights, CHANNELS, SAMPLES, 0..SAMPLES, Default::default());
    });
    let _ = target;
    rows
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn kernel_bandwidth() {
    for target in ComputeTarget::for_tests() {
        let client = match target.client() {
            Ok(c) => c,
            Err(e) => {
                println!("{}: skipped ({e})", target.name());
                continue;
            }
        };
        let rows = bench(target, &client);
        let ceiling = rows[0].bytes / rows[0].ms / 1e6;
        println!("\n{} — {CHANNELS} × {SAMPLES} float32 ({:.1} MB), ceiling {ceiling:.0} GB/s", target.name(), (CHANNELS * SAMPLES * 4) as f64 / 1e6);
        println!("{:<30} {:>10} {:>10} {:>9}", "kernel", "ms", "GB/s", "of ceil.");
        for r in &rows {
            let gbs = r.bytes / r.ms / 1e6;
            println!("{:<30} {:>10.3} {:>10.1} {:>8.0}%", r.name, r.ms, gbs, 100.0 * gbs / ceiling);
        }
    }
}
