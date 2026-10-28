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
use dspk_ops::peaks::{find_peak_candidates_on_device, Polarity};
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

/// Recording-like samples for the peak search: noise of σ ≈ 1 (sum of four uniforms) plus a
/// negative spike of 8σ every `SPIKE_EVERY` samples on each channel, at a channel-dependent phase.
/// [`data`] would make about a third of all samples local extrema above any modest height.
fn spiky(channels: usize, samples: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_u32;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32 - 0.5
    };
    let mut out = Vec::with_capacity(channels * samples);
    for c in 0..channels {
        for t in 0..samples {
            let noise = (uniform() + uniform() + uniform() + uniform()) * 1.732;
            let spike = if (t + 37 * c) % SPIKE_EVERY == 0 { -8.0 } else { 0.0 };
            out.push(noise + spike);
        }
    }
    out
}

/// Samples between two spikes of [`spiky`] (30 per second at 30 kHz).
const SPIKE_EVERY: usize = 1_000;
/// Peak-search threshold for [`spiky`]: 5σ.
const SPIKE_HEIGHT: f32 = 5.0;

struct Row {
    name: &'static str,
    ms: f64,
    /// Least bytes the kernel must move.
    bytes: f64,
    /// Arithmetic operations (adds, multiplies, comparisons) the algorithm does, or 0 where
    /// bandwidth alone describes the kernel.
    ops: f64,
}

fn bench(target: ComputeTarget, client: &Client) -> Vec<Row> {
    let n = CHANNELS * SAMPLES;
    let block = (n * 4) as f64;
    let x = buffer::upload(client, &data(n));
    let out = buffer::empty::<f32>(client, n);
    let mut rows = Vec::new();
    let mut time = |name: &'static str, bytes: f64, ops_per_sample: f64, f: &mut dyn FnMut()| {
        let t = time_device(client, ITERATIONS, f);
        rows.push(Row { name, ms: t.as_secs_f64() * 1e3, bytes, ops: ops_per_sample * n as f64 });
    };
    let rw = 2.0 * block;
    // Fifth-order Butterworth band-pass: five second-order sections, 5 multiplies + 4 adds each
    let iir_ops = 5.0 * 9.0;
    let fir_ops = 2.0 * 63.0;
    // Rank counting compares every pair of the window, twice (less, less-or-equal)
    let rank_ops = |width: f64| 2.0 * width * width;

    time("scale (ceiling)", rw, 0.0, &mut || execute_scaling::<f32>(client, &x, &out, n, 2.0, 1.0));
    time("clamp", rw, 0.0, &mut || execute_clamp::<f32>(client, &x, &out, n, -1.0, 1.0));
    time("car (mean)", rw, 0.0, &mut || execute_direct_car::<f32>(client, &x, &out, CHANNELS, SAMPLES));
    // Per sample, every channel is ranked against every other: 2 · C² per sample = 2 · C per value
    time("cmr (median)", rw, rank_ops(CHANNELS as f64) / CHANNELS as f64, &mut || {
        execute_common_median::<f32>(client, &x, &out, CHANNELS, SAMPLES)
    });
    for (name, passes, spec) in [
        ("iir bandpass 5, forward", 1.0, FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)).with_mode(FilterMode::Forward)),
        ("iir bandpass 5, zero-phase", 2.0, FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)).with_mode(FilterMode::ForwardBackward)),
    ] {
        time(name, rw, passes * iir_ops, &mut || execute_filter::<f32>(client, &spec, SAMPLE_RATE, &x, &out, CHANNELS, SAMPLES).unwrap());
    }
    let taps_host: Vec<f32> = firwin(63, 0.2, FirWindow::Hamming).into_iter().map(|v| v as f32).collect();
    let taps = buffer::upload(client, &taps_host);
    time("fir 63 taps, causal", rw, fir_ops, &mut || execute_fir::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, 63, EdgeMode::Reflect));
    time("fir 63 taps, centred", rw, fir_ops, &mut || execute_fir_centered::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, 31, EdgeMode::Reflect));
    time("median width 5", rw, rank_ops(5.0), &mut || execute_median::<f32>(client, &x, &out, CHANNELS, SAMPLES, 5, EdgeMode::Reflect));
    time("median width 31", rw, rank_ops(31.0), &mut || execute_median::<f32>(client, &x, &out, CHANNELS, SAMPLES, 31, EdgeMode::Reflect));
    time("teager-kaiser", rw, 0.0, &mut || execute_teager_kaiser::<f32>(client, &x, &out, CHANNELS, SAMPLES, EdgeMode::Reflect));
    let resampled = buffer::empty::<f32>(client, CHANNELS * resample_poly_len(SAMPLES, 1, 3));
    let window = ResampleFilter::Window(FirWindow::Kaiser { beta: 5.0 });
    // Kaiser window of 2 · 10 · max(up, down) + 1 taps; each output (a third of the inputs) sums them
    let resample_ops = 2.0 * (2.0 * 10.0 * 3.0 + 1.0) / 3.0;
    time("resample_poly 1/3", block * (1.0 + 1.0 / 3.0), resample_ops, &mut || {
        resample_poly::<f32>(client, &x, &resampled, CHANNELS, SAMPLES, 1, 3, &window, dspk_ops::resampler::RESAMPLE_POLY_DEFAULT_EDGE);
    });
    let (mean, std) = (buffer::empty::<f32>(client, CHANNELS), buffer::empty::<f32>(client, CHANNELS));
    time("channel mean/std", block, 0.0, &mut || execute_channel_mean_std::<f32>(client, &x, &mean, &std, CHANNELS, SAMPLES));
    time("channel noise (median |x|)", block, 0.0, &mut || {
        execute_channel_noise_std::<f32>(client, &x, CHANNELS, SAMPLES, 0..SAMPLES);
    });
    // Negative spikes, searched as minima (maxima of −x) above 5σ
    let spikes = buffer::upload(client, &spiky(CHANNELS, SAMPLES));
    let heights: Handle = buffer::upload(client, &vec![SPIKE_HEIGHT; CHANNELS]);
    time("peak candidates (spikes)", block, 0.0, &mut || {
        let _ = find_peak_candidates_on_device::<f32>(client, &spikes, &heights, CHANNELS, SAMPLES, 0..SAMPLES, Polarity::Negative);
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
        println!("{:<30} {:>10} {:>10} {:>9} {:>10}", "kernel", "ms", "GB/s", "of ceil.", "Gop/s");
        for r in &rows {
            let gbs = r.bytes / r.ms / 1e6;
            let gops = if r.ops > 0.0 { format!("{:.0}", r.ops / r.ms / 1e6) } else { String::new() };
            println!("{:<30} {:>10.3} {:>10.1} {:>8.0}% {:>10}", r.name, r.ms, gbs, 100.0 * gbs / ceiling, gops);
        }
    }
}
