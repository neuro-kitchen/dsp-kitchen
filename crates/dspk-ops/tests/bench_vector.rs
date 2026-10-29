//! Vectorized kernels against their scalar versions (`.tasks/12_cpu-vectorization.md`): every
//! kernel choice of an operation on a recording-sized block, one row per choice, so the gain of
//! each vector width shows per runtime (CubeCL's CPU runtime maps `Vector<F, N>` to SIMD). The
//! autotuner picks among the same choices in normal use; this shows all of them.
//!
//! ```text
//! cargo test -p dspk-ops --profile validate --test bench_vector -- --ignored --nocapture
//! ```

use cubecl::prelude::*;
use dspk_core::compute::bench::time_device;
use dspk_core::ComputeTarget;
use dspk_ops::core::{buffer, EdgeMode};
use dspk_ops::filter::fir::{execute_fir_centered_with, execute_fir_with, FirKernel};
use dspk_ops::filter::non_linear::{execute_median_with, MedianKernel};
use dspk_ops::filter::{DeviceFilter, FilterBand, FilterMode, FilterSpec, PassLayout};
use dspk_ops::resampler::{resample_poly_len, resample_poly_with, ResampleFilter, ResampleKernel};
use dspk_ops::spatial::{execute_common_median_with, CmrKernel};

const ITERATIONS: usize = 7;
const CHANNELS: usize = 384;
/// 2 s at 30 kHz.
const SAMPLES: usize = 60_000;
const RADIUS: usize = 31;
const TAPS: usize = 2 * RADIUS + 1;

fn data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i.wrapping_mul(7919)) % 1013) as f32 * 0.01 - 5.0).collect()
}

fn bench(client: &Client) {
    let n = CHANNELS * SAMPLES;
    let x = buffer::upload(client, &data(n));
    let out = buffer::empty::<f32>(client, n);
    let taps_host: Vec<f32> = (0..TAPS).map(|k| ((k * 37) % 17) as f32 * 0.01 - 0.08).collect();
    let taps = buffer::upload(client, &taps_host);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(4).filter(|&w| w > 1).collect();
    println!("  vector widths offered for f32: {widths:?}");
    let kernels: Vec<FirKernel> = [FirKernel::Direct, FirKernel::Tiled].into_iter().chain(widths.iter().map(|&width| FirKernel::Vectorized { width })).collect();
    for centered in [false, true] {
        for &kernel in &kernels {
            let t = time_device(client, ITERATIONS, || {
                if centered {
                    execute_fir_centered_with::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, RADIUS, EdgeMode::Reflect, Some(kernel));
                } else {
                    execute_fir_with::<f32>(client, &x, &out, &taps, CHANNELS, SAMPLES, TAPS, EdgeMode::Reflect, Some(kernel));
                }
            });
            let shape = if centered { "centred" } else { "causal" };
            println!("  FIR {TAPS} taps, {shape:7} {:28} {:8.2} ms", format!("{kernel:?}"), t.as_secs_f64() * 1e3);
        }
    }
    for (up, down) in [(1usize, 3usize), (2, 3), (3, 1)] {
        let resampled = buffer::empty::<f32>(client, CHANNELS * resample_poly_len(SAMPLES, up, down));
        let kernels = std::iter::once(ResampleKernel::Direct).chain(widths.iter().map(|&width| ResampleKernel::Vectorized { width }));
        for kernel in kernels {
            let t = time_device(client, ITERATIONS, || {
                resample_poly_with::<f32>(client, &x, &resampled, CHANNELS, SAMPLES, up, down, &ResampleFilter::default(), EdgeMode::Zeros, Some(kernel));
            });
            println!("  resample {up}/{down}{:13} {:28} {:8.2} ms", "", format!("{kernel:?}"), t.as_secs_f64() * 1e3);
        }
    }
    for width in [5usize, 9, 31] {
        let kernels = std::iter::once(MedianKernel::Direct).chain(widths.iter().map(|&w| MedianKernel::Vectorized { width: w }));
        for kernel in kernels {
            let t = time_device(client, ITERATIONS, || execute_median_with::<f32>(client, &x, &out, CHANNELS, SAMPLES, width, EdgeMode::Zeros, Some(kernel)));
            println!("  median width {width:<11} {:28} {:8.2} ms", format!("{kernel:?}"), t.as_secs_f64() * 1e3);
        }
    }
    // Fifth-order band-pass, forward; blocks per pass fixed so every layout gets the same split
    let spec = FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)).with_mode(FilterMode::Forward);
    let layouts: Vec<PassLayout> = [PassLayout::ChannelMajor, PassLayout::TimeMajor].into_iter().chain(widths.iter().map(|&width| PassLayout::TimeMajorVectorized { width })).collect();
    for blocks in [1usize, 16, 64] {
        for &layout in &layouts {
            let filter = DeviceFilter::<f32>::new(client, &spec, 30_000.0).unwrap().with_block_len(SAMPLES.div_ceil(blocks)).with_layout(layout);
            let scratch = buffer::empty::<f32>(client, filter.scratch_len(CHANNELS, SAMPLES).max(1));
            let state = buffer::empty::<f32>(client, CHANNELS * filter.state_len());
            let t = time_device(client, ITERATIONS, || filter.apply(client, &x, &out, &scratch, &state, CHANNELS, SAMPLES));
            println!("  iir bandpass 5, {blocks:>3} blocks {:28} {:8.2} ms", format!("{layout:?}"), t.as_secs_f64() * 1e3);
        }
    }
    let cpu = client.name() == "cpu";
    let kernels = [CmrKernel::Rank, CmrKernel::Bitonic].into_iter().filter(|&k| !(cpu && k == CmrKernel::Bitonic)).chain(widths.iter().map(|&width| CmrKernel::Vectorized { width }));
    for kernel in kernels {
        let t = time_device(client, ITERATIONS, || execute_common_median_with::<f32>(client, &x, &out, CHANNELS, SAMPLES, Some(kernel)));
        println!("  common median reference  {:28} {:8.2} ms", format!("{kernel:?}"), t.as_secs_f64() * 1e3);
    }
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn vector_widths() {
    for target in ComputeTarget::for_tests() {
        match target.client() {
            Ok(client) => {
                println!("\n{} — {CHANNELS} × {SAMPLES} float32", target.name());
                bench(&client);
            }
            Err(e) => println!("{}: skipped ({e})", target.name()),
        }
    }
}
