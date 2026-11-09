//! Tunes every autotuned operation on this device ahead of time.
//!
//! An operation with several kernels benchmarks them the first time it meets a device and problem
//! size, and the choice is saved for later processes. [`autotune`] does that work up front, over a
//! grid of common sizes on recording-like data, and compiles the kernels on the way, so a first
//! real run neither stops to benchmark nor to compile. A size off the grid still tunes when it is
//! first met.

use std::time::Instant;

use cubecl::prelude::*;

use crate::core::buffer;
use crate::core::edge::EdgeMode;
use crate::error::OpsResult;
use crate::peaks::{find_peak_candidates, Polarity};
use crate::pipeline::{Pipeline, PipelineStage};
use crate::resampler::poly::{execute_resample_poly, resample_poly_len};
use crate::resampler::ResampleFilter;
use crate::spatial::SpatialWhitening;

/// Channel counts tuned by default.
pub const DEFAULT_CHANNELS: [usize; 4] = [32, 64, 128, 384];
/// Samples per channel tuned by default.
pub const DEFAULT_SAMPLES: [usize; 4] = [1 << 13, 1 << 15, 1 << 17, 1 << 19];
/// Sample rate (Hz) of the data tuned on.
pub const DEFAULT_SAMPLE_RATE: f64 = 30_000.0;
/// Sizes above this many values are left out by default (256 MB of `f32`, per buffer).
pub const DEFAULT_MAX_VALUES: usize = 1 << 26;

/// What [`autotune`] covers.
#[derive(Clone, Debug)]
pub struct AutotuneOptions {
    pub channels: Vec<usize>,
    pub samples: Vec<usize>,
    pub sample_rate: f64,
    pub max_values: usize,
}

impl Default for AutotuneOptions {
    fn default() -> Self {
        Self { channels: DEFAULT_CHANNELS.to_vec(), samples: DEFAULT_SAMPLES.to_vec(), sample_rate: DEFAULT_SAMPLE_RATE, max_values: DEFAULT_MAX_VALUES }
    }
}

/// What a run of [`autotune`] did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutotuneSummary {
    /// Channel × sample sizes covered.
    pub sizes: usize,
    pub seconds: f64,
}

/// One operation per tuner. Matching what a run does matters: the keys include the filter order, the
/// window width and the resampling ratio.
fn stages(sample_rate: f64) -> Vec<PipelineStage> {
    vec![
        PipelineStage::bandpass(300.0, (sample_rate * 0.4).min(6_000.0)),
        PipelineStage::fir(vec![1.0 / 31.0; 31]),
        PipelineStage::median(5),
        PipelineStage::common_average_reference(),
        PipelineStage::common_median_reference(),
    ]
}

/// Resampling ratios tuned, `(up, down)`.
const RATIOS: [(usize, usize); 2] = [(1, 2), (1, 30)];

/// Noise with sparse negative spikes, `[channels, samples]` channel-major: median and sorting
/// kernels run differently on real signals than on constants.
fn recording_like(channels: usize, samples: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    (0..channels * samples)
        .map(|i| {
            let noise = (0..4).map(|_| uniform()).sum::<f32>();
            if (i * 7919) % 1009 == 0 { noise - 8.0 } else { noise }
        })
        .collect()
}

/// Runs every tuned operation over the grid on `client`'s device. `progress(done, total)` is called
/// after each operation.
pub fn autotune(client: &Client, options: &AutotuneOptions, mut progress: impl FnMut(usize, usize)) -> OpsResult<AutotuneSummary> {
    let start = Instant::now();
    let sizes: Vec<(usize, usize)> = options
        .channels
        .iter()
        .flat_map(|&c| options.samples.iter().map(move |&s| (c, s)))
        .filter(|&(c, s)| c > 0 && s > 0 && c * s <= options.max_values)
        .collect();
    let stage_list = stages(options.sample_rate);
    // Pipeline stages, resampling ratios, peaks and whitening
    let total = sizes.len() * (stage_list.len() + RATIOS.len() + 2);
    let mut done = 0;
    let mut advance = |progress: &mut dyn FnMut(usize, usize)| {
        done += 1;
        progress(done, total);
    };
    for &(channels, samples) in &sizes {
        let data = recording_like(channels, samples);
        let input = buffer::upload(client, &data);
        for stage in &stage_list {
            run(client, &input, stage.clone(), channels, samples, options.sample_rate)?;
            advance(&mut progress);
        }
        for (up, down) in RATIOS {
            let len = resample_poly_len(samples, up, down);
            let output = buffer::empty::<f32>(client, channels * len);
            let filter = ResampleFilter::default();
            execute_resample_poly::<f32>(client, &input, &output, channels, samples, up, down, &filter, EdgeMode::Zeros);
            let _ = buffer::download_prefix::<f32>(client, output, 1);
            advance(&mut progress);
        }
        let heights = buffer::upload(client, &vec![4.0_f32; channels]);
        let _ = find_peak_candidates::<f32>(client, &input, &heights, channels, samples, 0..samples, Polarity::Negative);
        advance(&mut progress);
        // Fitted on a short stretch: only applying it is tuned
        let fit = samples.min(4 * channels.max(256));
        let whitening = SpatialWhitening::fit_zca::<f32>(client, &recording_like(channels, fit), channels, fit, 1e-6);
        run(client, &input, PipelineStage::SpatialWhitening(whitening), channels, samples, options.sample_rate)?;
        advance(&mut progress);
    }
    Ok(AutotuneSummary { sizes: sizes.len(), seconds: start.elapsed().as_secs_f64() })
}

fn run(client: &Client, input: &cubecl::server::Handle, stage: PipelineStage, channels: usize, samples: usize, sample_rate: f64) -> OpsResult<()> {
    let mut pipeline = Pipeline::new();
    pipeline.add(stage);
    pipeline.validate(sample_rate)?;
    let output = pipeline.execute::<f32>(client, input, channels, samples, sample_rate)?;
    let _ = buffer::download_prefix::<f32>(client, output, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covers_the_grid(client: &Client) {
        let options = AutotuneOptions { channels: vec![4], samples: vec![512, 4096], ..Default::default() };
        let mut last = (0, 0);
        let summary = autotune(client, &options, |d, t| last = (d, t)).unwrap();
        assert_eq!(summary.sizes, 2, "{}", client.name());
        assert_eq!(last.0, last.1, "{}: every step reported", client.name());
    }
    runtime_test!(test_autotune_covers_the_grid, covers_the_grid);

    fn leaves_out_sizes_over_the_limit(client: &Client) {
        let options = AutotuneOptions { channels: vec![4], samples: vec![256, 1024], max_values: 4 * 256, ..Default::default() };
        assert_eq!(autotune(client, &options, |_, _| {}).unwrap().sizes, 1, "{}", client.name());
    }
    runtime_test!(test_autotune_leaves_out_sizes_over_the_limit, leaves_out_sizes_over_the_limit);
}
