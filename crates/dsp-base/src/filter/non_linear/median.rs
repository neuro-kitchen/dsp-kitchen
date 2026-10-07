use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use super::kernels::median_filter_kernel;
use crate::core::{DspFloat, EdgeMode};

/// Samples on each side of the centre in the 9-point median window.
pub const MEDIAN9_RADIUS: usize = 4;

/// Widest running-median window: the window is held in registers and ranked in `width²` steps.
pub const MAX_MEDIAN_WIDTH: usize = 31;

/// Edge handling of the running median, matching `scipy.signal.medfilt` (zero padding).
pub const MEDIAN_DEFAULT_EDGE: EdgeMode = EdgeMode::Zeros;

/// Running median over an odd `width` (≤ [`MAX_MEDIAN_WIDTH`]) along time of a `[channels, samples]`
/// buffer.
///
/// # Panics
/// If `width` is even, zero, or larger than [`MAX_MEDIAN_WIDTH`].
pub fn execute_median<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    width: usize,
    edge: EdgeMode,
) {
    assert!(width % 2 == 1 && width <= MAX_MEDIAN_WIDTH, "median width {width} must be odd and at most {MAX_MEDIAN_WIDTH}");
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;
    unsafe {
        median_filter_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elements),
            BufferArg::from_raw_parts(output.clone(), total_elements),
            channels as u32,
            samples as u32,
            width as u32,
            edge.id(),
        );
    }
}

/// 9-point running median with [`MEDIAN_DEFAULT_EDGE`] (the branch-free `med9` network).
pub fn execute_median_9p<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    execute_median::<F>(client, input, output, channels, samples, 2 * MEDIAN9_RADIUS + 1, MEDIAN_DEFAULT_EDGE);
}
