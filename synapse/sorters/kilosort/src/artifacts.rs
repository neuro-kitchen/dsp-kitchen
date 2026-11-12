//! Kilosort4's `artifact_threshold`: a window holding a value at or above the threshold is taken
//! to contain a recording artifact and zeroed (upstream zeroes the batch). Off by default, as
//! upstream (∞).
//!
//! Upstream compares the filtered data before whitening, in the recording's int16 units; here the
//! test runs on the whitened window, in σ, so one setting means the same on every recording's
//! scale. It runs from template learning on (not in the whitening fit).
//!
//! **What crosses the bus:** nothing. A flag is raised and read on the device, so the windows are
//! never waited for.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::{element_position, LaunchGeometry};
use dspk_core::HaloWindow;
use dspk_ops::core::{buffer, DspFloat};

/// `flag[0] += 1` for every `|x[i]| ≥ threshold`. One unit per value.
#[cube(launch)]
pub fn artifact_flag_kernel<F: DspFloat>(x: &[F], flag: &mut [Atomic<u32>], len: u32, threshold: F) {
    let i = element_position() as u32;
    if i < len && F::abs(x[i as usize]) >= threshold {
        Atomic::fetch_add(&flag[0], 1u32);
    }
}

/// `out[i] = 0` when `flag[0] > 0`, else `x[i]`. One unit per value.
#[cube(launch)]
pub fn blank_flagged_kernel<F: Float>(x: &[F], out: &mut [F], flag: &[u32], len: u32) {
    let i = element_position() as u32;
    if i < len {
        if flag[0] > 0u32 {
            out[i as usize] = F::new(0.0f32);
        } else {
            out[i as usize] = x[i as usize];
        }
    }
}

/// Zeroes the windows that hold an artifact (module docs), as a per-window step.
pub struct ArtifactBlanker {
    client: Client,
    channels: usize,
    threshold: f32,
    output: Handle,
    max_samples: usize,
}

impl ArtifactBlanker {
    pub fn new(client: &Client, channels: usize, max_samples: usize, threshold: f32) -> Self {
        Self { client: client.clone(), channels, threshold, output: buffer::empty::<f32>(client, channels * max_samples), max_samples }
    }

    /// The window, or zeros if it holds an artifact (the blanker's buffer: valid until the next
    /// call).
    pub fn blank(&mut self, input: &Handle, samples: usize) -> Handle {
        assert!(samples <= self.max_samples, "window of {samples} samples exceeds the blanker's {}", self.max_samples);
        let len = self.channels * samples;
        let flag = buffer::zeros::<u32>(&self.client, 1);
        let geom = LaunchGeometry::elementwise(&self.client, len);
        // SAFETY: `input` and `output` hold `len` values, `flag` one
        unsafe {
            artifact_flag_kernel::launch::<f32>(
                &self.client,
                geom.cube_count.clone(),
                geom.cube_dim.clone(),
                BufferArg::from_raw_parts(input.clone(), len),
                BufferArg::from_raw_parts(flag.clone(), 1),
                len as u32,
                self.threshold,
            );
            blank_flagged_kernel::launch::<f32>(
                &self.client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), len),
                BufferArg::from_raw_parts(self.output.clone(), len),
                BufferArg::from_raw_parts(flag, 1),
                len as u32,
            );
        }
        self.output.clone()
    }
}

impl dspk_synapse::sorter::WindowAligner for ArtifactBlanker {
    fn align(&mut self, window: &Handle, _at: &HaloWindow, samples: usize) -> Handle {
        self.blank(window, samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::compute::ComputeTarget;

    /// A window with one value past the threshold is zeroed; one below it passes unchanged.
    #[test]
    fn artifact_windows_are_zeroed() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, samples) = (3usize, 50usize);
        let mut blanker = ArtifactBlanker::new(&client, channels, samples, 20.0);
        let quiet: Vec<f32> = (0..channels * samples).map(|i| ((i % 7) as f32 - 3.0) * 2.0).collect();
        let out = buffer::download_prefix::<f32>(&client, blanker.blank(&buffer::upload(&client, &quiet), samples), channels * samples);
        assert_eq!(out, quiet, "below the threshold: unchanged");
        let mut loud = quiet.clone();
        loud[77] = -25.0;
        let out = buffer::download_prefix::<f32>(&client, blanker.blank(&buffer::upload(&client, &loud), samples), channels * samples);
        assert!(out.iter().all(|&v| v == 0.0), "an artifact: zeroed");
    }
}
