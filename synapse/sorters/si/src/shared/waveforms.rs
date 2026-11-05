//! Waveforms around peaks, as SpyKING CIRCUS 2 and Tridesclous 2 cut them: `n_before` samples
//! before each peak to `n_after` after, on the channels within `radius_um` of the peak's channel
//! (both sorters use radius 0: the peak's own channel, plus any channel at the same position).
//! Only those channels are stored, `[events, width, max_neighbours]` (padding slots 0), gathered on
//! the device and downloaded once per call.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::{element_position, LaunchGeometry};
use dspk_ops::core::buffer;
use dspk_synapse::features::kernels::NO_CHANNEL;
use dspk_synapse::features::ChannelNeighbourhoods;

/// One unit per element `(j, s, k)` of `out` (`[events, width, max_neighbours]`): sample
/// `ev_samples[j] − n_before + s` of the `k`-th neighbourhood channel of `ev_channels[j]` (`table`,
/// padded with [`NO_CHANNEL`], which gives 0). The caller keeps every window inside the trace.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_waveforms_kernel<F: Float>(
    trace: &[F],
    ev_samples: &[u32],
    ev_channels: &[u32],
    table: &[u32],
    out: &mut [F],
    total: u32,
    samples: u32,
    width: u32,
    max_neighbours: u32,
    n_before: u32,
) {
    let e = element_position() as u32;
    if e < total {
        let k = e % max_neighbours;
        let s = (e / max_neighbours) % width;
        let j = e / (max_neighbours * width);
        let nb = table[(ev_channels[j as usize] * max_neighbours + k) as usize];
        let mut v = F::new(0.0f32);
        if nb != NO_CHANNEL {
            v = trace[(nb * samples + ev_samples[j as usize] + s - n_before) as usize];
        }
        out[e as usize] = v;
    }
}

/// The waveforms of many peaks (module docs).
#[derive(Debug, Clone)]
pub struct PeakWaveforms {
    n_before: usize,
    width: usize,
    channels: usize,
    neighbours: ChannelNeighbourhoods,
    events: usize,
    /// `[events, width, max_neighbours]`.
    values: Vec<f32>,
}

impl PeakWaveforms {
    /// No peaks yet. `positions`: `(x, y)` µm per channel.
    pub fn new(positions: &[[f32; 2]], radius_um: f32, n_before: usize, n_after: usize) -> Self {
        Self {
            n_before,
            width: n_before + n_after,
            channels: positions.len(),
            neighbours: ChannelNeighbourhoods::within_radius(positions, radius_um),
            events: 0,
            values: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.events
    }

    pub fn is_empty(&self) -> bool {
        self.events == 0
    }

    fn row_len(&self) -> usize {
        self.width * self.neighbours.max_neighbours
    }

    /// Appends the waveforms of peaks at local samples `samples` on `channels` of the
    /// `[channels, n_samples]` device trace.
    ///
    /// # Panics
    ///
    /// If a window leaves the trace, or the arrays differ in length.
    pub fn extract(&mut self, client: &Client, trace: &Handle, n_samples: usize, samples: &[u32], channels: &[u32]) {
        assert_eq!(samples.len(), channels.len(), "one channel per peak");
        let n = samples.len();
        if n == 0 {
            return;
        }
        assert!(
            samples.iter().all(|&t| t as usize >= self.n_before && t as usize + self.width - self.n_before <= n_samples),
            "waveform windows must lie inside the trace"
        );
        let total = n * self.row_len();
        let out = buffer::empty::<f32>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        let table = &self.neighbours.table;
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            gather_waveforms_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), self.channels * n_samples),
                BufferArg::from_raw_parts(buffer::upload(client, samples), n),
                BufferArg::from_raw_parts(buffer::upload(client, channels), n),
                BufferArg::from_raw_parts(buffer::upload(client, table), table.len()),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                n_samples as u32,
                self.width as u32,
                self.neighbours.max_neighbours as u32,
                self.n_before as u32,
            );
        }
        self.values.extend(buffer::download::<f32>(client, out));
        self.events += n;
    }

    /// Peak `i`'s waveform, `[width, max_neighbours]`.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.row_len()..(i + 1) * self.row_len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::compute::ComputeTarget;

    /// Radius 0 cuts each peak's own channel; samples are the trace's, `n_before` before the peak.
    #[test]
    fn own_channel_waveforms_are_the_trace_around_each_peak() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, n) = (3usize, 50usize);
        let positions: Vec<[f32; 2]> = (0..channels).map(|c| [0.0, 25.0 * c as f32]).collect();
        let x: Vec<f32> = (0..channels * n).map(|i| i as f32).collect();
        let trace = buffer::upload(&client, &x);
        let mut w = PeakWaveforms::new(&positions, 0.0, 2, 3);
        w.extract(&client, &trace, n, &[10, 20], &[1, 2]);
        assert_eq!(w.len(), 2);
        assert_eq!(w.row(0), &[58.0, 59.0, 60.0, 61.0, 62.0]);
        assert_eq!(w.row(1), &[118.0, 119.0, 120.0, 121.0, 122.0]);
    }
}
