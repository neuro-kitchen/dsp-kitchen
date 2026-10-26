//! Template subtraction on the device, equal to [`super::subtract_template_multichannel`].
//!
//! The host subtracts events one by one: an event's alignment sees the residual left by earlier
//! events whose windows overlap it. Here events are grouped into layers — an event goes one layer
//! after the latest earlier event it overlaps — so events of a layer never touch the same samples
//! and run in parallel, while overlapping events still run in their original order.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::subtraction::{TemplateFilter, MIN_TEMPLATE_ENERGY};
use super::kernels::{template_scores_kernel, template_subtract_kernel};
use crate::core::{buffer, cast, cast_f32, DspFloat};

/// Layer of every event (in input order): one after the latest earlier event whose search window
/// (`template_len + 2 · max_lag` samples) overlaps it.
fn layers(starts: &[i64], window: i64) -> Vec<usize> {
    // Earlier events by window start; values: their layers
    let mut placed: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    starts
        .iter()
        .map(|&s| {
            let layer = placed
                .range(s - window + 1..s + window)
                .flat_map(|(_, ls)| ls.iter().copied())
                .max()
                .map_or(0, |l| l + 1);
            placed.entry(s).or_default().push(layer);
            layer
        })
        .collect()
}

impl TemplateFilter {
    /// Subtracts the template at every event from a `[channels, samples]` device buffer of `F` in
    /// place, with the same alignment, scaling and event order as [`Self::apply_multichannel`]
    /// (a single-channel template works on a single-channel signal).
    pub fn apply_device<F: DspFloat>(
        &self,
        client: &Client,
        signal: &Handle,
        channels: usize,
        samples: usize,
        event_indices: &[u64],
    ) {
        assert_eq!(channels, self.template_channels, "channel count must match between signal and template");
        let t_len = self.template_samples;
        if t_len == 0 || samples == 0 || event_indices.is_empty() {
            return;
        }
        let energies: Vec<f32> = self.template.chunks(t_len).map(|row| row.iter().map(|v| v * v).sum()).collect();
        let template = buffer::upload(client, &cast_f32::<F>(&self.template));
        let energies = buffer::upload(client, &cast_f32::<F>(&energies));
        let lags = 2 * self.max_lag + 1;

        // Window start of each event at zero lag, and its layer
        let starts: Vec<i64> = event_indices.iter().map(|&ev| ev as i64 - self.center_offset as i64).collect();
        let event_layer = layers(&starts, (t_len + 2 * self.max_lag) as i64);
        let num_layers = event_layer.iter().max().map_or(0, |l| l + 1);

        for layer in 0..num_layers {
            let layer_starts: Vec<i32> =
                starts.iter().zip(&event_layer).filter(|(_, l)| **l == layer).map(|(s, _)| *s as i32).collect();
            let n = layer_starts.len();
            let starts_h = buffer::upload(client, &layer_starts);
            let scores = buffer::empty::<F>(client, n * lags);

            let geom = LaunchGeometry::elementwise(client, n * lags);
            unsafe {
                template_scores_kernel::launch::<F>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(signal.clone(), channels * samples),
                    BufferArg::from_raw_parts(template.clone(), channels * t_len),
                    BufferArg::from_raw_parts(starts_h.clone(), n),
                    BufferArg::from_raw_parts(scores.clone(), n * lags),
                    channels as u32,
                    samples as u32,
                    t_len as u32,
                    n as u32,
                    lags as u32,
                    self.max_lag as u32,
                );
            }
            let geom = LaunchGeometry::elementwise(client, n * channels);
            unsafe {
                template_subtract_kernel::launch::<F>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(signal.clone(), channels * samples),
                    BufferArg::from_raw_parts(template.clone(), channels * t_len),
                    BufferArg::from_raw_parts(energies.clone(), channels),
                    BufferArg::from_raw_parts(starts_h, n),
                    BufferArg::from_raw_parts(scores, n * lags),
                    channels as u32,
                    samples as u32,
                    t_len as u32,
                    n as u32,
                    lags as u32,
                    self.max_lag as u32,
                    self.dynamic_scaling as u32,
                    cast::<F>(MIN_TEMPLATE_ENERGY as f64),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_events_get_later_layers() {
        // Window 10: events 0 and 5 overlap, 30 is free, 8 overlaps both 0 and 5
        assert_eq!(layers(&[0, 5, 30, 8], 10), vec![0, 1, 0, 2]);
    }

    fn matches_host(client: &Client) {
        let (channels, samples, t_len) = (3usize, 400usize, 9usize);
        let template: Vec<f32> = (0..channels * t_len).map(|i| ((i * 37) % 17) as f32 - 8.0).collect();
        let filter = TemplateFilter::new_multichannel(template.clone(), channels, t_len, 4).with_max_lag(3);
        let mut signal: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 23) as f32 * 0.1).collect();
        // Overlapping and isolated events, injected with a lag and a gain
        let events = [50u64, 55, 200, 60, 390, 2];
        for (k, &ev) in events.iter().enumerate() {
            for c in 0..channels {
                for t in 0..t_len {
                    let at = ev as isize - 4 + t as isize + (k % 3) as isize - 1;
                    if (0..samples as isize).contains(&at) {
                        signal[c * samples + at as usize] += (1.5 + k as f32 * 0.2) * template[c * t_len + t];
                    }
                }
            }
        }
        let mut host = signal.clone();
        filter.apply_multichannel(&mut host, channels, samples, &events);

        let device = buffer::upload(client, &signal);
        filter.apply_device::<f32>(client, &device, channels, samples, &events);
        let got = buffer::download::<f32>(client, device);
        for (i, (g, h)) in got.iter().zip(&host).enumerate() {
            assert!((g - h).abs() < 1e-3, "{} sample {i}: {g} vs {h}", client.name());
        }
    }
    runtime_test!(test_template_device_matches_host, matches_host);
}
