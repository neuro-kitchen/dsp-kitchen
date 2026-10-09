//! Features of every peak on its neighbouring channels, as SpikeInterface's `extract_peaks_svd`
//! (`sortingcomponents/waveforms/peak_svd.py`, MIT), which SpyKING CIRCUS 2 and Tridesclous 2
//! cluster on.
//!
//! - **Fit** ([`LocalSvd::fit`]): at most `n_peaks_fit` peaks (uniform, [`subsample_peaks`]); each
//!   gives its waveform on its own channel, `ms_before` before to `ms_after` after the peak. Kept
//!   only when its largest `|sample|` is at the peak, then signed so the peak sample is positive. The
//!   components are the top `n_components` right singular vectors of those rows: sklearn's
//!   `TruncatedSVD` (no centring), here the top eigenvectors of the rows' Gram matrix `XᵀX`, each
//!   signed so its largest-magnitude entry is positive (sklearn's `svd_flip`, right-vector based).
//! - **Transform** ([`LocalSvd::transform`]): every peak's waveform on each channel within
//!   `radius_um` of its own ([`ChannelNeighbourhoods`]), projected on the components:
//!   `[peaks, n_components, max_neighbours]`, channels in increasing order, 0 past a peak's own
//!   neighbourhood (upstream's unaligned sparse layout).
//!
//! On the device: the fit gathers the rows (one kernel), multiplies `XᵀX` (`matmul`) and solves the
//! `[width, width]` eigenproblem (Jacobi); only the components come back. The transform is one
//! kernel over every (peak, component, channel); the features stay on the device. Peak samples are
//! indices into the given `[channels, samples]` buffer; callers leave `ms_before` / `ms_after` of
//! margin (samples past the buffer read its edge sample).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, symmetric_eigen, EigenOptions, MatrixView};
use dsp_core::compute::LaunchGeometry;

use super::kernels::local_svd::{gather_fit_waveforms_kernel, project_local_kernel, NO_CHANNEL};
use crate::sorting::{subsample_peaks, SubsampleOptions};

/// Settings of [`LocalSvd`] (upstream's defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSvdOptions {
    pub n_components: usize,
    /// Window before and after the peak (ms).
    pub ms_before: f64,
    pub ms_after: f64,
    /// Channels within this distance of a peak's channel (µm) get features.
    pub radius_um: f32,
    /// Most peaks the components are fitted on.
    pub n_peaks_fit: usize,
    pub seed: u64,
}

impl Default for LocalSvdOptions {
    fn default() -> Self {
        Self { n_components: 5, ms_before: 0.5, ms_after: 1.5, radius_um: 120.0, n_peaks_fit: 5000, seed: 0 }
    }
}

impl LocalSvdOptions {
    /// `(n_before, width)` in samples at `sample_rate_hz`.
    pub fn window(&self, sample_rate_hz: f64) -> (usize, usize) {
        let before = (self.ms_before * sample_rate_hz / 1000.0).round() as usize;
        let after = (self.ms_after * sample_rate_hz / 1000.0).round() as usize;
        (before, before + after)
    }
}

/// Channels within a radius of each channel, in increasing order: row `ch` of
/// `[channels, max_neighbours]`, padded with [`NO_CHANNEL`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelNeighbourhoods {
    pub table: Vec<u32>,
    pub max_neighbours: usize,
}

impl ChannelNeighbourhoods {
    /// Neighbourhoods of `positions` (`(x, y)` µm per channel) within `radius_um` (inclusive).
    pub fn within_radius(positions: &[[f32; 2]], radius_um: f32) -> Self {
        let rows: Vec<Vec<u32>> = positions
            .iter()
            .map(|a| {
                positions
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| (a[0] - b[0]).hypot(a[1] - b[1]) <= radius_um)
                    .map(|(j, _)| j as u32)
                    .collect()
            })
            .collect();
        let max_neighbours = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut table = vec![NO_CHANNEL; positions.len() * max_neighbours];
        for (ch, row) in rows.iter().enumerate() {
            table[ch * max_neighbours..ch * max_neighbours + row.len()].copy_from_slice(row);
        }
        Self { table, max_neighbours }
    }

    /// Channels of `channel`'s neighbourhood.
    pub fn of(&self, channel: usize) -> impl Iterator<Item = usize> + '_ {
        self.table[channel * self.max_neighbours..(channel + 1) * self.max_neighbours].iter().take_while(|&&c| c != NO_CHANNEL).map(|&c| c as usize)
    }
}

/// Fitted components (module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSvd {
    /// `[n_components, width]`, row `c` the `c`-th component.
    pub components: Vec<f32>,
    pub n_components: usize,
    pub n_before: usize,
    pub width: usize,
    /// Peaks the fit used (after the shape check).
    pub n_fit: usize,
}

impl LocalSvd {
    /// Fits the components on the peaks of `data` (`[channels, samples]` `f32` on `client`'s
    /// device), peak `i` at sample `peak_samples[i]` of channel `peak_channels[i]`.
    ///
    /// # Panics
    ///
    /// If the peak arrays differ in length, or no peak passes the shape check (fewer valid rows
    /// than `n_components` cannot give the components).
    pub fn fit(
        client: &Client,
        data: &Handle,
        channels: usize,
        samples: usize,
        peak_samples: &[u32],
        peak_channels: &[u32],
        sample_rate_hz: f64,
        opts: &LocalSvdOptions,
    ) -> Self {
        assert_eq!(peak_samples.len(), peak_channels.len(), "one channel per peak");
        let (n_before, width) = opts.window(sample_rate_hz);
        let s64: Vec<u64> = peak_samples.iter().map(|&s| s as u64).collect();
        let ch: Vec<usize> = peak_channels.iter().map(|&c| c as usize).collect();
        let chosen = subsample_peaks(&s64, &ch, &SubsampleOptions { n_peaks: opts.n_peaks_fit, per_channel: false, seed: opts.seed });
        let fit_samples: Vec<u32> = chosen.iter().map(|&i| peak_samples[i]).collect();
        let fit_channels: Vec<u32> = chosen.iter().map(|&i| peak_channels[i]).collect();
        let n = chosen.len();
        assert!(n >= opts.n_components, "{n} peaks cannot give {} components", opts.n_components);

        let rows = buffer::empty::<f32>(client, n * width);
        let valid = buffer::empty::<u32>(client, n);
        let geom = LaunchGeometry::elementwise(client, n);
        // SAFETY: `data` holds `channels · samples`, the peak arrays `n`, `rows` `n · width`
        unsafe {
            gather_fit_waveforms_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(data.clone(), channels * samples),
                BufferArg::from_raw_parts(buffer::upload(client, &fit_samples), n),
                BufferArg::from_raw_parts(buffer::upload(client, &fit_channels), n),
                BufferArg::from_raw_parts(rows.clone(), n * width),
                BufferArg::from_raw_parts(valid.clone(), n),
                n as u32,
                samples as u32,
                n_before as u32,
                width as u32,
            );
        }
        // XᵀX: rejected rows are zeros and add nothing
        let gram = buffer::empty::<f32>(client, width * width);
        let x = MatrixView::row_major(&rows, n * width, n, width);
        matmul::<f32>(client, &x.transposed(), &x, &gram, width * width);
        let eig = symmetric_eigen::<f32>(client, &gram, width, EigenOptions::default());
        let n_fit = buffer::download::<u32>(client, valid).iter().filter(|&&v| v == 1).count();
        assert!(n_fit >= opts.n_components, "{n_fit} peaks pass the shape check, fewer than {} components", opts.n_components);

        // Columns of `vectors` are eigenvectors, largest eigenvalue first
        let mut components = vec![0.0f32; opts.n_components * width];
        for c in 0..opts.n_components {
            let col: Vec<f64> = (0..width).map(|t| eig.vectors[t * width + c]).collect();
            let big = col.iter().copied().fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
            let sign = if big < 0.0 { -1.0 } else { 1.0 };
            for t in 0..width {
                components[c * width + t] = (sign * col[t]) as f32;
            }
        }
        Self { components, n_components: opts.n_components, n_before, width, n_fit }
    }

    /// Features of every peak on its neighbourhood: a `[peaks, n_components, max_neighbours]`
    /// device buffer of `f32` (module docs).
    ///
    /// # Panics
    ///
    /// If the peak arrays differ in length or a peak's channel has no neighbourhood row.
    #[allow(clippy::too_many_arguments)]
    pub fn transform(
        &self,
        client: &Client,
        data: &Handle,
        channels: usize,
        samples: usize,
        peak_samples: &[u32],
        peak_channels: &[u32],
        neighbours: &ChannelNeighbourhoods,
    ) -> Handle {
        assert_eq!(peak_samples.len(), peak_channels.len(), "one channel per peak");
        let rows = neighbours.table.len() / neighbours.max_neighbours;
        assert!(peak_channels.iter().all(|&c| (c as usize) < rows), "a peak channel has no neighbourhood");
        let (n, m) = (peak_samples.len(), neighbours.max_neighbours);
        let total = n * self.n_components * m;
        let features = buffer::empty::<f32>(client, total);
        if total == 0 {
            return features;
        }
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            project_local_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(data.clone(), channels * samples),
                BufferArg::from_raw_parts(buffer::upload(client, peak_samples), n),
                BufferArg::from_raw_parts(buffer::upload(client, peak_channels), n),
                BufferArg::from_raw_parts(buffer::upload(client, &neighbours.table), neighbours.table.len()),
                BufferArg::from_raw_parts(buffer::upload(client, &self.components), self.components.len()),
                BufferArg::from_raw_parts(features.clone(), total),
                n as u32,
                samples as u32,
                self.n_before as u32,
                self.width as u32,
                self.n_components as u32,
                m as u32,
            );
        }
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    const FS: f64 = 30_000.0;

    fn client() -> Option<Client> {
        ComputeTarget::from_env().ok()?.client().ok()
    }

    /// Two waveform shapes (the trough at the peak, rebounds after): every spike is a mix, so the
    /// first two components span the data.
    fn shapes(width: usize, n_before: usize) -> [Vec<f32>; 2] {
        let a = (0..width).map(|t| { let d = t as f32 - n_before as f32; -(-0.5 * (d / 2.0).powi(2)).exp() }).collect();
        let b = (0..width).map(|t| { let d = t as f32 - n_before as f32 - 8.0; 0.3 * (-0.5 * (d / 4.0).powi(2)).exp() }).collect();
        [a, b]
    }

    #[test]
    fn neighbourhoods_follow_the_radius() {
        let pos: Vec<[f32; 2]> = (0..5).map(|i| [0.0, 20.0 * i as f32]).collect();
        let n = ChannelNeighbourhoods::within_radius(&pos, 25.0);
        assert_eq!(n.max_neighbours, 3);
        assert_eq!(n.of(0).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(n.of(2).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn components_span_the_waveforms_and_features_match_the_host() {
        let Some(client) = client() else { return };
        let opts = LocalSvdOptions { n_components: 2, ..Default::default() };
        let (n_before, width) = opts.window(FS);
        let [a, b] = shapes(width, n_before);
        // 4 channels, 200 spikes, each at its own time; channel 1 carries the spike, its
        // neighbours a third of it
        let (channels, samples) = (4usize, 200 * 100);
        let mut data = vec![0.0f32; channels * samples];
        let mut peak_samples = Vec::new();
        for k in 0..200 {
            let t0 = k * 100 + 20;
            let (wa, wb) = (40.0 + (k % 7) as f32, 20.0 * ((k % 5) as f32 / 4.0));
            for t in 0..width {
                let v = wa * a[t] + wb * b[t];
                data[samples + t0 + t] += v;
                data[t0 + t] += v / 3.0;
                data[2 * samples + t0 + t] += v / 3.0;
            }
            peak_samples.push((t0 + n_before) as u32);
        }
        let peak_channels = vec![1u32; peak_samples.len()];
        let handle = buffer::upload(&client, &data);
        let svd = LocalSvd::fit(&client, &handle, channels, samples, &peak_samples, &peak_channels, FS, &opts);
        assert_eq!(svd.n_fit, 200);

        // Rank 2: every waveform is (almost) its projection on the two components
        for k in [0usize, 37, 199] {
            let t0 = peak_samples[k] as usize - n_before;
            let w: Vec<f32> = (0..width).map(|t| data[samples + t0 + t]).collect();
            let coef: Vec<f32> = (0..2).map(|c| (0..width).map(|t| w[t] * svd.components[c * width + t]).sum()).collect();
            let err: f32 = (0..width).map(|t| (w[t] - coef[0] * svd.components[t] - coef[1] * svd.components[width + t]).powi(2)).sum();
            let energy: f32 = w.iter().map(|v| v * v).sum();
            assert!(err / energy < 1e-4, "spike {k}: residual {:.2e}", err / energy);
        }

        // Transform: device features equal the host dot products on each neighbourhood
        let pos: Vec<[f32; 2]> = (0..channels).map(|i| [0.0, 25.0 * i as f32]).collect();
        let nb = ChannelNeighbourhoods::within_radius(&pos, 30.0);
        let feats = buffer::download::<f32>(&client, svd.transform(&client, &handle, channels, samples, &peak_samples, &peak_channels, &nb));
        let m = nb.max_neighbours;
        assert_eq!(feats.len(), peak_samples.len() * 2 * m);
        for p in [0usize, 100] {
            let t0 = peak_samples[p] as usize - n_before;
            for (w, ch) in nb.of(1).enumerate() {
                for c in 0..2 {
                    let host: f32 = (0..width).map(|t| data[ch * samples + t0 + t] * svd.components[c * width + t]).sum();
                    let dev = feats[(p * 2 + c) * m + w];
                    assert!((host - dev).abs() <= 1e-3 * host.abs().max(1.0), "peak {p} ch {ch} c {c}: {host} vs {dev}");
                }
            }
        }
    }

    /// A waveform whose largest sample is not at the peak is left out of the fit.
    #[test]
    fn misaligned_waveforms_are_left_out() {
        let Some(client) = client() else { return };
        let opts = LocalSvdOptions { n_components: 1, ..Default::default() };
        let (n_before, width) = opts.window(FS);
        let (channels, samples) = (1usize, 20 * 100);
        let mut data = vec![0.0f32; samples];
        let mut peak_samples = Vec::new();
        for k in 0..20 {
            let t0 = k * 100 + 10;
            // Odd spikes: the largest sample is 5 samples after the peak
            let at = if k % 2 == 0 { n_before } else { n_before + 5 };
            data[t0 + at] = -10.0;
            data[t0 + n_before] += -1.0;
            peak_samples.push((t0 + n_before) as u32);
        }
        let handle = buffer::upload(&client, &data);
        let svd = LocalSvd::fit(&client, &handle, channels, samples, &peak_samples, &vec![0; 20], FS, &opts);
        assert_eq!(svd.n_fit, 10);
        // Signed positive at the peak: the component's largest entry is at the peak, positive
        let comp = &svd.components[..width];
        let (arg, _) = comp.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        assert_eq!(arg, n_before);
    }
}
