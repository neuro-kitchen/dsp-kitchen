//! Waveforms of a sorting's spikes, cut on demand from the recording (they are not stored): windows
//! spread over the recording are read once, snippets of a subset of each unit's spikes are cut
//! from them, and their first principal components are the features the views plot.

use std::collections::HashMap;

use dspk_core::{DspResult, RecordingInfo, RecordingSource};

use super::sorting::{SortingData, DEFAULT_HALF_WIDTH_MS};

/// Waveforms are read from windows spread over **the span the sorting covers** (its first spike
/// to its last: a sorting of the first 200 s of a 47 min recording finds its spikes there), each
/// this long (s), every channel at once and kept for the session (one processing of each window
/// serves every unit and channel): as many as fit in [`window_budget_bytes`], between
/// [`MIN_WINDOWS`] and [`MAX_WINDOWS`]. More windows, more spikes per unit to show.
pub const WINDOW_SEC: f64 = 1.0;
pub const MIN_WINDOWS: usize = 2;
pub const MAX_WINDOWS: usize = 64;

/// RAM the windows may take: an eighth of the app's memory limit
/// ([`dspk_core::memory::host_cache`]), at most 512 MiB.
pub fn window_budget_bytes() -> usize {
    (dspk_core::memory::host_cache().limit() / 8).min(512 << 20) as usize
}

/// Waveforms of some spikes of the sorting view: `snippets[k]` is spike `spikes[k]` on one
/// channel, `before + after` samples.
pub struct Snippets {
    pub channel: usize,
    pub before: u64,
    pub after: u64,
    /// (unit index or unsorted, spike time, samples).
    pub rows: Vec<(Option<u32>, u64, Vec<f32>)>,
}

impl Snippets {
    /// Up to `per_unit` spikes of each unit in `units` (evenly spread over the windows) and as
    /// many unsorted ones, on `channel`, from `windows` (read from the view's signal: the
    /// processed preview when the view has ops, what the sorter saw).
    pub fn read(sorting: &SortingData, windows: &Windows, units: &[usize], channel: usize, per_unit: usize) -> Self {
        let (before, after) = units
            .first()
            .and_then(|&u| sorting.units.get(u))
            .map_or_else(|| { let h = (DEFAULT_HALF_WIDTH_MS * 1e-3 * sorting.sample_rate) as u64; (h, h) }, |u| (u.extent.before, u.extent.after));
        let cap = per_unit.div_ceil(windows.starts.len().max(1)).max(1);
        let wanted = |u: Option<u32>| match u {
            Some(u) => units.contains(&(u as usize)),
            None => true,
        };
        let n = (before + after) as usize;
        let mut rows = Vec::new();
        if channel >= windows.channels {
            return Self { channel, before, after, rows };
        }
        for (w, &s0) in windows.starts.iter().enumerate() {
            let s1 = s0 + windows.len;
            // Spikes whose snippet lies inside the window, at most `cap` per unit, spread
            let mut by_unit: HashMap<Option<u32>, Vec<u64>> = HashMap::new();
            for k in sorting.in_range(s0 + before, s1.saturating_sub(after)) {
                if wanted(sorting.unit_of[k]) {
                    by_unit.entry(sorting.unit_of[k]).or_default().push(sorting.times[k]);
                }
            }
            let block = windows.block(w, channel);
            for (u, times) in by_unit {
                for t in spread(&times, cap) {
                    let a = (t - before - s0) as usize;
                    if a + n <= block.len() {
                        rows.push((u, t, block[a..a + n].to_vec()));
                    }
                }
            }
        }
        rows.sort_by_key(|r| (r.0.is_some(), r.1));
        Self { channel, before, after, rows }
    }

    /// First two principal components of the snippets: dspk-synapse's `extract_waveform_pca` on
    /// the device; on the host (power iteration) when no device answers.
    pub fn pca2(&self) -> Vec<[f32; 2]> {
        if let Some(points) = self.pca2_device() {
            return points;
        }
        self.pca2_host()
    }

    fn pca2_device(&self) -> Option<Vec<[f32; 2]>> {
        let n = (self.before + self.after) as usize;
        if self.rows.len() < 3 || n == 0 {
            return None;
        }
        let client = dspk_core::ComputeTarget::from_env().ok()?.client().ok()?;
        let snippets: Vec<crate::WaveformSnippet> = self
            .rows
            .iter()
            .map(|(_, t, w)| crate::WaveformSnippet { primary_channel: self.channel, center_sample: *t, subsample_offset: 0.0, channel_ids: vec![self.channel], num_samples: n, waveform: w.clone() })
            .collect();
        let (_, rows) = crate::features::extract_waveform_pca(&client, &snippets, 2)?;
        Some(rows.into_iter().map(|r| [r.first().copied().unwrap_or(0.0), r.get(1).copied().unwrap_or(0.0)]).collect())
    }

    /// Each snippet projected on two templates, `a` and `b` (two units' mean waveforms on this
    /// channel): Phy's template-feature view. Spikes of two units that are one show one cloud.
    /// A template of another length than the snippets, or all zeros, projects to 0.
    pub fn template_features(&self, a: &[f32], b: &[f32]) -> Vec<[f32; 2]> {
        let project = |row: &[f32], t: &[f32]| {
            let norm = t.iter().map(|v| v * v).sum::<f32>().sqrt();
            if t.len() != row.len() || norm <= f32::EPSILON {
                return 0.0;
            }
            row.iter().zip(t).map(|(r, t)| r * t).sum::<f32>() / norm
        };
        self.rows.iter().map(|(_, _, row)| [project(row, a), project(row, b)]).collect()
    }

    /// The host fallback (a few thousand rows of tens of samples; power iteration).
    pub fn pca2_host(&self) -> Vec<[f32; 2]> {
        match self.pca_basis() {
            Some(basis) => self.rows.iter().map(|(_, _, r)| basis.project(r)).collect(),
            None => vec![[0.0, 0.0]; self.rows.len()],
        }
    }

    /// The snippets' mean and first two principal components (host, power iteration): a basis
    /// other spikes are projected on alike (a lasso drawn on the features selects every spike of
    /// the unit by it, not only the ones drawn). `None` for fewer than three snippets.
    pub fn pca_basis(&self) -> Option<PcaBasis> {
        let d = (self.before + self.after) as usize;
        let n = self.rows.len();
        if n < 3 || d == 0 {
            return None;
        }
        let mut mean = vec![0.0f64; d];
        for (_, _, r) in &self.rows {
            for (m, &v) in mean.iter_mut().zip(r) {
                *m += v as f64 / n as f64;
            }
        }
        let mut cov = vec![0.0f64; d * d];
        for (_, _, r) in &self.rows {
            let x: Vec<f64> = r.iter().zip(&mean).map(|(&v, m)| v as f64 - m).collect();
            for i in 0..d {
                for j in i..d {
                    cov[i * d + j] += x[i] * x[j];
                }
            }
        }
        for i in 0..d {
            for j in 0..i {
                cov[i * d + j] = cov[j * d + i];
            }
        }
        let pc1 = power_iteration(&cov, d, None);
        let pc2 = power_iteration(&cov, d, Some(&pc1));
        Some(PcaBasis { mean, pc1, pc2 })
    }
}

/// Snippets' mean and first two principal components: what [`Snippets::pca_basis`] found.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PcaBasis {
    pub mean: Vec<f64>,
    pub pc1: Vec<f64>,
    pub pc2: Vec<f64>,
}

impl PcaBasis {
    /// A snippet's (PC1, PC2); 0 for a snippet of another length.
    pub fn project(&self, row: &[f32]) -> [f32; 2] {
        if row.len() != self.mean.len() {
            return [0.0, 0.0];
        }
        let x = row.iter().zip(&self.mean).map(|(&v, m)| v as f64 - m);
        let (a, b) = x.zip(self.pc1.iter().zip(&self.pc2)).fold((0.0, 0.0), |(a, b), (v, (p, q))| (a + v * p, b + v * q));
        [a as f32, b as f32]
    }
}

/// Windows of every channel of a signal, read once (see [`WINDOW_SEC`]).
pub struct Windows {
    /// First sample of each window.
    pub starts: Vec<u64>,
    /// Samples per window.
    pub len: u64,
    pub channels: usize,
    /// Per window: `channels × len` values, channel-major.
    data: Vec<Vec<f32>>,
}

impl Windows {
    /// Reads the windows of `source` (every channel; blocking).
    pub fn read(source: &dyn RecordingSource) -> Self {
        Self::read_with(source.info(), None, |channels, samples, out| source.read(channels, samples, out))
    }

    /// Reads the windows of a signal described by `info` through `read` (as
    /// [`RecordingSource::read`]: `samples` of `channels`, channel-major): for signals that are
    /// not a [`RecordingSource`], e.g. a viewer's processed preview.
    ///
    /// `span`: where the windows go (a sorting's spikes: [`SortingData::span`]); `None`, the whole
    /// recording.
    pub fn read_with(info: &RecordingInfo, span: Option<std::ops::Range<u64>>, read: impl Fn(&[usize], std::ops::Range<u64>, &mut [f32]) -> DspResult<()>) -> Self {
        let (total, channels) = (info.samples, info.channels.len());
        let span = span.map_or(0..total, |s| s.start.min(total)..s.end.min(total).max(s.start.min(total)));
        let (first, extent) = (span.start, span.end - span.start);
        let rate = info.sample_rate_hz();
        let len = ((WINDOW_SEC * rate) as u64).clamp(1, total.max(1));
        let per_window = channels.max(1) * len as usize * 4;
        // As many as fit, but no more than the span holds side by side
        let fit = (extent / len.max(1)).max(1) as usize;
        let count = (window_budget_bytes() / per_window.max(1)).clamp(MIN_WINDOWS, MAX_WINDOWS).min(fit.max(1));
        let starts: Vec<u64> = if extent <= len || count < 2 {
            vec![first.min(total.saturating_sub(len))]
        } else {
            (0..count).map(|i| first + (extent - len) * i as u64 / (count - 1) as u64).collect()
        };
        let all: Vec<usize> = (0..channels).collect();
        let mut data = Vec::with_capacity(starts.len());
        let mut kept = Vec::with_capacity(starts.len());
        for &s0 in &starts {
            let n = len.min(total - s0);
            let mut block = vec![0.0f32; channels * n as usize];
            if read(&all, s0..s0 + n, &mut block).is_ok() && n == len {
                data.push(block);
                kept.push(s0);
            }
        }
        Self { starts: kept, len, channels, data }
    }

    /// Window `w` of `channel`.
    pub fn block(&self, w: usize, channel: usize) -> &[f32] {
        let n = self.len as usize;
        &self.data[w][channel * n..(channel + 1) * n]
    }
}

/// Up to `k` of `times`, evenly spread.
fn spread(times: &[u64], k: usize) -> Vec<u64> {
    if times.len() <= k {
        return times.to_vec();
    }
    (0..k).map(|i| times[i * (times.len() - 1) / (k - 1).max(1)]).collect()
}

/// The leading eigenvector of the symmetric `d × d` matrix `m` (orthogonal to `deflate`).
fn power_iteration(m: &[f64], d: usize, deflate: Option<&[f64]>) -> Vec<f64> {
    let mut v: Vec<f64> = (0..d).map(|i| 1.0 + (i as f64 * 0.618).fract()).collect();
    for _ in 0..200 {
        if let Some(p) = deflate {
            let dot: f64 = v.iter().zip(p).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(p).for_each(|(a, b)| *a -= dot * b);
        }
        let mut w = vec![0.0; d];
        for i in 0..d {
            w[i] = (0..d).map(|j| m[i * d + j] * v[j]).sum();
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }
        v = w.into_iter().map(|x| x / norm).collect();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_and_pca() {
        assert_eq!(spread(&[1, 2, 3, 4, 5], 3), vec![1, 3, 5]);
        // Two groups of snippets along different directions separate on PC1
        let mut rows = Vec::new();
        for k in 0..20 {
            let s = if k < 10 { 1.0 } else { -1.0 };
            rows.push((None, k as u64, vec![s * 5.0 + k as f32 * 0.01, 0.0, -s * 5.0]));
        }
        let snippets = Snippets { channel: 0, before: 1, after: 2, rows };
        let pcs = snippets.pca2_host();
        assert!(pcs[..10].iter().all(|p| p[0].signum() == pcs[0][0].signum()));
        assert!(pcs[10..].iter().all(|p| p[0].signum() == -pcs[0][0].signum()));
    }

    /// Snippets shaped like template `a` lie along x, like `b` along y.
    #[test]
    fn snippets_project_on_the_templates_they_look_like() {
        let (a, b) = (vec![0.0, -2.0, 1.0], vec![1.0, 0.0, 0.0]);
        let rows = vec![(Some(0), 0, vec![0.0, -4.0, 2.0]), (Some(1), 1, vec![3.0, 0.0, 0.0])];
        let f = Snippets { channel: 0, before: 1, after: 2, rows }.template_features(&a, &b);
        assert!(f[0][0] > 4.0 && f[0][1].abs() < 1e-6, "{f:?}");
        assert!(f[1][0].abs() < 1e-6 && (f[1][1] - 3.0).abs() < 1e-6, "{f:?}");
        // A template of another length projects to 0
        let g = Snippets { channel: 0, before: 1, after: 2, rows: vec![(None, 0, vec![1.0, 1.0, 1.0])] }.template_features(&[1.0], &b);
        assert_eq!(g[0][0], 0.0);
    }
}
