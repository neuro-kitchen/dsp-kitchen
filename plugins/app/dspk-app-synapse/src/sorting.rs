//! `.sorting.zarr` results as the views use them: read once, spikes indexed by time (a window's
//! spikes are two binary searches), each unit's template extent (the samples and channels where
//! its mean template has signal), and the unit table. Waveforms and PCA features for the sorting
//! view are computed on demand from the recording for a subset of spikes (they are not stored).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dspk_synapse::io::sorting_zarr::{SortingZarr, SortingZarrUnit};
use dspk_view::SignalBackend;

/// A channel is in a unit's extent when its template's peak-to-peak is at least this fraction of
/// the unit's largest.
pub const EXTENT_CHANNEL_FRACTION: f32 = 0.25;
/// Most channels in a unit's extent (the largest first).
pub const EXTENT_MAX_CHANNELS: usize = 8;
/// Waveforms are read from this many windows spread over the recording, each this long (s).
pub const SNIPPET_WINDOWS: usize = 8;
pub const SNIPPET_WINDOW_SEC: f64 = 2.0;
/// Half-width of the extent when a store has no templates (ms).
pub const DEFAULT_HALF_WIDTH_MS: f64 = 1.0;

/// Where a unit's spikes are highlighted: samples before / after the spike time, and channels.
#[derive(Debug, Clone, PartialEq)]
pub struct Extent {
    pub before: u64,
    pub after: u64,
    /// Channels of the recording, largest template amplitude first.
    pub channels: Vec<usize>,
}

/// One unit: its manifest row, its extent, and its mean template on its extent channels.
#[derive(Debug, Clone)]
pub struct Unit {
    pub info: SortingZarrUnit,
    pub extent: Extent,
    /// Mean template on `extent.channels` (rows), `before + after` samples each; empty without
    /// templates.
    pub template: Vec<Vec<f32>>,
    /// Merged by curation into the unit at this index (the row stays, empty).
    pub merged_into: Option<usize>,
}

/// The app's notes on a store, inside its folder (`dspk-app.json`): where in the recording a
/// sorting of a part of it starts (its spike times count from there).
pub const APP_NOTES_FILE: &str = "dspk-app.json";

/// A loaded sorting.
#[derive(Clone)]
pub struct SortingData {
    pub path: PathBuf,
    pub sorter: String,
    pub sample_rate: f64,
    /// Spike times (samples), ascending, and the unit index (into `units`) of each; spikes of
    /// clusters the manifest does not list are `None` (drawn as unsorted).
    pub times: Vec<u64>,
    pub unit_of: Vec<Option<u32>>,
    /// Amplitude of each spike (same order; empty when the store has none).
    pub amplitudes: Vec<f32>,
    pub units: Vec<Unit>,
    /// Unit index by unit id.
    pub by_id: HashMap<i32, usize>,
    /// Channels of the recording it was sorted on (templates' rows), when known: a view that
    /// reads a channel subset maps the sorting's channel `i` to its `i`-th channel.
    pub channels: Option<usize>,
}

impl SortingData {
    pub fn load(path: &Path) -> Result<Self> {
        let store = SortingZarr::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
        let mut data = Self::from_store(path, store);
        // A sorting of a part of the recording: its times count from the part's start
        let offset = start_sec(path).map_or(0, |s| (s * data.sample_rate).round().max(0.0) as u64);
        if offset > 0 {
            data.times.iter_mut().for_each(|t| *t += offset);
        }
        Ok(data)
    }

    /// Units not merged away, by index.
    pub fn alive(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.units.len()).filter(|&u| self.units[u].merged_into.is_none())
    }

    pub fn from_store(path: &Path, store: SortingZarr) -> Self {
        let m = &store.manifest;
        let rate = m.sample_rate_hz;
        let by_id: HashMap<i32, usize> = m.units.iter().enumerate().map(|(i, u)| (u.unit_id as i32, i)).collect();
        let half = (DEFAULT_HALF_WIDTH_MS * 1e-3 * rate).round() as u64;
        let units = m
            .units
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let (extent, template) = match &store.templates_mean {
                    Some(t) if t.shape.len() == 3 && i < t.shape[0] => template_extent(&t.data, t.shape[1], t.shape[2], i, u.primary_channel),
                    _ => (Extent { before: half, after: half, channels: vec![u.primary_channel] }, Vec::new()),
                };
                Unit { info: u.clone(), extent, template, merged_into: None }
            })
            .collect();
        // Spikes sorted by time (stores are usually sorted already; a stable sort keeps ties)
        let mut order: Vec<usize> = (0..store.spike_times.len()).collect();
        order.sort_by_key(|&i| store.spike_times[i]);
        let times = order.iter().map(|&i| store.spike_times[i]).collect();
        let unit_of = order.iter().map(|&i| by_id.get(&store.spike_clusters.get(i).copied().unwrap_or(-1)).map(|&u| u as u32)).collect();
        let amplitudes = if store.amplitudes.len() == store.spike_times.len() { order.iter().map(|&i| store.amplitudes[i]).collect() } else { Vec::new() };
        let channels = store.templates_mean.as_ref().and_then(|t| t.shape.get(1).copied());
        Self { path: path.to_path_buf(), sorter: m.sorter_name.clone(), sample_rate: rate, times, unit_of, amplitudes, units, by_id, channels }
    }

    /// Indices of the spikes with times in `samples`.
    pub fn in_range(&self, start: u64, end: u64) -> std::ops::Range<usize> {
        let lo = self.times.partition_point(|&t| t < start);
        let hi = self.times.partition_point(|&t| t < end);
        lo..hi.max(lo)
    }

    /// Spike times of unit `unit` (an index into `units`).
    pub fn unit_times(&self, unit: usize) -> Vec<u64> {
        self.times.iter().zip(&self.unit_of).filter(|(_, u)| **u == Some(unit as u32)).map(|(&t, _)| t).collect()
    }

    /// (time s, amplitude) of up to `max` spikes of `unit`, evenly spread.
    pub fn unit_amplitudes(&self, unit: usize, max: usize) -> Vec<(f32, f32)> {
        if self.amplitudes.is_empty() {
            return Vec::new();
        }
        let idx: Vec<usize> = (0..self.times.len()).filter(|&k| self.unit_of[k] == Some(unit as u32)).collect();
        let step = (idx.len() / max.max(1)).max(1);
        idx.iter().step_by(step).map(|&k| ((self.times[k] as f64 / self.sample_rate) as f32, self.amplitudes[k])).collect()
    }

    /// `kilosort4 · 287 units · 133 544 spikes`
    pub fn summary(&self) -> String {
        format!("{} · {} units · {} spikes", self.sorter, self.alive().count(), self.times.len())
    }
}

/// Where the sorting at `store` starts in its recording (s), from the app's notes.
pub fn start_sec(store: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(store.join(APP_NOTES_FILE)).ok()?;
    serde_json::from_str::<serde_json::Value>(&text).ok()?.get("start_sec")?.as_f64()
}

/// Notes that the sorting at `store` starts `start_sec` into its recording.
pub fn write_start_sec(store: &Path, start_sec: f64) -> std::io::Result<()> {
    std::fs::write(store.join(APP_NOTES_FILE), serde_json::json!({ "start_sec": start_sec }).to_string())
}

/// Unit `u`'s extent from the mean templates `[units, channels, samples]`: its alignment (the
/// sample of the largest |value| on its primary channel is the spike time, as sorters align
/// spikes on their peak) and the channels with at least [`EXTENT_CHANNEL_FRACTION`] of its
/// largest peak-to-peak, at most [`EXTENT_MAX_CHANNELS`].
fn template_extent(data: &[f32], channels: usize, samples: usize, u: usize, primary: usize) -> (Extent, Vec<Vec<f32>>) {
    let unit = &data[u * channels * samples..(u + 1) * channels * samples];
    let row = |c: usize| &unit[c * samples..(c + 1) * samples];
    let ptp = |c: usize| {
        let r = row(c);
        r.iter().copied().fold(f32::NEG_INFINITY, f32::max) - r.iter().copied().fold(f32::INFINITY, f32::min)
    };
    let mut by_ptp: Vec<(usize, f32)> = (0..channels).map(|c| (c, ptp(c))).filter(|(_, p)| p.is_finite()).collect();
    by_ptp.sort_by(|a, b| b.1.total_cmp(&a.1));
    let largest = by_ptp.first().map_or(0.0, |&(_, p)| p);
    let mut picked: Vec<usize> = by_ptp.iter().take_while(|&&(_, p)| p >= EXTENT_CHANNEL_FRACTION * largest).take(EXTENT_MAX_CHANNELS).map(|&(c, _)| c).collect();
    let primary = if primary < channels { primary } else { picked.first().copied().unwrap_or(0) };
    if !picked.contains(&primary) {
        picked.insert(0, primary);
    }
    let peak = row(primary).iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).map_or(samples / 2, |(i, _)| i);
    let template = picked.iter().map(|&c| row(c).to_vec()).collect();
    (Extent { before: peak as u64, after: (samples - peak) as u64, channels: picked }, template)
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
    /// Up to `per_unit` spikes of each unit in `units` (evenly spread over the recording) and as
    /// many unsorted ones, on `channel`, read from `signal` (the processed preview when the view
    /// has ops: what the sorter saw).
    pub fn read(sorting: &SortingData, signal: &dyn SignalBackend, units: &[usize], channel: usize, per_unit: usize) -> Self {
        let (before, after) = units
            .first()
            .and_then(|&u| sorting.units.get(u))
            .map_or_else(|| { let h = (DEFAULT_HALF_WIDTH_MS * 1e-3 * sorting.sample_rate) as u64; (h, h) }, |u| (u.extent.before, u.extent.after));
        let total = signal.info().samples;
        // A few windows spread over the recording, each read once for every unit: one processed
        // block serves all the spikes in it (a spike at a time would process the whole
        // channel group and its filter halo per spike)
        let w = ((SNIPPET_WINDOW_SEC * sorting.sample_rate) as u64).min(total);
        let starts: Vec<u64> = if total <= w { vec![0] } else { (0..SNIPPET_WINDOWS).map(|i| (total - w) * i as u64 / (SNIPPET_WINDOWS - 1) as u64).collect() };
        let cap = per_unit.div_ceil(starts.len()).max(1);
        let wanted = |u: Option<u32>| match u {
            Some(u) => units.contains(&(u as usize)),
            None => true,
        };
        let n = (before + after) as usize;
        let mut rows = Vec::new();
        let mut block = vec![0.0f32; w as usize];
        for &s0 in &starts {
            let s1 = (s0 + w).min(total);
            // Spikes whose snippet lies inside the window, at most `cap` per unit, spread
            let mut by_unit: std::collections::HashMap<Option<u32>, Vec<u64>> = std::collections::HashMap::new();
            for k in sorting.in_range(s0 + before, s1.saturating_sub(after)) {
                if wanted(sorting.unit_of[k]) {
                    by_unit.entry(sorting.unit_of[k]).or_default().push(sorting.times[k]);
                }
            }
            if by_unit.is_empty() {
                continue;
            }
            let len = (s1 - s0) as usize;
            if signal.read(&[channel], s0..s1, &mut block[..len]).is_err() {
                continue;
            }
            for (u, times) in by_unit {
                for t in spread(&times, cap) {
                    let a = (t - before - s0) as usize;
                    rows.push((u, t, block[a..a + n].to_vec()));
                }
            }
        }
        rows.sort_by_key(|r| (r.0.is_some(), r.1));
        Self { channel, before, after, rows }
    }

    /// First two principal components of the snippets (host: a few thousand rows of tens of
    /// samples; power iteration on the covariance).
    pub fn pca2(&self) -> Vec<[f32; 2]> {
        let d = (self.before + self.after) as usize;
        let n = self.rows.len();
        if n < 3 || d == 0 {
            return vec![[0.0, 0.0]; n];
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
        self.rows
            .iter()
            .map(|(_, _, r)| {
                let x = r.iter().zip(&mean).map(|(&v, m)| v as f64 - m);
                let (a, b) = x.zip(pc1.iter().zip(&pc2)).fold((0.0, 0.0), |(a, b), (v, (p, q))| (a + v * p, b + v * q));
                [a as f32, b as f32]
            })
            .collect()
    }
}

/// Counts of inter-spike intervals of sorted `times` (samples at `rate` Hz) in bins of `bin_ms`
/// from 0 to `max_ms`.
pub fn isi_histogram(times: &[u64], rate: f64, max_ms: f64, bin_ms: f64) -> Vec<u32> {
    let bins = (max_ms / bin_ms).round() as usize;
    let mut h = vec![0u32; bins];
    for w in times.windows(2) {
        let ms = (w[1] - w[0]) as f64 / rate * 1e3;
        let b = (ms / bin_ms) as usize;
        if b < bins {
            h[b] += 1;
        }
    }
    h
}

/// Autocorrelogram of sorted `times`: counts of lags in `[-half_ms, half_ms)` (zero lag left
/// out) in bins of `bin_ms`; two pointers, so linear in spikes × spikes within the window.
pub fn autocorrelogram(times: &[u64], rate: f64, half_ms: f64, bin_ms: f64) -> Vec<u32> {
    let bins = (2.0 * half_ms / bin_ms).round() as usize;
    let mut h = vec![0u32; bins];
    let half = (half_ms * 1e-3 * rate) as u64;
    let mut lo = 0;
    for (i, &t) in times.iter().enumerate() {
        while times[lo] + half < t {
            lo += 1;
        }
        for &u in &times[lo..i] {
            // u < t: a negative lag and its mirror
            let ms = (t - u) as f64 / rate * 1e3;
            let (neg, pos) = (((half_ms - ms) / bin_ms) as isize, ((half_ms + ms) / bin_ms) as isize);
            if neg >= 0 && (neg as usize) < bins {
                h[neg as usize] += 1;
            }
            if pos >= 0 && (pos as usize) < bins {
                h[pos as usize] += 1;
            }
        }
    }
    h
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
pub(crate) mod tests {
    use super::*;
    use dspk_synapse::io::sorting_zarr::SortingZarrUnit;

    /// A sorting of units `(id, primary channel, spikes)`, spikes 10 samples apart in unit order,
    /// templates of 4 samples on the primary channel.
    pub(crate) fn sorting_of_units(units: &[(usize, usize, usize)]) -> SortingData {
        let mut times = Vec::new();
        let mut unit_of = Vec::new();
        let mut t = 0;
        for (i, &(_, _, n)) in units.iter().enumerate() {
            for _ in 0..n {
                times.push(t);
                unit_of.push(Some(i as u32));
                t += 10;
            }
        }
        let units: Vec<Unit> = units
            .iter()
            .map(|&(id, ch, n)| Unit {
                info: SortingZarrUnit {
                    unit_id: id,
                    primary_channel: ch,
                    quality_label: "mua".into(),
                    snr: 5.0,
                    firing_rate_hz: 1.0,
                    isi_violation_ratio: 0.0,
                    presence_ratio: 1.0,
                    amplitude_cutoff: 0.0,
                    composite_score: f64::NAN,
                    num_spikes: n,
                },
                extent: Extent { before: 1, after: 3, channels: vec![ch] },
                template: vec![vec![0.0, -1.0 - id as f32 * 0.1, 0.5, 0.0]],
                merged_into: None,
            })
            .collect();
        let by_id = units.iter().enumerate().map(|(i, u)| (u.info.unit_id as i32, i)).collect();
        SortingData { path: "/x.sorting.zarr".into(), sorter: "test".into(), sample_rate: 1000.0, times, unit_of, amplitudes: Vec::new(), units, by_id, channels: None }
    }

    #[test]
    fn test_extent_follows_the_template() {
        // 2 units × 3 channels × 5 samples; unit 0 peaks (negative) at sample 1 of channel 2
        let mut t = vec![0.0f32; 2 * 3 * 5];
        t[2 * 5 + 1] = -10.0;
        t[5 + 1] = -4.0; // channel 1: 40 % of the peak-to-peak: in
        t[1] = -1.0; // channel 0: 10 %: out
        let (e, tpl) = template_extent(&t, 3, 5, 0, 2);
        assert_eq!(e, Extent { before: 1, after: 4, channels: vec![2, 1] });
        assert_eq!(tpl.len(), 2);
    }

    #[test]
    fn test_isi_and_acg() {
        // Spikes every 10 ms at 1 kHz
        let times: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let isi = isi_histogram(&times, 1000.0, 50.0, 1.0);
        assert_eq!(isi[10], 99);
        let acg = autocorrelogram(&times, 1000.0, 50.0, 1.0);
        assert_eq!(acg.len(), 100);
        assert_eq!(acg[60], 99, "+10 ms");
        assert_eq!(acg[40], 99, "-10 ms");
        assert_eq!(acg[50], 0, "no zero lag");
    }

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
        let pcs = snippets.pca2();
        assert!(pcs[..10].iter().all(|p| p[0].signum() == pcs[0][0].signum()));
        assert!(pcs[10..].iter().all(|p| p[0].signum() == -pcs[0][0].signum()));
    }
}
