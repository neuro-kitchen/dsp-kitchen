//! `.sorting.zarr` results as the views use them: read once, spikes indexed by time (a window's
//! spikes are two binary searches), each unit's template extent (the samples and channels where
//! its mean template has signal), and the unit table. Waveforms and PCA features for the sorting
//! view are computed on demand from the recording for a subset of spikes (they are not stored).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use dspk_synapse::io::sorting_zarr::{SortingZarr, SortingZarrUnit};
use dspk_view::SignalBackend;

/// A channel is in a unit's extent when its template's peak-to-peak is at least this fraction of
/// the unit's largest.
pub const EXTENT_CHANNEL_FRACTION: f32 = 0.25;
/// Most channels in a unit's extent (the largest first).
pub const EXTENT_MAX_CHANNELS: usize = 8;
/// Waveforms are read from windows spread over the recording, each this long (s), every channel
/// at once and kept for the session (one processing of each window serves every unit and
/// channel): as many as fit in [`WINDOW_BUDGET_BYTES`], between [`MIN_WINDOWS`] and
/// [`MAX_WINDOWS`].
pub const WINDOW_SEC: f64 = 1.0;
pub const WINDOW_BUDGET_BYTES: usize = 256 << 20;
pub const MIN_WINDOWS: usize = 2;
pub const MAX_WINDOWS: usize = 12;
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
    /// Its row in [`SortingData::templates`] (a merged unit: its largest part's).
    pub template_row: Option<usize>,
    /// Merged by curation into the unit at this index (the row stays, empty).
    pub merged_into: Option<usize>,
}

/// Every unit's mean template on every channel: `[units, channels, samples]`.
#[derive(Debug)]
pub struct Templates {
    pub data: Vec<f32>,
    pub channels: usize,
    pub samples: usize,
}

impl Templates {
    /// Row `u`: `channels × samples`, channel-major.
    pub fn row(&self, u: usize) -> &[f32] {
        let n = self.channels * self.samples;
        &self.data[u * n..(u + 1) * n]
    }
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
    /// Spike indices (into `times`) of each unit, in time order.
    pub by_unit: Vec<Vec<u32>>,
    /// Position (µm, x and y) of each channel on the probe, when the store has the probe.
    pub positions: Vec<Option<(f32, f32)>>,
    pub templates: Option<Arc<Templates>>,
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
                let row = store.templates_mean.as_ref().filter(|t| t.shape.len() == 3 && i < t.shape[0]).map(|_| i);
                Unit { info: u.clone(), extent, template, template_row: row, merged_into: None }
            })
            .collect();
        // Spikes sorted by time (stores are usually sorted already; a stable sort keeps ties)
        let mut order: Vec<usize> = (0..store.spike_times.len()).collect();
        order.sort_by_key(|&i| store.spike_times[i]);
        let times = order.iter().map(|&i| store.spike_times[i]).collect();
        let unit_of = order.iter().map(|&i| by_id.get(&store.spike_clusters.get(i).copied().unwrap_or(-1)).map(|&u| u as u32)).collect();
        let amplitudes = if store.amplitudes.len() == store.spike_times.len() { order.iter().map(|&i| store.amplitudes[i]).collect() } else { Vec::new() };
        let channels = store.templates_mean.as_ref().and_then(|t| t.shape.get(1).copied());
        let path = path.to_path_buf();
        let sorter = m.sorter_name.clone();
        let mut positions: Vec<Option<(f32, f32)>> = Vec::new();
        if let Some(p) = &m.probe {
            for site in p.sites() {
                if positions.len() <= site.channel_id {
                    positions.resize(site.channel_id + 1, None);
                }
                positions[site.channel_id] = Some((site.position.x_um, site.position.y_um));
            }
        }
        let templates = store.templates_mean.filter(|t| t.shape.len() == 3).map(|t| Arc::new(Templates { channels: t.shape[1], samples: t.shape[2], data: t.data }));
        let mut s = Self { path, sorter, sample_rate: rate, times, unit_of, amplitudes, units, by_id, channels, by_unit: Vec::new(), positions, templates };
        s.index_units();
        s
    }

    /// Rebuilds [`Self::by_unit`] (after spikes changed units).
    pub fn index_units(&mut self) {
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); self.units.len()];
        for (k, u) in self.unit_of.iter().enumerate() {
            if let Some(u) = u {
                if let Some(v) = by.get_mut(*u as usize) {
                    v.push(k as u32);
                }
            }
        }
        self.by_unit = by;
    }

    /// Indices of the spikes with times in `samples`.
    pub fn in_range(&self, start: u64, end: u64) -> std::ops::Range<usize> {
        let lo = self.times.partition_point(|&t| t < start);
        let hi = self.times.partition_point(|&t| t < end);
        lo..hi.max(lo)
    }

    /// Spike times of unit `unit` (an index into `units`).
    pub fn unit_times(&self, unit: usize) -> Vec<u64> {
        self.by_unit.get(unit).map_or_else(Vec::new, |ks| ks.iter().map(|&k| self.times[k as usize]).collect())
    }

    /// (time s, amplitude) of up to `max` spikes of `unit`, evenly spread.
    pub fn unit_amplitudes(&self, unit: usize, max: usize) -> Vec<(f32, f32)> {
        if self.amplitudes.is_empty() {
            return Vec::new();
        }
        let idx: Vec<usize> = self.by_unit.get(unit).map_or_else(Vec::new, |ks| ks.iter().map(|&k| k as usize).collect());
        let step = (idx.len() / max.max(1)).max(1);
        idx.iter().step_by(step).map(|&k| ((self.times[k] as f64 / self.sample_rate) as f32, self.amplitudes[k])).collect()
    }

    /// Where unit `u` sits on the probe: its extent channels' positions weighted by its
    /// template's peak-to-peak there.
    pub fn unit_position(&self, u: usize) -> Option<(f32, f32)> {
        let unit = &self.units[u];
        let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
        for (k, &c) in unit.extent.channels.iter().enumerate() {
            let Some((x, y)) = self.positions.get(c).copied().flatten() else { continue };
            let w = unit.template.get(k).map_or(1.0, |r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max) - r.iter().copied().fold(f32::INFINITY, f32::min)).max(0.0);
            sx += x * w;
            sy += y * w;
            sw += w;
        }
        (sw > 0.0).then(|| (sx / sw, sy / sw)).or_else(|| self.positions.get(unit.info.primary_channel).copied().flatten())
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
        let snippets: Vec<dspk_synapse::WaveformSnippet> = self
            .rows
            .iter()
            .map(|(_, t, w)| dspk_synapse::WaveformSnippet { primary_channel: self.channel, center_sample: *t, subsample_offset: 0.0, channel_ids: vec![self.channel], num_samples: n, waveform: w.clone() })
            .collect();
        let (_, rows) = dspk_synapse::features::extract_waveform_pca(&client, &snippets, 2)?;
        Some(rows.into_iter().map(|r| [r.first().copied().unwrap_or(0.0), r.get(1).copied().unwrap_or(0.0)]).collect())
    }

    /// The host fallback (a few thousand rows of tens of samples; power iteration).
    pub fn pca2_host(&self) -> Vec<[f32; 2]> {
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
    /// Reads the windows of `signal` (every channel; blocking).
    pub fn read(signal: &dyn SignalBackend) -> Self {
        let info = signal.info();
        let (total, channels) = (info.samples, info.channels.len());
        let rate = info.sample_rate_hz();
        let len = ((WINDOW_SEC * rate) as u64).clamp(1, total.max(1));
        let per_window = channels.max(1) * len as usize * 4;
        let count = (WINDOW_BUDGET_BYTES / per_window.max(1)).clamp(MIN_WINDOWS, MAX_WINDOWS);
        let starts: Vec<u64> = if total <= len { vec![0] } else { (0..count).map(|i| (total - len) * i as u64 / (count - 1) as u64).collect() };
        let all: Vec<usize> = (0..channels).collect();
        let mut data = Vec::with_capacity(starts.len());
        let mut kept = Vec::with_capacity(starts.len());
        for &s0 in &starts {
            let n = len.min(total - s0);
            let mut block = vec![0.0f32; channels * n as usize];
            if signal.read(&all, s0..s0 + n, &mut block).is_ok() && n == len {
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

/// The windows of `signal`, read the first time and kept (the two most recent signals: the
/// preview is a new signal after each edit of the ops).
pub fn windows_of(signal: &Arc<dyn SignalBackend>) -> Arc<Windows> {
    static CACHE: Mutex<Vec<(usize, Arc<Windows>)>> = Mutex::new(Vec::new());
    let key = Arc::as_ptr(signal) as *const () as usize;
    if let Some(w) = CACHE.lock().ok().and_then(|c| c.iter().find(|(k, _)| *k == key).map(|(_, w)| w.clone())) {
        return w;
    }
    let w = Arc::new(Windows::read(signal.as_ref()));
    if let Ok(mut c) = CACHE.lock() {
        c.retain(|(k, _)| *k != key);
        c.push((key, w.clone()));
        if c.len() > 2 {
            c.remove(0);
        }
    }
    w
}

/// Counts of inter-spike intervals of sorted `times` (samples at `rate` Hz) in bins of `bin_ms`
/// from 0 to `max_ms` (dspk-synapse's `compute_isi_histogram`).
pub fn compute_isi_histogram(times: &[u64], rate: f64, max_ms: f64, bin_ms: f64) -> Vec<u32> {
    dspk_synapse::metrics::firing::compute_isi_histogram(times, rate, bin_ms, max_ms).1.into_iter().map(|c| c.min(u32::MAX as u64) as u32).collect()
}

/// Autocorrelogram of sorted `times`: counts of lags in `[-half_ms, half_ms]` in bins of `bin_ms`
/// (dspk-synapse's `compute_autocorrelogram`; the middle bin is zero lag).
pub fn autocorrelogram(times: &[u64], rate: f64, half_ms: f64, bin_ms: f64) -> Vec<u32> {
    counts(dspk_synapse::metrics::correlogram::compute_autocorrelogram(times, rate, bin_ms as f32, half_ms as f32).counts)
}

fn counts(c: Vec<u64>) -> Vec<u32> {
    c.into_iter().map(|v| v.min(u32::MAX as u64) as u32).collect()
}

/// Cross-correlogram of sorted trains `a` and `b` (lags `b − a`; dspk-synapse's
/// `compute_crosscorrelogram`).
pub fn crosscorrelogram(a: &[u64], b: &[u64], rate: f64, half_ms: f64, bin_ms: f64) -> Vec<u32> {
    counts(dspk_synapse::metrics::correlogram::compute_crosscorrelogram(a, b, rate, bin_ms as f32, half_ms as f32).counts)
}

/// Spikes within this much of each other (ms) count as the same spike when comparing two runs.
pub const MATCH_TOLERANCE_MS: f64 = 0.4;

/// Each unit of `a` (alive) with its best match in `b`: (unit of `a`, unit of `b`, agreement =
/// matched / (n_a + n_b − matched)), best first. Spikes match when within `tolerance_ms`.
pub fn compare(a: &SortingData, b: &SortingData, tolerance_ms: f64) -> Vec<(usize, usize, f32)> {
    let tol = (tolerance_ms * 1e-3 * a.sample_rate).round() as u64;
    let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
    for (k, &t) in a.times.iter().enumerate() {
        let Some(ua) = a.unit_of[k] else { continue };
        // Each spike of `a` counts once per unit of `b` near it
        let mut seen: Vec<u32> = Vec::new();
        for j in b.in_range(t.saturating_sub(tol), t + tol + 1) {
            if let Some(ub) = b.unit_of[j] {
                if !seen.contains(&ub) {
                    seen.push(ub);
                    *pairs.entry((ua, ub)).or_default() += 1;
                }
            }
        }
    }
    let n = |s: &SortingData, u: usize| s.by_unit.get(u).map_or(0, Vec::len) as f32;
    let mut best: HashMap<u32, (u32, f32)> = HashMap::new();
    for (&(ua, ub), &m) in &pairs {
        let score = m as f32 / (n(a, ua as usize) + n(b, ub as usize) - m as f32).max(1.0);
        let slot = best.entry(ua).or_insert((ub, 0.0));
        if score > slot.1 {
            *slot = (ub, score);
        }
    }
    let mut out: Vec<(usize, usize, f32)> = a.alive().filter_map(|u| best.get(&(u as u32)).map(|&(v, sc)| (u, v as usize, sc))).collect();
    out.sort_by(|x, y| y.2.total_cmp(&x.2));
    out
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
                template_row: None,
                merged_into: None,
            })
            .collect();
        let by_id = units.iter().enumerate().map(|(i, u)| (u.info.unit_id as i32, i)).collect();
        let mut s = SortingData { path: "/x.sorting.zarr".into(), sorter: "test".into(), sample_rate: 1000.0, times, unit_of, amplitudes: Vec::new(), units, by_id, channels: None, by_unit: Vec::new(), positions: Vec::new(), templates: None };
        s.index_units();
        s
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
        let isi = compute_isi_histogram(&times, 1000.0, 50.0, 1.0);
        assert_eq!(isi[10], 99);
        let acg = autocorrelogram(&times, 1000.0, 50.0, 1.0);
        assert_eq!(acg.len(), 101, "±50 bins and zero lag");
        assert_eq!(acg[60], 99, "+10 ms");
        assert_eq!(acg[40], 99, "-10 ms");
        assert_eq!(acg[50], 0, "no zero lag");
        let mut ccg = crosscorrelogram(&times, &times, 1000.0, 50.0, 1.0);
        ccg[50] = 0;
        assert_eq!(ccg, acg, "a train with itself is its autocorrelogram (but for each spike with itself)");
    }

    #[test]
    fn test_compare_matches_shared_spikes() {
        let a = sorting_of_units(&[(1, 0, 4), (2, 0, 2)]);
        // b: the same spikes, the first unit split differently
        let b = sorting_of_units(&[(7, 0, 4), (8, 0, 2)]);
        let m = compare(&a, &b, 0.4);
        assert_eq!(m[0], (0, 0, 1.0));
        assert_eq!(m[1], (1, 1, 1.0));
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
        let pcs = snippets.pca2_host();
        assert!(pcs[..10].iter().all(|p| p[0].signum() == pcs[0][0].signum()));
        assert!(pcs[10..].iter().all(|p| p[0].signum() == -pcs[0][0].signum()));
    }
}
