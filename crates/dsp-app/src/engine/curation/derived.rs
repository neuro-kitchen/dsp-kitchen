//! The data each curation view draws, for the selected clusters (phy's views).
//!
//! Spike-train statistics come from `dsp-synapse` (correlograms, ISI, firing rate, PCA, snippet
//! reads), histograms and min / max from `dsp-base`. Waveforms, raw amplitudes and computed
//! features need the recording; without it they are empty (views say so) instead of made up.
//! Each view samples a bounded number of spikes per cluster, spread over the recording.

use std::collections::{BTreeMap, BTreeSet};

use dsp_base::math::{bin_centers, histogram};
use dsp_base::resampler::minmax::peak_to_peak;
use dsp_core::RecordingSource;
use dsp_synapse::{compute_autocorrelogram, compute_crosscorrelogram, compute_instantaneous_firing_rate, extract_waveform_pca, isi_histogram, read_snippets, Correlogram, WaveformSnippet};

use super::{ClusterId, SortingData};

/// Waveforms of one cluster on its best channels.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterWaveforms {
    pub cluster_id: ClusterId,
    /// Sorted-channel indices, best first, and their positions (µm).
    pub channels: Vec<usize>,
    pub positions: Vec<[f32; 2]>,
    pub num_samples: usize,
    /// Spikes read from the recording, each `[channels × num_samples]` (empty without one).
    pub sampled: Vec<Vec<f32>>,
    /// Mean of `sampled` (empty without a recording).
    pub mean: Vec<f32>,
    /// The sorter's template on the same channels (empty without templates).
    pub template: Vec<f32>,
}

/// A spike in a scatter plot, with its index in the spike arrays (for lasso splits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikePoint {
    pub spike_index: usize,
    pub cluster_id: ClusterId,
    pub x: f32,
    pub y: f32,
}

/// Feature grid on the best channels of the first selected cluster: cell `(r, c)` plots the
/// first principal component on channel `c` (x) against channel `r` (y); the diagonal plots it
/// against time. `(background, selected)` points per cell; empty when features are unavailable.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureGridData {
    pub channels: Vec<usize>,
    pub cells: Vec<Vec<(Vec<SpikePoint>, Vec<SpikePoint>)>>,
    /// `pc_features.npy`, or principal components of waveforms read from the recording.
    pub source: FeatureSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSource {
    Sorter,
    Computed,
    /// Neither `pc_features.npy` nor a recording: nothing to plot.
    Unavailable,
}

/// Auto- (diagonal) and cross-correlograms of up to 20 selected clusters.
#[derive(Debug, Clone)]
pub struct CorrelogramMatrix {
    pub clusters: Vec<ClusterId>,
    pub bin_ms: f32,
    pub window_ms: f32,
    pub refractory_ms: f32,
    pub cells: Vec<Vec<Correlogram>>,
}

/// What the Amplitudes view plots on y.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmplitudeMode {
    /// `amplitudes.npy` (the sorter's template scaling).
    Template,
    /// Peak-to-peak of the spike on its cluster's best channel, read from the recording (µV).
    Raw,
    /// First principal component on the best channel.
    Feature,
}

impl AmplitudeMode {
    pub const ALL: [AmplitudeMode; 3] = [AmplitudeMode::Template, AmplitudeMode::Raw, AmplitudeMode::Feature];

    pub fn label(self) -> &'static str {
        match self {
            Self::Template => "Template amplitude",
            Self::Raw => "Raw peak-to-peak (µV)",
            Self::Feature => "First principal component",
        }
    }
}

/// Amplitude of spikes over time, and a histogram per selected cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct AmplitudePlotData {
    pub mode: AmplitudeMode,
    pub duration_sec: f64,
    pub y_min: f32,
    pub y_max: f32,
    /// Spikes of other clusters (grey), then the selected clusters' spikes.
    pub background: Vec<SpikePoint>,
    pub points: Vec<SpikePoint>,
    /// `(cluster, bin centres, counts)` per selected cluster.
    pub histograms: Vec<(ClusterId, Vec<f32>, Vec<u64>)>,
}

/// Up to `n` items of `items`, evenly spread.
fn spread<T: Copy>(items: &[T], n: usize) -> Vec<T> {
    let step = (items.len() / n.max(1)).max(1);
    items.iter().copied().step_by(step).take(n).collect()
}

/// Samples before the centre of a window of `len` (phy centres the trough at a third).
fn pre_samples(len: usize) -> usize {
    len / 2
}

/// The first principal component of each snippet (one value per snippet), `None` when fewer than
/// two snippets.
fn first_pc(snippets: &[WaveformSnippet]) -> Option<Vec<f32>> {
    if snippets.len() < 2 {
        return None;
    }
    let (_, projected) = extract_waveform_pca(snippets, 1)?;
    Some(projected.into_iter().map(|p| p.first().copied().unwrap_or(0.0)).collect())
}

impl SortingData {
    /// Waveform window length: the templates', else 61 samples (2 ms at 30 kHz).
    fn window_samples(&self) -> usize {
        self.sorting.templates.as_ref().map_or(61, |t| t.samples)
    }

    /// Recording channel of sorted channel `ch`.
    fn recording_channel(&self, ch: usize) -> usize {
        self.sorting.channel_map.get(ch).copied().unwrap_or(ch)
    }

    /// Up to `max_spikes` waveforms of `cid` on its `max_channels` best channels, their mean and
    /// the sorter's template there.
    pub fn compute_waveforms(&self, cid: ClusterId, recording: Option<&dyn RecordingSource>, max_spikes: usize, max_channels: usize) -> ClusterWaveforms {
        let s = &self.sorting;
        let channels = self.top_channels(cid, max_channels);
        let positions = channels.iter().map(|&c| s.channel_positions.get(c).copied().unwrap_or([0.0, c as f32])).collect();
        let num_samples = self.window_samples();
        let template = match (self.representative_template(cid), s.templates.as_ref()) {
            (Some(t), Some(tm)) => channels.iter().flat_map(|&c| tm.trace(t, c)).collect(),
            _ => Vec::new(),
        };
        let (mut sampled, mut mean) = (Vec::new(), Vec::new());
        if let Some(rec) = recording {
            let centers: Vec<u64> = spread(&self.spike_indices(cid), max_spikes).into_iter().map(|i| s.spike_times[i]).collect();
            let rec_channels: Vec<usize> = channels.iter().map(|&c| self.recording_channel(c)).collect();
            let pre = pre_samples(num_samples);
            sampled = read_snippets(rec, &centers, &rec_channels, pre, num_samples - pre, true).into_iter().map(|w| w.waveform).collect();
            if !sampled.is_empty() {
                mean = vec![0.0f32; channels.len() * num_samples];
                for w in &sampled {
                    mean.iter_mut().zip(w).for_each(|(m, v)| *m += v / sampled.len() as f32);
                }
            }
        }
        ClusterWaveforms { cluster_id: cid, channels, positions, num_samples, sampled, mean, template }
    }

    /// First principal component of spikes `indices` on sorted channel `ch`: from
    /// `pc_features.npy` when the sorter saved it for that channel, else computed from waveforms
    /// read from the recording. `None` when neither is possible.
    fn pc1(&self, indices: &[usize], ch: usize, recording: Option<&dyn RecordingSource>) -> Option<(Vec<f32>, FeatureSource)> {
        let s = &self.sorting;
        if let (Some((pcs, [n_spikes, n_pcs, n_cols])), Some((ind, [n_templates, ind_cols]))) = (&s.pc_features, &s.pc_feature_ind) {
            let column = |spike: usize| {
                let t = s.spike_templates.get(spike).map(|&t| t as usize)?;
                (t < *n_templates).then(|| (0..*ind_cols).position(|k| ind[t * ind_cols + k] == ch))?
            };
            // Spikes whose template has no column for this channel sit at 0, as in phy
            let values = indices.iter().map(|&i| if i < *n_spikes && *n_pcs > 0 { column(i).map_or(0.0, |c| pcs[(i * n_pcs) * n_cols + c]) } else { 0.0 }).collect();
            return Some((values, FeatureSource::Sorter));
        }
        let rec = recording?;
        let len = self.window_samples();
        let pre = pre_samples(len);
        let centers: Vec<u64> = indices.iter().map(|&i| s.spike_times[i]).collect();
        let snippets = read_snippets(rec, &centers, &[self.recording_channel(ch)], pre, len - pre, true);
        // Snippets the recording could not give (at its edges) are dropped by the read: keep the
        // others aligned with their spikes
        let read: BTreeMap<u64, usize> = snippets.iter().enumerate().map(|(k, w)| (w.center_sample, k)).collect();
        let pcs = first_pc(&snippets)?;
        Some((centers.iter().map(|c| read.get(c).map_or(0.0, |&k| pcs[k])).collect(), FeatureSource::Computed))
    }

    /// The feature grid for `selected` (up to 300 spikes of each, 300 background spikes).
    pub fn compute_features(&self, selected: &[ClusterId], recording: Option<&dyn RecordingSource>) -> FeatureGridData {
        let s = &self.sorting;
        let Some(&primary) = selected.first() else {
            return FeatureGridData { channels: Vec::new(), cells: Vec::new(), source: FeatureSource::Unavailable };
        };
        let channels = self.top_channels(primary, 4);
        let chosen: BTreeSet<ClusterId> = selected.iter().copied().collect();
        let mut spikes: Vec<usize> = selected.iter().flat_map(|&c| spread(&self.spike_indices(c), 300)).collect();
        let background: Vec<usize> = spread(&(0..s.spike_times.len()).filter(|&i| !chosen.contains(&s.spike_clusters[i])).collect::<Vec<_>>(), 300);
        let n_background = background.len();
        spikes.splice(0..0, background);

        let mut per_channel = Vec::with_capacity(channels.len());
        let mut source = FeatureSource::Unavailable;
        for &ch in &channels {
            match self.pc1(&spikes, ch, recording) {
                Some((values, from)) => {
                    source = from;
                    per_channel.push(values);
                }
                None => return FeatureGridData { channels, cells: Vec::new(), source: FeatureSource::Unavailable },
            }
        }
        let time = |i: usize| (s.spike_times[i] as f64 / self.sample_rate()) as f32;
        let g = channels.len();
        let cells = (0..g)
            .map(|r| {
                (0..g)
                    .map(|c| {
                        let point = |k: usize| {
                            let i = spikes[k];
                            let x = if r == c { time(i) } else { per_channel[c][k] };
                            SpikePoint { spike_index: i, cluster_id: s.spike_clusters[i], x, y: per_channel[r][k] }
                        };
                        ((0..n_background).map(point).collect(), (n_background..spikes.len()).map(point).collect())
                    })
                    .collect()
            })
            .collect();
        FeatureGridData { channels, cells, source }
    }

    /// Auto- and cross-correlograms of up to 20 of `selected`.
    pub fn compute_correlograms(&self, selected: &[ClusterId], bin_ms: f32, window_ms: f32, refractory_ms: f32) -> CorrelogramMatrix {
        let clusters: Vec<ClusterId> = selected.iter().copied().take(20).collect();
        let trains: Vec<Vec<u64>> = clusters.iter().map(|&c| self.spike_samples(c)).collect();
        let rate = self.sample_rate();
        let cells = (0..clusters.len())
            .map(|i| {
                (0..clusters.len())
                    .map(|j| if i == j { compute_autocorrelogram(&trains[i], rate, bin_ms, window_ms) } else { compute_crosscorrelogram(&trains[i], &trains[j], rate, bin_ms, window_ms) })
                    .collect()
            })
            .collect();
        CorrelogramMatrix { clusters, bin_ms, window_ms, refractory_ms, cells }
    }

    /// Which amplitude modes can be shown (template amplitudes need `amplitudes.npy`; raw ones the
    /// recording; features `pc_features.npy` or the recording).
    pub fn amplitude_modes(&self, has_recording: bool) -> Vec<AmplitudeMode> {
        AmplitudeMode::ALL
            .into_iter()
            .filter(|m| match m {
                AmplitudeMode::Template => !self.sorting.amplitudes.is_empty(),
                AmplitudeMode::Raw => has_recording,
                AmplitudeMode::Feature => has_recording || self.sorting.pc_features.is_some(),
            })
            .collect()
    }

    /// Amplitude of the spikes `indices` in `mode` (on `ch` for raw and feature amplitudes);
    /// `None` when the mode is unavailable.
    fn amplitudes_of(&self, indices: &[usize], ch: usize, mode: AmplitudeMode, recording: Option<&dyn RecordingSource>) -> Option<Vec<f32>> {
        let s = &self.sorting;
        match mode {
            AmplitudeMode::Template => (!s.amplitudes.is_empty()).then(|| indices.iter().map(|&i| s.amplitudes[i]).collect()),
            AmplitudeMode::Feature => self.pc1(indices, ch, recording).map(|(v, _)| v),
            AmplitudeMode::Raw => {
                let rec = recording?;
                let len = self.window_samples();
                let pre = pre_samples(len);
                let centers: Vec<u64> = indices.iter().map(|&i| s.spike_times[i]).collect();
                let read: BTreeMap<u64, f32> = read_snippets(rec, &centers, &[self.recording_channel(ch)], pre, len - pre, true).into_iter().map(|w| (w.center_sample, peak_to_peak(&w.waveform))).collect();
                Some(centers.iter().map(|c| read.get(c).copied().unwrap_or(f32::NAN)).collect())
            }
        }
    }

    /// Amplitudes over time of `selected` (up to `per_cluster` spikes each) over 500 background
    /// spikes, with 32-bin histograms; `None` when `mode` is unavailable.
    pub fn compute_amplitudes(&self, selected: &[ClusterId], mode: AmplitudeMode, per_cluster: usize, recording: Option<&dyn RecordingSource>) -> Option<AmplitudePlotData> {
        let s = &self.sorting;
        let ch = selected.first().map_or(0, |&c| self.best_channel(c));
        let chosen: BTreeSet<ClusterId> = selected.iter().copied().collect();
        let time = |i: usize| (s.spike_times[i] as f64 / self.sample_rate()) as f32;
        let points_of = |indices: &[usize]| -> Option<Vec<SpikePoint>> {
            let values = self.amplitudes_of(indices, ch, mode, recording)?;
            Some(indices.iter().zip(values).filter(|(_, y)| y.is_finite()).map(|(&i, y)| SpikePoint { spike_index: i, cluster_id: s.spike_clusters[i], x: time(i), y }).collect())
        };
        let background = points_of(&spread(&(0..s.spike_times.len()).filter(|&i| !chosen.contains(&s.spike_clusters[i])).collect::<Vec<_>>(), 500))?;
        let mut points = Vec::new();
        for &c in selected {
            points.extend(points_of(&spread(&self.spike_indices(c), per_cluster))?);
        }
        let (mut y_min, mut y_max) = points.iter().chain(&background).fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
        if !(y_min.is_finite() && y_max.is_finite()) {
            (y_min, y_max) = (0.0, 1.0);
        } else if y_max - y_min < 1e-5 {
            (y_min, y_max) = (y_min - 1.0, y_max + 1.0);
        }
        const BINS: usize = 32;
        let centers: Vec<f32> = bin_centers(y_min as f64, y_max as f64, BINS).into_iter().map(|c| c as f32).collect();
        let histograms = selected
            .iter()
            .map(|&c| (c, centers.clone(), histogram(points.iter().filter(|p| p.cluster_id == c).map(|p| p.y as f64), y_min as f64, y_max as f64, BINS)))
            .collect();
        Some(AmplitudePlotData { mode, duration_sec: self.duration_sec, y_min, y_max, background, points, histograms })
    }

    /// Inter-spike-interval histograms over `[0, max_ms)`: `(cluster, bin centres ms, counts)`.
    pub fn compute_isi(&self, selected: &[ClusterId], bin_ms: f32, max_ms: f32) -> Vec<(ClusterId, Vec<f32>, Vec<u64>)> {
        selected
            .iter()
            .map(|&c| {
                let (centers, counts) = isi_histogram(&self.spike_samples(c), self.sample_rate(), bin_ms as f64, max_ms as f64);
                (c, centers.into_iter().map(|v| v as f32).collect(), counts)
            })
            .collect()
    }

    /// Firing rate over the recording in `bin_sec` bins: `(cluster, bin centres s, Hz)`.
    pub fn compute_firing_rate(&self, selected: &[ClusterId], bin_sec: f32) -> Vec<(ClusterId, Vec<f32>, Vec<f32>)> {
        let rate = self.sample_rate();
        selected
            .iter()
            .map(|&c| {
                let times: Vec<f64> = self.spike_samples(c).into_iter().map(|t| t as f64 / rate).collect();
                // No smoothing: plain counts per bin, in Hz
                let curve = compute_instantaneous_firing_rate(&times, self.duration_sec, bin_sec.max(0.001) as f64 * 1e3, 0.0);
                (c, curve.time_bin_centers_sec.into_iter().map(|v| v as f32).collect(), curve.rate_hz)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::synthetic_folder;
    use super::*;

    /// A recording where unit `u`'s spikes have a -100 µV dip on channel 2u+1: 8 channels, 10 s.
    fn recording(data: &SortingData) -> dsp_core::MemoryRecording {
        let (channels, total) = (8usize, 300_000usize);
        let mut samples = vec![0.0f32; channels * total];
        for (i, &t) in data.sorting.spike_times.iter().enumerate() {
            let best = (data.sorting.spike_clusters[i] as usize * 2 + 1) % channels;
            if (t as usize) < total {
                samples[best * total + t as usize] = -100.0;
            }
        }
        dsp_core::MemoryRecording::new("rec", samples, channels, 30_000.0).unwrap()
    }

    #[test]
    fn test_views_without_a_recording_show_only_what_the_sorter_saved() {
        let data = SortingData::open(&synthetic_folder("no-recording")).unwrap();
        let w = data.compute_waveforms(1, None, 50, 4);
        assert!(w.sampled.is_empty() && w.mean.is_empty(), "no recording: no waveforms invented");
        assert_eq!(w.template.len(), 4 * w.num_samples);
        assert_eq!(data.compute_features(&[1], None).source, FeatureSource::Unavailable);
        assert_eq!(data.amplitude_modes(false), vec![AmplitudeMode::Template]);
        assert!(data.compute_amplitudes(&[1], AmplitudeMode::Raw, 100, None).is_none());
        let a = data.compute_amplitudes(&[1], AmplitudeMode::Template, 100, None).unwrap();
        assert!(!a.points.is_empty() && a.points.iter().all(|p| p.cluster_id == 1));
        assert_eq!(a.histograms[0].2.iter().sum::<u64>() as usize, a.points.len());
    }

    #[test]
    fn test_views_with_a_recording() {
        let data = SortingData::open(&synthetic_folder("recording")).unwrap();
        let rec = recording(&data);
        let w = data.compute_waveforms(1, Some(&rec), 50, 4);
        assert_eq!(w.channels[0], 3);
        assert_eq!(w.sampled.len(), 50);
        // The dip lands at the window's centre on the best channel
        let pre = pre_samples(w.num_samples);
        assert!(w.mean[pre] < -90.0, "{}", w.mean[pre]);
        let raw = data.compute_amplitudes(&[1], AmplitudeMode::Raw, 50, Some(&rec)).unwrap();
        assert!(raw.points.iter().all(|p| (p.y - 100.0).abs() < 1.0), "peak-to-peak of the dip");
        let f = data.compute_features(&[1, 2], Some(&rec));
        assert_eq!(f.source, FeatureSource::Computed);
        assert_eq!(f.cells.len(), 4);
        assert!(!f.cells[0][1].1.is_empty());
    }

    #[test]
    fn test_isi_and_firing_rate() {
        let data = SortingData::open(&synthetic_folder("isi")).unwrap();
        let (_, centers, counts) = &data.compute_isi(&[0], 1.0, 200.0)[0];
        assert_eq!(centers.len(), 200);
        // Unit 0 fires every 3750 samples (125 ms) at 30 kHz
        assert_eq!(counts.iter().enumerate().max_by_key(|&(_, n)| *n).unwrap().0, 125);
        let (_, _, rates) = &data.compute_firing_rate(&[0], 1.0)[0];
        assert!((rates[3] - 8.0).abs() <= 1.0, "{}", rates[3]);
    }
}
