//! Spike sorting pipeline and cluster statistics for the phy-style sorting widgets.
//!
//! Built from `dsp-synapse` stages: threshold detection → spatial deduplication → multichannel
//! snippet batch → PCA embedding + center-of-mass depth → density-peaks clustering.
//! Clustering runs on a subsample (the algorithm is O(N²)); every other spike takes the label
//! of its nearest clustered neighbour in feature space.

use dsp_core::{Position3D, SensorLayout, SensorSite};
use dsp_synapse::traits::FeatureEmbedder;
use dsp_synapse::{
    cluster_density_peaks, compute_autocorrelogram, compute_crosscorrelogram, compute_isi_violations,
    deduplicate_spikes_spatial, detect_spikes_multichannel, estimate_noise_std,
    extract_snippet_batch_multichannel, localize_spike_center_of_mass, Correlogram, PcaFeatureEmbedder,
    SnippetBatch, SpikeEvent,
};

use crate::data::Dataset;

/// User-tunable sorting parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SortParams {
    pub num_clusters: usize,
    /// Detection threshold in robust noise standard deviations.
    pub threshold: f32,
    /// Assumed electrode pitch for the linear probe layout (µm); `.meta` files carry no geometry.
    pub pitch_um: f32,
    /// Spatial radius for merging co-detections. `None` merges by time alone, which is the
    /// safe choice while the real probe geometry is unknown (channel order ≠ physical order).
    pub dedup_radius_um: Option<f32>,
}

impl SortParams {
    /// Roughly one unit per three channels, between 2 and 32 clusters.
    pub fn for_channels(channels: usize) -> Self {
        // Threshold and time-only dedup chosen on MEArec 32 ch ground truth (see `ground_truth_agreement`)
        Self { num_clusters: (channels / 3).clamp(2, 32), threshold: 5.5, pitch_um: 25.0, dedup_radius_um: None }
    }
}

/// Number of PCA components kept per spike.
pub const NUM_PCS: usize = 3;
/// Channels in each snippet (primary + nearest neighbours).
const NEIGHBORS: usize = 5;
/// Most samples (channels × time) sorted in one run, ~512 MB of f32: all of a 10 s sample,
/// the first ~11 s of a 384-channel Neuropixels recording.
pub const SORT_SAMPLE_BUDGET: usize = 128 << 20;
/// Largest subsample handed to density-peaks clustering.
const MAX_CLUSTER_POINTS: usize = 2500;

#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub id: u32,
    /// Indices into the sorting's spike arrays, in time order.
    pub spikes: Vec<usize>,
    pub peak_channel: usize,
    /// Mean trough amplitude on the primary channel (µV, negative).
    pub amplitude: f32,
    pub snr: f32,
    pub isi_violation_pct: f32,
    pub firing_rate_hz: f32,
    /// Mean waveform on `template_channels`, `[channel][sample]` flattened.
    pub template: Vec<f32>,
    pub template_channels: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Sorting {
    pub sample_rate: f64,
    pub duration_sec: f64,
    pub pre_samples: usize,
    /// Snippets `[spike][neighbour channel][sample]`, neighbours ordered nearest-first.
    pub batch: SnippetBatch,
    pub times_sec: Vec<f64>,
    pub amplitudes: Vec<f32>,
    /// `[spike][NUM_PCS]` flattened.
    pub features: Vec<f32>,
    /// Estimated depth along the probe (µm).
    pub depth_um: Vec<f32>,
    pub labels: Vec<u32>,
    pub clusters: Vec<ClusterInfo>,
}

fn linear_layout(channels: usize, pitch_um: f32) -> SensorLayout {
    let sites = (0..channels)
        .map(|ch| SensorSite::new(ch, Position3D::new(0.0, ch as f32 * pitch_um, 0.0), 0))
        .collect();
    SensorLayout::new("Linear (assumed)", sites)
}

fn ptp(wave: &[f32]) -> f32 {
    let (mn, mx) = wave.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    (mx - mn).max(0.0)
}

impl Sorting {
    pub fn num_spikes(&self) -> usize {
        self.times_sec.len()
    }

    pub fn cluster(&self, id: u32) -> Option<&ClusterInfo> {
        self.clusters.iter().find(|c| c.id == id)
    }

    /// Spike sample indices of a cluster (sorted).
    pub fn cluster_samples(&self, id: u32) -> Vec<u64> {
        self.cluster(id)
            .map(|c| c.spikes.iter().map(|&i| (self.times_sec[i] * self.sample_rate).round() as u64).collect())
            .unwrap_or_default()
    }

    /// Auto (a == b) or cross correlogram between two clusters.
    pub fn correlogram(&self, a: u32, b: u32, bin_ms: f32, window_ms: f32) -> Correlogram {
        let sa = self.cluster_samples(a);
        if a == b {
            compute_autocorrelogram(&sa, self.sample_rate, bin_ms, window_ms)
        } else {
            compute_crosscorrelogram(&sa, &self.cluster_samples(b), self.sample_rate, bin_ms, window_ms)
        }
    }

    /// Inter-spike interval histogram (counts) over `[0, max_ms)`.
    pub fn isi_histogram(&self, id: u32, bin_ms: f32, max_ms: f32) -> Vec<u32> {
        let bins = (max_ms / bin_ms).ceil().max(1.0) as usize;
        let mut hist = vec![0u32; bins];
        if let Some(c) = self.cluster(id) {
            for w in c.spikes.windows(2) {
                let isi_ms = ((self.times_sec[w[1]] - self.times_sec[w[0]]) * 1000.0) as f32;
                let b = (isi_ms / bin_ms) as usize;
                if b < bins {
                    hist[b] += 1;
                }
            }
        }
        hist
    }

    /// Runs the pipeline over `dataset`, or over its first [`SORT_SAMPLE_BUDGET`] samples
    /// (all channels) when it is longer; chunked full-length sorting is Task 30.
    pub fn run(dataset: &Dataset, params: &SortParams) -> Sorting {
        let sr = dataset.sample_rate;
        let channels = dataset.total_channels;
        let samples = dataset.total_samples.min(SORT_SAMPLE_BUDGET / channels.max(1));
        // An unreadable range sorts as silence (no spikes) rather than failing the UI
        let data = dataset.read_all(0..samples).unwrap_or_else(|_| vec![0.0; channels * samples]);
        let data = data.as_slice();
        let layout = linear_layout(channels, params.pitch_um);

        // 1. Detection, channel by channel (refractory state stays per channel)
        let refractory = (sr * 0.001) as usize;
        let mut events: Vec<SpikeEvent> = Vec::new();
        let mut noise = Vec::with_capacity(channels);
        for ch in 0..channels {
            let trace = &data[ch * samples..(ch + 1) * samples];
            noise.push(estimate_noise_std(&trace[..samples.min((sr * 10.0) as usize)]));
            events.extend(detect_spikes_multichannel(trace, 1, samples, params.threshold, refractory).into_iter().map(
                |mut e| {
                    e.channel_id = ch;
                    e
                },
            ));
        }

        // 2. One event per action potential, on its deepest channel
        let radius = params.dedup_radius_um.unwrap_or(f32::MAX);
        let deduped = deduplicate_spikes_spatial(&events, &layout, radius, (sr * 0.0005) as u64);

        // 3. Snippets: 0.5 ms before, 1.0 ms after the trough, primary + nearest channels
        let (pre, post) = (((sr * 0.0005) as usize).max(4), ((sr * 0.001) as usize).max(8));
        let k = NEIGHBORS.min(channels).max(1);
        let batch = extract_snippet_batch_multichannel(data, channels, samples, &deduped, &layout, k, pre, post, false);
        let n = batch.num_spikes;

        let times_sec: Vec<f64> = batch.center_samples.iter().map(|&s| s as f64 / sr).collect();
        let amplitudes: Vec<f32> = (0..n).map(|i| batch.channel_slice(i, 0)[pre.min(batch.num_samples - 1)]).collect();

        // 4. Features: PCA of the multichannel snippet + center-of-mass depth
        // Too few spikes for PCA, or a failed projection: no PCA features (dim 0), cluster on depth
        let pca = if n > NUM_PCS { PcaFeatureEmbedder { num_components: NUM_PCS }.embed(&batch) } else { Ok((Vec::new(), 0)) };
        let (pcs, dim) = pca.unwrap_or_else(|e| {
            tracing::warn!("PCA features unavailable: {e}");
            (Vec::new(), 0)
        });
        let mut features = vec![0.0f32; n * NUM_PCS];
        for i in 0..n {
            for c in 0..NUM_PCS.min(dim) {
                features[i * NUM_PCS + c] = pcs[i * dim + c];
            }
        }
        let depth_um: Vec<f32> = (0..n)
            .map(|i| {
                let ids = batch.spike_channel_ids(i);
                let amps: Vec<f32> = (0..ids.len()).map(|c| ptp(batch.channel_slice(i, c))).collect();
                localize_spike_center_of_mass(ids, &amps, &layout, 2.0)[1]
            })
            .collect();

        // 5. Clustering on (depth, PC1, PC2), each scaled to comparable units
        let pc_sd = std_dev((0..n).map(|i| features[i * NUM_PCS])).max(1e-6);
        let point = |i: usize| [depth_um[i] / params.pitch_um, features[i * NUM_PCS] / pc_sd, features[i * NUM_PCS + 1] / pc_sd];
        let labels = cluster_points(n, params.num_clusters, point);

        let mut sorting = Sorting {
            sample_rate: sr,
            duration_sec: samples as f64 / sr,
            pre_samples: pre,
            batch,
            times_sec,
            amplitudes,
            features,
            depth_um,
            labels,
            clusters: Vec::new(),
        };
        sorting.build_clusters(&noise);
        sorting
    }

    /// Groups spikes per label, renumbers clusters by depth (top of probe first), computes stats.
    fn build_clusters(&mut self, noise: &[f32]) {
        let n_labels = self.labels.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_labels];
        for (i, &l) in self.labels.iter().enumerate() {
            groups[l as usize].push(i);
        }
        groups.retain(|g| !g.is_empty());
        let mean_depth = |g: &Vec<usize>| g.iter().map(|&i| self.depth_um[i]).sum::<f32>() / g.len() as f32;
        groups.sort_by(|a, b| mean_depth(a).total_cmp(&mean_depth(b)));

        let (ns, pre) = (self.batch.num_samples, self.pre_samples);
        let mut clusters = Vec::with_capacity(groups.len());
        for (id, spikes) in groups.into_iter().enumerate() {
            let mut counts = std::collections::HashMap::<usize, usize>::new();
            for &i in &spikes {
                *counts.entry(self.batch.primary_channels[i]).or_default() += 1;
            }
            let peak_channel = counts.into_iter().max_by_key(|&(ch, c)| (c, usize::MAX - ch)).map_or(0, |(ch, _)| ch);

            // Template from spikes sharing the peak channel (same neighbour set)
            let members: Vec<usize> = spikes.iter().copied().filter(|&i| self.batch.primary_channels[i] == peak_channel).collect();
            let template_channels = members.first().map(|&i| self.batch.spike_channel_ids(i).to_vec()).unwrap_or_default();
            let mut template = vec![0.0f32; template_channels.len() * ns];
            for &i in &members {
                for (t, v) in template.iter_mut().zip(self.batch.snippet_slice(i)) {
                    *t += v / members.len() as f32;
                }
            }

            let amplitude = spikes.iter().map(|&i| self.amplitudes[i]).sum::<f32>() / spikes.len() as f32;
            let trough = template.get(pre).copied().unwrap_or(amplitude);
            let snr = trough.abs() / noise.get(peak_channel).copied().unwrap_or(1.0).max(1e-6);
            let samples: Vec<u64> = spikes.iter().map(|&i| (self.times_sec[i] * self.sample_rate).round() as u64).collect();
            let isi = compute_isi_violations(&samples, self.sample_rate, self.duration_sec, 1.5, 0.0);
            for &i in &spikes {
                self.labels[i] = id as u32;
            }
            clusters.push(ClusterInfo {
                id: id as u32,
                spikes,
                peak_channel,
                amplitude,
                snr,
                isi_violation_pct: isi.violation_rate_pct,
                firing_rate_hz: samples.len() as f32 / self.duration_sec.max(1e-9) as f32,
                template,
                template_channels,
            });
        }
        self.clusters = clusters;
    }
}

fn std_dev(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let (n, sum) = values.clone().fold((0usize, 0.0f64), |(n, s), v| (n + 1, s + v as f64));
    if n < 2 {
        return 0.0;
    }
    let mean = sum / n as f64;
    let var = values.map(|v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt() as f32
}

/// Density-peaks on an evenly strided subsample, then nearest-neighbour label transfer.
fn cluster_points(n: usize, k: usize, point: impl Fn(usize) -> [f32; 3]) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let stride = n.div_ceil(MAX_CLUSTER_POINTS).max(1);
    let sample: Vec<usize> = (0..n).step_by(stride).collect();
    let m = sample.len();
    let flat: Vec<f32> = sample.iter().flat_map(|&i| point(i)).collect();

    // Cutoff: ~2% quantile of pairwise distances (Rodriguez & Laio's rule of thumb)
    let mut dists = Vec::new();
    let step = (m / 200).max(1);
    for a in (0..m).step_by(step) {
        for b in (a + 1..m).step_by(step) {
            dists.push(dist(&flat[a * 3..a * 3 + 3], &flat[b * 3..b * 3 + 3]));
        }
    }
    dists.sort_by(f32::total_cmp);
    let dc = dists.get(dists.len() / 50).copied().unwrap_or(1.0).max(1e-3);

    let res = cluster_density_peaks(&flat, m, 3, dc, k.max(1));
    let mut labels = vec![0u32; n];
    for (j, &i) in sample.iter().enumerate() {
        labels[i] = res.labels[j] as u32;
    }
    if stride > 1 {
        for i in 0..n {
            if i % stride != 0 {
                let p = point(i);
                let nearest = (0..m)
                    .min_by(|&a, &b| dist(&p, &flat[a * 3..a * 3 + 3]).total_cmp(&dist(&p, &flat[b * 3..b * 3 + 3])))
                    .unwrap_or(0);
                labels[i] = res.labels[nearest] as u32;
            }
        }
    }
    labels
}

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The synthetic recording has a large 60 Hz hum, so it needs a lower threshold than real data.
    pub fn synthetic_params(channels: usize) -> SortParams {
        SortParams { threshold: 3.0, ..SortParams::for_channels(channels) }
    }

    #[test]
    fn test_sorting_on_synthetic_recording() {
        let ds = Dataset::generate_synthetic(8, 30_000.0, 2.0);
        let s = Sorting::run(&ds, &synthetic_params(8));
        assert!(s.num_spikes() > 20, "found {} spikes", s.num_spikes());
        assert_eq!(s.labels.len(), s.num_spikes());
        assert_eq!(s.features.len(), s.num_spikes() * NUM_PCS);
        // Every spike belongs to exactly one cluster; ids are 0..n in depth order
        let total: usize = s.clusters.iter().map(|c| c.spikes.len()).sum();
        assert_eq!(total, s.num_spikes());
        for (i, c) in s.clusters.iter().enumerate() {
            assert_eq!(c.id as usize, i);
            assert_eq!(c.template.len(), c.template_channels.len() * s.batch.num_samples);
            assert!(c.spikes.windows(2).all(|w| s.times_sec[w[0]] <= s.times_sec[w[1]]));
        }
        let acg = s.correlogram(0, 0, 1.0, 20.0);
        assert_eq!(acg.counts.len(), acg.bin_centers_ms.len());
        assert_eq!(s.isi_histogram(0, 1.0, 50.0).len(), 50);
    }

    #[test]
    fn test_cluster_points_separates_blobs() {
        let pts: Vec<[f32; 3]> = (0..600).map(|i| {
            let c = (i % 3) as f32 * 10.0;
            let j = ((i * 7919) % 100) as f32 / 100.0;
            [c + j, c - j, j]
        }).collect();
        let labels = cluster_points(pts.len(), 3, |i| pts[i]);
        for g in 0..3 {
            let first = labels[g];
            assert!((g..600).step_by(3).all(|i| labels[i] == first));
        }
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[1], labels[2]);
    }

    /// Agreement with MEArec ground truth. Run with:
    /// `cargo test -p croc-app --release ground_truth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn ground_truth_agreement() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../playground/data");
        let ds = Dataset::open(&root.join("mearec_32ch_10s.bin")).unwrap();
        let gt: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("mearec_32ch_ground_truth.json")).unwrap()).unwrap();
        let t0 = std::time::Instant::now();
        let s = Sorting::run(&ds, &SortParams::for_channels(ds.total_channels));
        println!("{} spikes, {} clusters in {:.2?}", s.num_spikes(), s.clusters.len(), t0.elapsed());

        let tol = (ds.sample_rate * 0.0005) as i64;
        let sorted: Vec<(u64, u32)> = {
            let mut v: Vec<(u64, u32)> = (0..s.num_spikes()).map(|i| ((s.times_sec[i] * s.sample_rate).round() as u64, s.labels[i])).collect();
            v.sort();
            v
        };
        for (unit, v) in gt.as_object().unwrap() {
            let gt_samples: Vec<i64> = v["sample_indices"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
            let mut per_cluster = vec![0usize; s.clusters.len()];
            let mut detected = 0;
            for &g in &gt_samples {
                let lo = sorted.partition_point(|&(t, _)| (t as i64) < g - tol);
                if let Some(&(t, l)) = sorted.get(lo) {
                    if (t as i64 - g).abs() <= tol {
                        per_cluster[l as usize] += 1;
                        detected += 1;
                    }
                }
            }
            let (best, hits) = per_cluster.iter().enumerate().max_by_key(|&(_, &h)| h).map(|(c, &h)| (c, h)).unwrap_or((0, 0));
            let size = s.clusters.get(best).map_or(1, |c| c.spikes.len());
            let accuracy = hits as f32 / (gt_samples.len() + size - hits).max(1) as f32;
            println!(
                "unit {unit:>2}: {:4} GT spikes, detected {:5.1}%, best cluster #{best:<2} accuracy {:5.1}%",
                gt_samples.len(),
                100.0 * detected as f32 / gt_samples.len().max(1) as f32,
                100.0 * accuracy
            );
        }
    }
}
