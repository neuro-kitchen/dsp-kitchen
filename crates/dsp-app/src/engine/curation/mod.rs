//! Spike-sorting curation data model, Phy/Kilosort/Zarr/NWB loader, derived data computations,
//! undo/redo command history, and partial folder saver (Steps 7, 8, 9, 10).

pub mod filter;
pub mod npy;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use dsp_core::RecordingSource;
use dsp_synapse::{
    compute_autocorrelogram, compute_crosscorrelogram, extract_waveform_pca, Correlogram,
    SortedUnit, SortingOutput, UnitQualityLabel, WaveformSnippet, WaveformTemplate,
};
use serde::{Deserialize, Serialize};

use crate::engine::compute::ClusterId;
use crate::engine::data::Dataset;

pub use filter::matches_filter;

/// Quality group assigned during curation (Phy convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum ClusterGroup {
    Good,
    Mua,
    Noise,
    #[default]
    Unsorted,
}

impl ClusterGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Mua => "mua",
            Self::Noise => "noise",
            Self::Unsorted => "unsorted",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" | "singleunit" => Self::Good,
            "mua" | "multiunit" => Self::Mua,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
}

/// Sortable columns of the Clusters table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSortColumn {
    Id,
    Channel,
    Depth,
    Spikes,
    FiringRate,
    Amplitude,
    ContamPct,
    KsLabel,
    Group,
}

/// Summary row for one cluster in the Clusters table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterMeta {
    pub id: ClusterId,
    pub ch: usize,
    pub depth: f32,
    pub sh: usize,
    pub n_spikes: usize,
    pub fr: f32,
    pub amp: f32,
    pub contam_pct: f32,
    pub ks_label: String,
    pub group: ClusterGroup,
    pub custom: BTreeMap<String, String>,
}

/// Derived waveform curves for one cluster across its top channels.
#[derive(Debug, Clone)]
pub struct ClusterWaveforms {
    pub cluster_id: ClusterId,
    pub channels: Vec<usize>,
    pub positions: Vec<[f32; 2]>,
    pub num_samples: usize,
    /// Sampled individual spike waveforms: each entry is `[channels.len() * num_samples]`.
    pub sampled: Vec<Vec<f32>>,
    /// Mean waveform across sampled spikes: `[channels.len() * num_samples]`.
    pub mean: Vec<f32>,
    /// Sorter template waveform: `[channels.len() * num_samples]`.
    pub template: Vec<f32>,
}

/// Point in a 2D feature or amplitude scatter, linked back to its global spike index for lasso
/// splitting.
#[derive(Debug, Clone, Copy)]
pub struct SpikePoint {
    pub spike_index: usize,
    pub cluster_id: ClusterId,
    pub x: f32,
    pub y: f32,
}

/// 4×4 PC feature matrix data on the best 4 channels (or time on the diagonal).
#[derive(Debug, Clone)]
pub struct FeatureGridData {
    pub channels: Vec<usize>,
    /// Cell `(row, col)` in a 4×4 grid: `(background_points, cluster_points)`.
    pub cells: Vec<Vec<(Vec<SpikePoint>, Vec<SpikePoint>)>>,
}

/// $n \times n$ auto/cross-correlogram matrix for selected clusters.
#[derive(Debug, Clone)]
pub struct CorrelogramMatrix {
    pub clusters: Vec<ClusterId>,
    pub bin_ms: f32,
    pub window_ms: f32,
    pub refractory_ms: f32,
    pub cells: Vec<Vec<Correlogram>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmplitudeMode {
    Template,
    Raw,
    Feature,
}

impl AmplitudeMode {
    pub fn next(self) -> Self {
        match self {
            Self::Template => Self::Raw,
            Self::Raw => Self::Feature,
            Self::Feature => Self::Template,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Template => "Template amplitude",
            Self::Raw => "Raw peak amplitude (µV)",
            Self::Feature => "PC1 feature amplitude",
        }
    }
}

/// Amplitudes over time + marginal amplitude histogram per cluster.
#[derive(Debug, Clone)]
pub struct AmplitudePlotData {
    pub mode: AmplitudeMode,
    pub duration_sec: f64,
    pub y_min: f32,
    pub y_max: f32,
    pub background: Vec<SpikePoint>,
    pub points: Vec<SpikePoint>,
    /// Per selected cluster: `(cluster_id, bin_centers, counts)`.
    pub histograms: Vec<(ClusterId, Vec<f32>, Vec<u64>)>,
}

/// Reversible edit command on `SortingData`.
#[derive(Debug, Clone)]
pub enum CurationCommand {
    Merge {
        sources: Vec<ClusterId>,
        target: ClusterId,
        /// `(spike_index, old_cluster_id)` for every spike moved into `target`.
        prev_spikes: Vec<(usize, ClusterId)>,
        prev_metas: Vec<ClusterMeta>,
        new_meta: ClusterMeta,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    Split {
        source: ClusterId,
        created_inside: ClusterId,
        created_outside: ClusterId,
        /// `(spike_index, new_cluster_id)` for every spike originally in `source`.
        assigned_spikes: Vec<(usize, ClusterId)>,
        prev_meta: ClusterMeta,
        inside_meta: ClusterMeta,
        outside_meta: ClusterMeta,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    Label {
        changes: Vec<(ClusterId, ClusterGroup, ClusterGroup)>,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    CustomLabel {
        key: String,
        changes: Vec<(ClusterId, Option<String>, Option<String>)>,
    },
}

impl CurationCommand {
    pub fn touched_clusters(&self) -> Vec<ClusterId> {
        match self {
            Self::Merge { sources, target, .. } => {
                let mut v = sources.clone();
                v.push(*target);
                v
            }
            Self::Split { source, created_inside, created_outside, .. } => vec![*source, *created_inside, *created_outside],
            Self::Label { changes, .. } => changes.iter().map(|(id, _, _)| *id).collect(),
            Self::CustomLabel { changes, .. } => changes.iter().map(|(id, _, _)| *id).collect(),
        }
    }
}

/// Loaded sorting session and its in-memory curation state.
#[derive(Clone)]
pub struct SortingData {
    pub folder: Option<PathBuf>,
    pub name: String,
    pub dat_path: Option<PathBuf>,
    pub sample_rate: f64,
    pub duration_sec: f64,
    pub spike_times: Vec<u64>,
    pub spike_clusters: Vec<ClusterId>,
    pub spike_templates: Vec<ClusterId>,
    pub amplitudes: Vec<f32>,
    pub spike_positions: Vec<[f32; 2]>,
    pub channel_map: Vec<usize>,
    pub channel_positions: Vec<[f32; 2]>,
    pub channel_shanks: Vec<usize>,
    /// Shape `[num_templates, template_samples, num_channels]`.
    pub templates: Vec<f32>,
    pub num_templates: usize,
    pub template_samples: usize,
    pub num_channels: usize,
    /// Shape `[num_templates, num_templates]`.
    pub similar_templates: Vec<f32>,
    /// Optional `pc_features.npy` `[n_spikes, n_features, n_pc_chan]` and `pc_feature_ind.npy`
    /// `[num_templates, n_pc_chan]`.
    pub pc_features: Option<(Vec<f32>, [usize; 3])>,
    pub pc_feature_ind: Option<(Vec<usize>, [usize; 2])>,
    /// Optional `template_features.npy` `[n_spikes, n_tf]` and `template_feature_ind.npy`
    /// `[num_templates, n_tf]`.
    pub template_features: Option<(Vec<f32>, [usize; 2])>,
    pub template_feature_ind: Option<(Vec<usize>, [usize; 2])>,
    pub clusters: BTreeMap<ClusterId, ClusterMeta>,
    pub custom_keys: Vec<String>,
    next_cluster_id: ClusterId,
    pub dirty: bool,
    pub undo_stack: Vec<CurationCommand>,
    pub redo_stack: Vec<CurationCommand>,
}

impl SortingData {
    /// Opens a Phy / Kilosort 1–4 directory, `.sorting.zarr`, or NWB `/units` file.
    pub fn open(path: &Path, recording: Option<&Arc<Dataset>>) -> Result<Self> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or_default();
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if path.is_dir() && path.join("spike_times.npy").exists() {
            Self::open_phy_dir(path, recording)
        } else if file_name.ends_with(".zarr") || matches!(ext, "nwb" | "h5" | "hdf5") {
            let so = dsp_synapse::load_sorting(path).with_context(|| format!("failed to load sorting from {}", path.display()))?;
            Ok(Self::from_sorting_output(so, Some(path.to_path_buf()), recording))
        } else if path.is_file() && path.parent().is_some_and(|d| d.join("spike_times.npy").exists()) {
            Self::open_phy_dir(path.parent().unwrap(), recording)
        } else {
            bail!("{} is not a Phy/Kilosort folder (missing spike_times.npy) or .sorting.zarr / .nwb", path.display())
        }
    }

    fn open_phy_dir(dir: &Path, recording: Option<&Arc<Dataset>>) -> Result<Self> {
        let spike_times = npy::read_u64_vec(&dir.join("spike_times.npy"))?;
        let raw_clusters = npy::read_i32_vec(&dir.join("spike_clusters.npy"))?;
        let n = spike_times.len().min(raw_clusters.len());
        let spike_times: Vec<u64> = spike_times.into_iter().take(n).collect();
        let spike_clusters: Vec<ClusterId> = raw_clusters.into_iter().take(n).map(|c| c.max(0) as ClusterId).collect();

        let spike_templates: Vec<ClusterId> = if dir.join("spike_templates.npy").exists() {
            npy::read_i32_vec(&dir.join("spike_templates.npy"))
                .map(|v| v.into_iter().take(n).map(|c| c.max(0) as ClusterId).collect())
                .unwrap_or_else(|_| spike_clusters.clone())
        } else {
            spike_clusters.clone()
        };

        let amplitudes = if dir.join("amplitudes.npy").exists() {
            npy::read_f32_vec(&dir.join("amplitudes.npy")).unwrap_or_else(|_| vec![1.0; n])
        } else {
            vec![1.0; n]
        };

        let spike_positions = if dir.join("spike_positions.npy").exists() {
            npy::read_f32_nd(&dir.join("spike_positions.npy"))
                .ok()
                .and_then(|(data, shape)| {
                    let cols = *shape.get(1)?;
                    (cols >= 2).then(|| data.chunks_exact(cols).take(n).map(|r| [r[0], r[1]]).collect())
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Parse params.py
        let (mut sample_rate, raw_dat_path, mut n_channels_dat) = parse_params_py(&dir.join("params.py"));
        if let Some(rec) = recording {
            if sample_rate <= 0.0 {
                sample_rate = rec.sample_rate;
            }
            if n_channels_dat == 0 {
                n_channels_dat = rec.total_channels;
            }
        }
        if sample_rate <= 0.0 {
            sample_rate = 30_000.0;
        }
        let dat_path = raw_dat_path.and_then(|raw| resolve_dat_path(dir, &raw));

        // Channel map & positions
        let channel_positions: Vec<[f32; 2]> = if dir.join("channel_positions.npy").exists() {
            npy::read_f32_nd(&dir.join("channel_positions.npy"))
                .ok()
                .and_then(|(data, shape)| {
                    let cols = *shape.get(1).unwrap_or(&2);
                    (cols >= 2).then(|| data.chunks_exact(cols).map(|r| [r[0], r[1]]).collect())
                })
                .unwrap_or_default()
        } else if let Some(rec) = recording {
            (0..rec.total_channels)
                .map(|ch| rec.site(ch).map_or([0.0, ch as f32 * 20.0], |s| [s.position.x_um, s.position.y_um]))
                .collect()
        } else {
            Vec::new()
        };

        let num_channels = if !channel_positions.is_empty() {
            channel_positions.len()
        } else if n_channels_dat > 0 {
            n_channels_dat
        } else {
            recording.map_or(32, |r| r.total_channels)
        };

        let channel_map: Vec<usize> = if dir.join("channel_map.npy").exists() {
            npy::read_i32_vec(&dir.join("channel_map.npy"))
                .map(|v| v.into_iter().map(|c| c.max(0) as usize).collect())
                .unwrap_or_else(|_| (0..num_channels).collect())
        } else {
            (0..num_channels).collect()
        };

        let channel_shanks: Vec<usize> = if dir.join("channel_shanks.npy").exists() {
            npy::read_i32_vec(&dir.join("channel_shanks.npy"))
                .map(|v| v.into_iter().map(|c| c.max(0) as usize).collect())
                .unwrap_or_else(|_| vec![0; num_channels])
        } else if let Some(rec) = recording {
            (0..num_channels).map(|i| rec.site(*channel_map.get(i).unwrap_or(&i)).map_or(0, |s| s.shank_id)).collect()
        } else {
            vec![0; num_channels]
        };

        let channel_positions = if channel_positions.len() == num_channels {
            channel_positions
        } else {
            (0..num_channels).map(|i| [(i % 2) as f32 * 20.0, (i / 2) as f32 * 20.0]).collect()
        };

        // Templates [num_templates, template_samples, num_channels]
        let max_cid = spike_clusters.iter().chain(spike_templates.iter()).copied().max().unwrap_or(0) as usize;
        let (templates, num_templates, template_samples, num_channels) = if dir.join("templates.npy").exists() {
            match npy::read_f32_nd(&dir.join("templates.npy")) {
                Ok((data, shape)) if shape.len() == 3 => {
                    // Check for sparse templates_ind.npy if shape[2] < channel_positions.len()
                    let (nt, ns, nc) = (shape[0], shape[1], shape[2]);
                    if nc < num_channels && dir.join("templates_ind.npy").exists() {
                        if let Ok((ind, ind_shape)) = npy::read_i32_nd(&dir.join("templates_ind.npy")) {
                            if ind_shape == [nt, nc] {
                                let mut dense = vec![0.0f32; nt * ns * num_channels];
                                for t in 0..nt {
                                    for c_local in 0..nc {
                                        let ch = ind[t * nc + c_local].max(0) as usize;
                                        if ch < num_channels {
                                            for s in 0..ns {
                                                dense[(t * ns + s) * num_channels + ch] = data[(t * ns + s) * nc + c_local];
                                            }
                                        }
                                    }
                                }
                                (dense, nt, ns, num_channels)
                            } else {
                                (data, nt, ns, nc)
                            }
                        } else {
                            (data, nt, ns, nc)
                        }
                    } else {
                        (data, nt, ns, nc)
                    }
                }
                _ => synthesize_templates(max_cid + 1, 61, num_channels, &spike_times, &spike_clusters, recording),
            }
        } else {
            synthesize_templates(max_cid + 1, 61, num_channels, &spike_times, &spike_clusters, recording)
        };

        // Similar templates [num_templates, num_templates]
        let similar_templates = if dir.join("similar_templates.npy").exists() {
            match npy::read_f32_nd(&dir.join("similar_templates.npy")) {
                Ok((data, shape)) if shape.len() == 2 && shape[0] == num_templates && shape[1] == num_templates => data,
                _ => compute_cosine_similarity(&templates, num_templates, template_samples * num_channels),
            }
        } else {
            compute_cosine_similarity(&templates, num_templates, template_samples * num_channels)
        };

        // Optional pc_features.npy & pc_feature_ind.npy
        let pc_features = dir.join("pc_features.npy").exists().then(|| {
            let (data, shape) = npy::read_f32_nd(&dir.join("pc_features.npy")).ok()?;
            (shape.len() == 3).then_some((data, [shape[0], shape[1], shape[2]]))
        }).flatten();
        let pc_feature_ind = dir.join("pc_feature_ind.npy").exists().then(|| {
            let (data, shape) = npy::read_i32_nd(&dir.join("pc_feature_ind.npy")).ok()?;
            (shape.len() == 2).then(|| (data.into_iter().map(|i| i.max(0) as usize).collect(), [shape[0], shape[1]]))
        }).flatten();

        // Optional template_features.npy & template_feature_ind.npy
        let template_features = dir.join("template_features.npy").exists().then(|| {
            let (data, shape) = npy::read_f32_nd(&dir.join("template_features.npy")).ok()?;
            (shape.len() == 2).then_some((data, [shape[0], shape[1]]))
        }).flatten();
        let template_feature_ind = dir.join("template_feature_ind.npy").exists().then(|| {
            let (data, shape) = npy::read_i32_nd(&dir.join("template_feature_ind.npy")).ok()?;
            (shape.len() == 2).then(|| (data.into_iter().map(|i| i.max(0) as usize).collect(), [shape[0], shape[1]]))
        }).flatten();

        // TSV metadata: cluster_group.tsv, cluster_KSLabel.tsv, cluster_Amplitude.tsv, cluster_ContamPct.tsv, cluster_info.tsv
        let group_tsv = parse_two_col_tsv(&dir.join("cluster_group.tsv"));
        let ks_tsv = parse_two_col_tsv(&dir.join("cluster_KSLabel.tsv"));
        let amp_tsv = parse_two_col_tsv(&dir.join("cluster_Amplitude.tsv"));
        let contam_tsv = parse_two_col_tsv(&dir.join("cluster_ContamPct.tsv"));
        let (info_rows, custom_keys) = parse_cluster_info_tsv(&dir.join("cluster_info.tsv"));

        let max_sample = spike_times.iter().copied().max().unwrap_or(0);
        let rec_dur = recording.map_or(0.0, |r| r.total_samples as f64 / r.sample_rate.max(1.0));
        let duration_sec = (max_sample as f64 / sample_rate).max(rec_dur).max(1.0);

        let name = dir.file_name().map_or_else(|| dir.display().to_string(), |s| s.to_string_lossy().into_owned());
        let mut data = Self {
            folder: Some(dir.to_path_buf()),
            name,
            dat_path,
            sample_rate,
            duration_sec,
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            spike_positions,
            channel_map,
            channel_positions,
            channel_shanks,
            templates,
            num_templates,
            template_samples,
            num_channels,
            similar_templates,
            pc_features,
            pc_feature_ind,
            template_features,
            template_feature_ind,
            clusters: BTreeMap::new(),
            custom_keys,
            next_cluster_id: (max_cid as ClusterId) + 1,
            dirty: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        };
        data.rebuild_all_clusters(&group_tsv, &ks_tsv, &amp_tsv, &contam_tsv, &info_rows);
        Ok(data)
    }

    pub fn from_sorting_output(so: SortingOutput, folder: Option<PathBuf>, recording: Option<&Arc<Dataset>>) -> Self {
        let (spike_times, raw_clusters, amplitudes, locs) = so.flattened_spikes();
        let spike_clusters: Vec<ClusterId> = raw_clusters.into_iter().map(|c| c.max(0) as ClusterId).collect();
        let spike_positions: Vec<[f32; 2]> = locs.into_iter().map(|p| [p[0], p[1]]).collect();
        let num_channels = so
            .probe
            .as_ref()
            .map(|p| p.total_channels())
            .or_else(|| recording.map(|r| r.total_channels))
            .unwrap_or(32)
            .max(1);
        let channel_positions: Vec<[f32; 2]> = if let Some(probe) = &so.probe {
            probe.contacts.iter().map(|c| [c.position.x_um, c.position.y_um]).collect()
        } else {
            (0..num_channels).map(|c| [(c % 2) as f32 * 20.0, (c / 2) as f32 * 20.0]).collect()
        };
        let channel_shanks: Vec<usize> = if let Some(probe) = &so.probe {
            probe.contacts.iter().map(|c| c.shank_id).collect()
        } else {
            vec![0; num_channels]
        };
        let channel_map: Vec<usize> = (0..num_channels).collect();

        let num_templates = so.units.iter().map(|u| u.unit_id + 1).max().unwrap_or(1);
        let template_samples = so.units.iter().filter_map(|u| u.template.as_ref()).map(|t| t.num_samples).max().unwrap_or(61);
        let mut templates = vec![0.0f32; num_templates * template_samples * num_channels];
        let mut group_tsv = BTreeMap::new();
        let mut ks_tsv = BTreeMap::new();
        for u in &so.units {
            let g = match u.quality_label {
                UnitQualityLabel::SingleUnit => "good",
                UnitQualityLabel::MultiUnit => "mua",
                UnitQualityLabel::Noise => "noise",
            };
            group_tsv.insert(u.unit_id as ClusterId, g.to_string());
            ks_tsv.insert(u.unit_id as ClusterId, g.to_string());
            if let Some(t) = &u.template {
                for (r, &ch) in t.channel_ids.iter().enumerate() {
                    if ch < num_channels {
                        for s in 0..t.num_samples.min(template_samples) {
                            templates[(u.unit_id * template_samples + s) * num_channels + ch] = t.row(r)[s];
                        }
                    }
                }
            }
        }
        let similar_templates = compute_cosine_similarity(&templates, num_templates, template_samples * num_channels);
        let duration_sec = (so.total_samples as f64 / so.sample_rate_hz.max(1.0)).max(1.0);
        let next_cluster_id = (num_templates as ClusterId).max(1);
        let name = folder
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or_else(|| "sorting".into(), |s| s.to_string_lossy().into_owned());
        let mut data = Self {
            folder,
            name,
            dat_path: None,
            sample_rate: so.sample_rate_hz,
            duration_sec,
            spike_times,
            spike_templates: spike_clusters.clone(),
            spike_clusters,
            amplitudes,
            spike_positions,
            channel_map,
            channel_positions,
            channel_shanks,
            templates,
            num_templates,
            template_samples,
            num_channels,
            similar_templates,
            pc_features: None,
            pc_feature_ind: None,
            template_features: None,
            template_feature_ind: None,
            clusters: BTreeMap::new(),
            custom_keys: Vec::new(),
            next_cluster_id,
            dirty: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        };
        data.rebuild_all_clusters(&group_tsv, &ks_tsv, &BTreeMap::new(), &BTreeMap::new(), &BTreeMap::new());
        data
    }

    /// Generates a synthetic 6-cluster sorting matching `dataset` (or a standalone 16-channel probe).
    pub fn synthetic(dataset: Option<&Arc<Dataset>>) -> Self {
        let num_channels = dataset.map_or(16, |d| d.total_channels.max(4));
        let sample_rate = dataset.map_or(30_000.0, |d| d.sample_rate);
        let duration_sec = dataset.map_or(10.0, |d| (d.total_samples as f64 / d.sample_rate.max(1.0)).max(2.0));
        let total_samples = (duration_sec * sample_rate) as u64;
        let num_units = 6usize;
        let template_samples = 61usize;
        let channel_ids: Vec<usize> = (0..num_channels).collect();

        let mut units = Vec::with_capacity(num_units);
        for u in 0..num_units {
            let best_ch = (u * 2 + 1) % num_channels;
            let rate_hz = 8.0 + (u as f64) * 4.5;
            let step = (sample_rate / rate_hz).round().max(30.0) as u64;
            let mut spike_samples = Vec::new();
            let mut spike_amplitudes = Vec::new();
            let mut spike_locations = Vec::new();
            let mut t = (u as u64 + 1) * 150;
            let mut idx = 0u64;
            while t + 40 < total_samples {
                let jitter = ((idx * 37 + u as u64 * 13) % 19) as u64;
                spike_samples.push(t + jitter);
                let amp = 45.0 + (u as f32) * 18.0 + ((idx % 7) as f32 - 3.0) * 2.5;
                spike_amplitudes.push(amp);
                spike_locations.push([(best_ch % 2) as f32 * 20.0, (best_ch / 2) as f32 * 20.0, 0.0]);
                t += step;
                idx += 1;
            }
            let mut mean = vec![0.0f32; num_channels * template_samples];
            let peak = -(50.0 + u as f32 * 20.0);
            for ch in 0..num_channels {
                let dist = (ch as isize - best_ch as isize).unsigned_abs() as f32;
                let atten = (-0.5 * (dist / 1.8).powi(2)).exp();
                for s in 0..template_samples {
                    let x = (s as f32 - 20.0) / 5.0;
                    let biphasic = peak * (-0.5 * x * x).exp() - 0.35 * peak * (-0.5 * (x - 1.8).powi(2)).exp();
                    mean[ch * template_samples + s] = atten * biphasic;
                }
            }
            let template = WaveformTemplate::with_count(
                channel_ids.clone(),
                template_samples,
                spike_samples.len(),
                mean.clone(),
                vec![3.0; mean.len()],
            );
            let quality_label = match u % 3 {
                0 => UnitQualityLabel::SingleUnit,
                1 => UnitQualityLabel::MultiUnit,
                _ => UnitQualityLabel::Noise,
            };
            units.push(SortedUnit {
                unit_id: u,
                primary_channel: best_ch,
                spike_samples,
                amplitudes_uv: spike_amplitudes,
                locations_um: spike_locations,
                template: Some(template),
                quality_label,
                snr: 6.5 + u as f32,
                firing_rate_hz: rate_hz,
                isi_violation_ratio: if u % 3 == 0 { 0.002 } else { 0.08 },
                presence_ratio: 0.98,
                amplitude_cutoff: 0.01,
            });
        }
        let contacts = (0..num_channels)
            .map(|c| dsp_core::SensorSite::new(c, dsp_core::Position3D::new((c % 2) as f32 * 20.0, (c / 2) as f32 * 20.0, 0.0), c / 8))
            .collect();
        let probe = Some(dsp_core::SensorLayout::new("synthetic_probe", contacts));
        let so = SortingOutput::new("synthetic", sample_rate, total_samples, probe, units, None);
        Self::from_sorting_output(so, None, dataset)
    }

    fn rebuild_all_clusters(
        &mut self,
        group_tsv: &BTreeMap<ClusterId, String>,
        ks_tsv: &BTreeMap<ClusterId, String>,
        amp_tsv: &BTreeMap<ClusterId, String>,
        contam_tsv: &BTreeMap<ClusterId, String>,
        info_rows: &BTreeMap<ClusterId, BTreeMap<String, String>>,
    ) {
        let mut counts: BTreeMap<ClusterId, (usize, f64, u32)> = BTreeMap::new();
        for (i, &cid) in self.spike_clusters.iter().enumerate() {
            let e = counts.entry(cid).or_insert((0, 0.0, cid));
            e.0 += 1;
            e.1 += self.amplitudes.get(i).copied().unwrap_or(1.0).abs() as f64;
            if let Some(&tid) = self.spike_templates.get(i) {
                e.2 = tid;
            }
        }
        self.clusters.clear();
        for (cid, (n_spikes, sum_amp, rep_tid)) in counts {
            let ch = self.best_channel_of_template(rep_tid as usize);
            let depth = self.channel_positions.get(ch).map_or(0.0, |p| p[1]);
            let sh = self.channel_shanks.get(ch).copied().unwrap_or(0);
            let fr = (n_spikes as f64 / self.duration_sec.max(1e-6)) as f32;
            let mean_amp = if n_spikes > 0 { (sum_amp / n_spikes as f64) as f32 } else { 0.0 };
            let amp = amp_tsv
                .get(&cid)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or_else(|| self.template_peak_amplitude(rep_tid as usize).max(mean_amp));
            let contam_pct = contam_tsv
                .get(&cid)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or_else(|| self.estimate_contam_pct(cid));
            let ks_label = ks_tsv.get(&cid).cloned().or_else(|| group_tsv.get(&cid).cloned()).unwrap_or_else(|| "unsorted".into());
            let group = group_tsv.get(&cid).map(|s| ClusterGroup::parse(s)).unwrap_or(ClusterGroup::Unsorted);
            let custom = info_rows.get(&cid).cloned().unwrap_or_default();
            self.clusters.insert(
                cid,
                ClusterMeta { id: cid, ch, depth, sh, n_spikes, fr, amp, contam_pct, ks_label, group, custom },
            );
        }
    }

    /// Best channel (0..num_channels) of template `tid` by peak-to-peak amplitude.
    pub fn best_channel_of_template(&self, tid: usize) -> usize {
        if self.num_templates == 0 || self.template_samples == 0 || self.num_channels == 0 {
            return 0;
        }
        let t = tid.min(self.num_templates - 1);
        let mut best_ch = 0;
        let mut best_ptp = -1.0f32;
        for ch in 0..self.num_channels {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for s in 0..self.template_samples {
                let v = self.templates[(t * self.template_samples + s) * self.num_channels + ch];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let ptp = hi - lo;
            if ptp > best_ptp {
                best_ptp = ptp;
                best_ch = ch;
            }
        }
        best_ch
    }

    pub fn template_peak_amplitude(&self, tid: usize) -> f32 {
        if self.num_templates == 0 || self.template_samples == 0 || self.num_channels == 0 {
            return 0.0;
        }
        let t = tid.min(self.num_templates - 1);
        let mut best = 0.0f32;
        for s in 0..self.template_samples {
            for ch in 0..self.num_channels {
                best = best.max(self.templates[(t * self.template_samples + s) * self.num_channels + ch].abs());
            }
        }
        best
    }

    /// Top `k` channels of cluster `cid` ordered by template amplitude (or distance to best channel).
    pub fn top_channels(&self, cid: ClusterId, k: usize) -> Vec<usize> {
        let k = k.clamp(1, self.num_channels.max(1));
        let rep_tid = self.representative_template(cid);
        if self.num_templates > 0 && self.template_samples > 0 && self.num_channels > 0 {
            let t = rep_tid.min(self.num_templates - 1);
            let mut scored: Vec<(usize, f32)> = (0..self.num_channels)
                .map(|ch| {
                    let mut ptp = 0.0f32;
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    for s in 0..self.template_samples {
                        let v = self.templates[(t * self.template_samples + s) * self.num_channels + ch];
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    if hi > lo {
                        ptp = hi - lo;
                    }
                    (ch, ptp)
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.into_iter().take(k).map(|(ch, _)| ch).collect()
        } else {
            (0..k).collect()
        }
    }

    pub fn representative_template(&self, cid: ClusterId) -> usize {
        if (cid as usize) < self.num_templates {
            return cid as usize;
        }
        for (i, &c) in self.spike_clusters.iter().enumerate() {
            if c == cid {
                return self.spike_templates.get(i).copied().unwrap_or(0) as usize;
            }
        }
        0
    }

    fn estimate_contam_pct(&self, cid: ClusterId) -> f32 {
        let samples = self.cluster_spike_samples(cid);
        let n = samples.len();
        if n < 2 || self.sample_rate <= 0.0 {
            return 0.0;
        }
        let refrac_samples = (0.002 * self.sample_rate).round() as u64;
        let mut viol = 0usize;
        for w in samples.windows(2) {
            if w[1].saturating_sub(w[0]) <= refrac_samples {
                viol += 1;
            }
        }
        ((viol as f32 / n as f32) * 100.0).min(100.0)
    }

    /// Global spike indices belonging to `cid`, in ascending time order.
    pub fn cluster_spike_indices(&self, cid: ClusterId) -> Vec<usize> {
        self.spike_clusters
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c == cid).then_some(i))
            .collect()
    }

    /// Sorted sample timestamps of `cid`.
    pub fn cluster_spike_samples(&self, cid: ClusterId) -> Vec<u64> {
        let mut s: Vec<u64> = self
            .spike_clusters
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c == cid).then_some(self.spike_times[i]))
            .collect();
        s.sort_unstable();
        s
    }

    /// Clusters ranked by similarity to `best_cluster` (excluding `best_cluster` itself).
    pub fn similar_to(&self, best_cluster: ClusterId) -> Vec<(ClusterId, f32)> {
        let t_a = self.representative_template(best_cluster);
        let mut out: Vec<(ClusterId, f32)> = self
            .clusters
            .keys()
            .copied()
            .filter(|&cid| cid != best_cluster)
            .map(|cid| {
                let t_b = self.representative_template(cid);
                let sim = if t_a < self.num_templates && t_b < self.num_templates && self.similar_templates.len() == self.num_templates * self.num_templates {
                    self.similar_templates[t_a * self.num_templates + t_b]
                } else {
                    0.0
                };
                (cid, sim)
            })
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    fn compute_meta_for(&self, cid: ClusterId, prev: Option<&ClusterMeta>) -> ClusterMeta {
        let indices = self.cluster_spike_indices(cid);
        let n_spikes = indices.len();
        let sum_amp: f64 = indices.iter().map(|&i| self.amplitudes.get(i).copied().unwrap_or(1.0).abs() as f64).sum();
        let rep_tid = self.representative_template(cid);
        let ch = self.best_channel_of_template(rep_tid);
        let depth = self.channel_positions.get(ch).map_or(0.0, |p| p[1]);
        let sh = self.channel_shanks.get(ch).copied().unwrap_or(0);
        let fr = (n_spikes as f64 / self.duration_sec.max(1e-6)) as f32;
        let mean_amp = if n_spikes > 0 { (sum_amp / n_spikes as f64) as f32 } else { 0.0 };
        let amp = self.template_peak_amplitude(rep_tid).max(mean_amp);
        let contam_pct = self.estimate_contam_pct(cid);
        ClusterMeta {
            id: cid,
            ch,
            depth,
            sh,
            n_spikes,
            fr,
            amp,
            contam_pct,
            ks_label: prev.map_or_else(|| "unsorted".into(), |p| p.ks_label.clone()),
            group: ClusterGroup::Unsorted,
            custom: prev.map_or_else(BTreeMap::new, |p| p.custom.clone()),
        }
    }

    // ------------------------------------------------------------------------
    // Curation Edits & Undo / Redo (Step 7 & Step 9)
    // ------------------------------------------------------------------------

    /// Merges `sources` (at least 2 clusters) into a single new cluster id (`G` in Phy).
    /// Returns `(new_cluster_id, touched_clusters)`.
    pub fn merge(&mut self, sources: &[ClusterId], prev_selection: &[ClusterId]) -> Option<(ClusterId, Vec<ClusterId>)> {
        let mut uniq: Vec<ClusterId> = sources.iter().copied().filter(|c| self.clusters.contains_key(c)).collect();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() < 2 {
            return None;
        }
        let target = self.next_cluster_id;
        self.next_cluster_id += 1;

        let source_set: BTreeSet<ClusterId> = uniq.iter().copied().collect();
        let mut prev_spikes = Vec::new();
        for (i, c) in self.spike_clusters.iter_mut().enumerate() {
            if source_set.contains(c) {
                prev_spikes.push((i, *c));
                *c = target;
            }
        }
        let prev_metas: Vec<ClusterMeta> = uniq.iter().filter_map(|c| self.clusters.remove(c)).collect();
        let new_meta = self.compute_meta_for(target, prev_metas.first());
        self.clusters.insert(target, new_meta.clone());

        let next_selection = vec![target];
        let cmd = CurationCommand::Merge {
            sources: uniq,
            target,
            prev_spikes,
            prev_metas,
            new_meta,
            prev_selection: prev_selection.to_vec(),
            next_selection,
        };
        let touched = cmd.touched_clusters();
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        self.dirty = true;
        Some((target, touched))
    }

    /// Splits `source` into two new clusters (`created_inside` for spikes in `inside_spike_indices`
    /// and `created_outside` for the remaining spikes of `source`, matching Phy's `K` split).
    pub fn split(
        &mut self,
        source: ClusterId,
        inside_spike_indices: &[usize],
        prev_selection: &[ClusterId],
    ) -> Option<(ClusterId, ClusterId, Vec<ClusterId>)> {
        let prev_meta = self.clusters.get(&source)?.clone();
        let inside_set: BTreeSet<usize> = inside_spike_indices.iter().copied().collect();
        let all_spikes = self.cluster_spike_indices(source);
        let inside_count = all_spikes.iter().filter(|i| inside_set.contains(i)).count();
        if inside_count == 0 || inside_count == all_spikes.len() {
            return None;
        }
        let created_inside = self.next_cluster_id;
        let created_outside = self.next_cluster_id + 1;
        self.next_cluster_id += 2;

        let mut assigned_spikes = Vec::with_capacity(all_spikes.len());
        for i in all_spikes {
            let next_c = if inside_set.contains(&i) { created_inside } else { created_outside };
            self.spike_clusters[i] = next_c;
            assigned_spikes.push((i, next_c));
        }
        self.clusters.remove(&source);
        let inside_meta = self.compute_meta_for(created_inside, Some(&prev_meta));
        let outside_meta = self.compute_meta_for(created_outside, Some(&prev_meta));
        self.clusters.insert(created_inside, inside_meta.clone());
        self.clusters.insert(created_outside, outside_meta.clone());

        let next_selection = vec![created_inside, created_outside];
        let cmd = CurationCommand::Split {
            source,
            created_inside,
            created_outside,
            assigned_spikes,
            prev_meta,
            inside_meta,
            outside_meta,
            prev_selection: prev_selection.to_vec(),
            next_selection,
        };
        let touched = cmd.touched_clusters();
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        self.dirty = true;
        Some((created_inside, created_outside, touched))
    }

    /// Sets the quality group (`good` / `mua` / `noise` / `unsorted`) on `targets`.
    pub fn set_group(
        &mut self,
        targets: &[ClusterId],
        group: ClusterGroup,
        prev_selection: &[ClusterId],
        next_selection: &[ClusterId],
    ) -> bool {
        let mut changes = Vec::new();
        for &cid in targets {
            if let Some(meta) = self.clusters.get_mut(&cid) {
                if meta.group != group {
                    changes.push((cid, meta.group, group));
                    meta.group = group;
                }
            }
        }
        if changes.is_empty() {
            return false;
        }
        self.undo_stack.push(CurationCommand::Label {
            changes,
            prev_selection: prev_selection.to_vec(),
            next_selection: next_selection.to_vec(),
        });
        self.redo_stack.clear();
        self.dirty = true;
        true
    }

    /// Sets a custom key=value label on `targets` (and registers `key` as a table column).
    pub fn set_custom_label(&mut self, targets: &[ClusterId], key: &str, value: &str) -> bool {
        let key = key.trim().to_string();
        if key.is_empty() {
            return false;
        }
        let val_opt = (!value.trim().is_empty()).then(|| value.trim().to_string());
        let mut changes = Vec::new();
        for &cid in targets {
            if let Some(meta) = self.clusters.get_mut(&cid) {
                let prev = meta.custom.get(&key).cloned();
                if prev != val_opt {
                    match &val_opt {
                        Some(v) => {
                            meta.custom.insert(key.clone(), v.clone());
                        }
                        None => {
                            meta.custom.remove(&key);
                        }
                    }
                    changes.push((cid, prev, val_opt.clone()));
                }
            }
        }
        if changes.is_empty() {
            return false;
        }
        if !self.custom_keys.contains(&key) {
            self.custom_keys.push(key.clone());
        }
        self.undo_stack.push(CurationCommand::CustomLabel { key, changes });
        self.redo_stack.clear();
        self.dirty = true;
        true
    }

    /// Undoes the last command. Returns `(touched_clusters, restored_selection)`.
    pub fn undo(&mut self) -> Option<(Vec<ClusterId>, Option<Vec<ClusterId>>)> {
        let cmd = self.undo_stack.pop()?;
        let touched = cmd.touched_clusters();
        let sel = match &cmd {
            CurationCommand::Merge { target, prev_spikes, prev_metas, prev_selection, .. } => {
                for &(i, old_c) in prev_spikes {
                    self.spike_clusters[i] = old_c;
                }
                self.clusters.remove(target);
                for m in prev_metas {
                    self.clusters.insert(m.id, m.clone());
                }
                Some(prev_selection.clone())
            }
            CurationCommand::Split { source, created_inside, created_outside, assigned_spikes, prev_meta, prev_selection, .. } => {
                for &(i, _) in assigned_spikes {
                    self.spike_clusters[i] = *source;
                }
                self.clusters.remove(created_inside);
                self.clusters.remove(created_outside);
                self.clusters.insert(*source, prev_meta.clone());
                Some(prev_selection.clone())
            }
            CurationCommand::Label { changes, prev_selection, .. } => {
                for (cid, prev_g, _) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        m.group = *prev_g;
                    }
                }
                Some(prev_selection.clone())
            }
            CurationCommand::CustomLabel { key, changes } => {
                for (cid, prev_v, _) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        match prev_v {
                            Some(v) => {
                                m.custom.insert(key.clone(), v.clone());
                            }
                            None => {
                                m.custom.remove(key);
                            }
                        }
                    }
                }
                None
            }
        };
        self.redo_stack.push(cmd);
        self.dirty = !self.undo_stack.is_empty();
        Some((touched, sel))
    }

    /// Redoes the last undone command. Returns `(touched_clusters, restored_selection)`.
    pub fn redo(&mut self) -> Option<(Vec<ClusterId>, Option<Vec<ClusterId>>)> {
        let cmd = self.redo_stack.pop()?;
        let touched = cmd.touched_clusters();
        let sel = match &cmd {
            CurationCommand::Merge { sources, target, prev_spikes, new_meta, next_selection, .. } => {
                for &(i, _) in prev_spikes {
                    self.spike_clusters[i] = *target;
                }
                for s in sources {
                    self.clusters.remove(s);
                }
                self.clusters.insert(*target, new_meta.clone());
                Some(next_selection.clone())
            }
            CurationCommand::Split { source, created_inside, created_outside, assigned_spikes, inside_meta, outside_meta, next_selection, .. } => {
                for &(i, new_c) in assigned_spikes {
                    self.spike_clusters[i] = new_c;
                }
                self.clusters.remove(source);
                self.clusters.insert(*created_inside, inside_meta.clone());
                self.clusters.insert(*created_outside, outside_meta.clone());
                Some(next_selection.clone())
            }
            CurationCommand::Label { changes, next_selection, .. } => {
                for (cid, _, next_g) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        m.group = *next_g;
                    }
                }
                Some(next_selection.clone())
            }
            CurationCommand::CustomLabel { key, changes } => {
                for (cid, _, next_v) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        match next_v {
                            Some(v) => {
                                m.custom.insert(key.clone(), v.clone());
                            }
                            None => {
                                m.custom.remove(key);
                            }
                        }
                    }
                }
                None
            }
        };
        self.undo_stack.push(cmd);
        self.dirty = true;
        Some((touched, sel))
    }

    // ------------------------------------------------------------------------
    // Partial Save (Step 7)
    // ------------------------------------------------------------------------

    /// Writes `spike_clusters.npy`, `cluster_group.tsv`, and `cluster_info.tsv` in `dir`, making a
    /// `.bak` copy of each file the first time it is overwritten, without touching any other file.
    pub fn save_to(&mut self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        for name in ["spike_clusters.npy", "cluster_group.tsv", "cluster_info.tsv"] {
            let file = dir.join(name);
            let bak = dir.join(format!("{name}.bak"));
            if file.exists() && !bak.exists() {
                let _ = std::fs::copy(&file, &bak);
            }
        }

        let i32_clusters: Vec<i32> = self.spike_clusters.iter().map(|&c| c as i32).collect();
        npy::write_i32_1d(&dir.join("spike_clusters.npy"), &i32_clusters)?;

        let mut grp = File::create(dir.join("cluster_group.tsv"))?;
        writeln!(grp, "cluster_id\tgroup")?;
        for c in self.clusters.values() {
            writeln!(grp, "{}\t{}", c.id, c.group.as_str())?;
        }

        let mut info = File::create(dir.join("cluster_info.tsv"))?;
        write!(info, "cluster_id\tch\tdepth\tsh\tn_spikes\tfr\tamp\tcontam_pct\tKSLabel\tgroup")?;
        for k in &self.custom_keys {
            write!(info, "\t{k}")?;
        }
        writeln!(info)?;
        for c in self.clusters.values() {
            write!(
                info,
                "{}\t{}\t{:.1}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}",
                c.id,
                c.ch,
                c.depth,
                c.sh,
                c.n_spikes,
                c.fr,
                c.amp,
                c.contam_pct,
                c.ks_label,
                c.group.as_str()
            )?;
            for k in &self.custom_keys {
                write!(info, "\t{}", c.custom.get(k).map_or("", String::as_str))?;
            }
            writeln!(info)?;
        }

        self.folder = Some(dir.to_path_buf());
        self.dirty = false;
        Ok(())
    }

    pub fn save(&mut self) -> Result<()> {
        let dir = self.folder.clone().context("no sorting folder to save to")?;
        self.save_to(&dir)
    }

    // ------------------------------------------------------------------------
    // Derived Computations for Curation Views (Steps 8 & 10)
    // ------------------------------------------------------------------------

    /// Computes sampled waveforms, mean waveform, and template waveform for `cid` across its top
    /// `max_channels` channels.
    pub fn compute_waveforms(
        &self,
        cid: ClusterId,
        dataset: Option<&Arc<Dataset>>,
        max_spikes: usize,
        max_channels: usize,
    ) -> ClusterWaveforms {
        let channels = self.top_channels(cid, max_channels);
        let positions: Vec<[f32; 2]> = channels
            .iter()
            .map(|&ch| self.channel_positions.get(ch).copied().unwrap_or([0.0, ch as f32 * 20.0]))
            .collect();
        let ns = self.template_samples.max(41);
        let half = (ns / 2) as u64;
        let rep_tid = self.representative_template(cid);

        let mut template = vec![0.0f32; channels.len() * ns];
        if self.num_templates > 0 && self.template_samples == ns {
            let t = rep_tid.min(self.num_templates - 1);
            for (r, &ch) in channels.iter().enumerate() {
                if ch < self.num_channels {
                    for s in 0..ns {
                        template[r * ns + s] = self.templates[(t * ns + s) * self.num_channels + ch];
                    }
                }
            }
        }

        let indices = self.cluster_spike_indices(cid);
        let step = (indices.len() / max_spikes.max(1)).max(1);
        let chosen: Vec<usize> = indices.iter().copied().step_by(step).take(max_spikes).collect();

        let mut sampled = Vec::with_capacity(chosen.len());
        let mut mean = vec![0.0f32; channels.len() * ns];

        if let Some(ds) = dataset {
            let rec_channels: Vec<usize> = channels
                .iter()
                .map(|&ch| self.channel_map.get(ch).copied().unwrap_or(ch).min(ds.total_channels.saturating_sub(1)))
                .collect();
            let mut buf = vec![0.0f32; rec_channels.len() * ns];
            for &spk_idx in &chosen {
                let center = self.spike_times[spk_idx];
                if center >= half && (center + ns as u64 - half) <= ds.total_samples as u64 {
                    let start = center - half;
                    if ds.read(&rec_channels, start..start + ns as u64, &mut buf).is_ok() {
                        // Remove per-channel baseline mean
                        let mut snip = buf.clone();
                        for r in 0..rec_channels.len() {
                            let row = &mut snip[r * ns..(r + 1) * ns];
                            let dc = row.iter().sum::<f32>() / ns as f32;
                            for (s, v) in row.iter_mut().enumerate() {
                                *v -= dc;
                                mean[r * ns + s] += *v;
                            }
                        }
                        sampled.push(snip);
                    }
                }
            }
        }

        if sampled.is_empty() {
            // Synthesize realistic noisy spike draws around the template when no raw file is attached
            for (k, &spk_idx) in chosen.iter().enumerate() {
                let scale = self.amplitudes.get(spk_idx).copied().unwrap_or(1.0).clamp(0.5, 2.0);
                let mut snip = vec![0.0f32; channels.len() * ns];
                for i in 0..snip.len() {
                    let jitter = (((spk_idx * 31 + i * 17 + k * 13) % 100) as f32 / 50.0 - 1.0) * 3.5;
                    snip[i] = template[i] * scale + jitter;
                    mean[i] += snip[i];
                }
                sampled.push(snip);
            }
        }

        if !sampled.is_empty() {
            let inv = 1.0 / sampled.len() as f32;
            for v in &mut mean {
                *v *= inv;
            }
        } else {
            mean.clone_from(&template);
        }

        ClusterWaveforms { cluster_id: cid, channels, positions, num_samples: ns, sampled, mean, template }
    }

    /// Computes the 4×4 PC feature grid on the best 4 channels of the primary cluster.
    /// Uses `pc_features.npy` when present; otherwise extracts waveform PCs via
    /// `dsp_synapse::extract_waveform_pca`.
    pub fn compute_features(&self, selected: &[ClusterId], dataset: Option<&Arc<Dataset>>) -> FeatureGridData {
        let primary = selected.first().copied().or_else(|| self.clusters.keys().next().copied()).unwrap_or(0);
        let channels = self.top_channels(primary, 4);
        let g = channels.len().max(1);

        // Sample up to 300 spikes per selected cluster + 300 background spikes
        let sel_set: BTreeSet<ClusterId> = selected.iter().copied().collect();
        let mut sel_spikes = Vec::new();
        for &cid in selected {
            let idx = self.cluster_spike_indices(cid);
            let step = (idx.len() / 300).max(1);
            sel_spikes.extend(idx.into_iter().step_by(step).take(300));
        }
        let bg_step = (self.spike_times.len() / 300).max(1);
        let bg_spikes: Vec<usize> = (0..self.spike_times.len())
            .step_by(bg_step)
            .filter(|&i| !sel_set.contains(&self.spike_clusters[i]))
            .take(300)
            .collect();

        let all_spikes: Vec<usize> = bg_spikes.iter().chain(sel_spikes.iter()).copied().collect();
        // Per-channel PC1 value for each spike in `all_spikes`
        let mut ch_pc1: Vec<Vec<f32>> = vec![vec![0.0; all_spikes.len()]; g];

        if let (Some((pcs, [n_spk, n_feat, n_pc_ch])), Some((ind, [n_tmpl, _]))) = (&self.pc_features, &self.pc_feature_ind) {
            for (local_i, &spk) in all_spikes.iter().enumerate() {
                if spk >= *n_spk || *n_feat == 0 {
                    continue;
                }
                let tid = (self.spike_templates.get(spk).copied().unwrap_or(0) as usize).min(n_tmpl.saturating_sub(1));
                for (c_pos, &target_ch) in channels.iter().enumerate() {
                    if let Some(pc_col) = (0..*n_pc_ch).position(|k| ind.get(tid * *n_pc_ch + k).copied() == Some(target_ch)) {
                        ch_pc1[c_pos][local_i] = pcs[(spk * *n_feat) * *n_pc_ch + pc_col];
                    }
                }
            }
        } else {
            // Compute PC1 per channel from waveform snippets using dsp_synapse::extract_waveform_pca
            let ns = self.template_samples.max(41);
            let half = (ns / 2) as u64;
            for (c_pos, &ch) in channels.iter().enumerate() {
                let mut snippets = Vec::with_capacity(all_spikes.len());
                let rec_ch = self.channel_map.get(ch).copied().unwrap_or(ch);
                for &spk in &all_spikes {
                    let mut wf = vec![0.0f32; ns];
                    let mut read_ok = false;
                    if let Some(ds) = dataset {
                        let center = self.spike_times[spk];
                        if rec_ch < ds.total_channels && center >= half && center + ns as u64 - half <= ds.total_samples as u64 {
                            read_ok = ds.read(&[rec_ch], (center - half)..(center - half + ns as u64), &mut wf).is_ok();
                        }
                    }
                    if !read_ok {
                        let tid = (self.spike_templates.get(spk).copied().unwrap_or(0) as usize).min(self.num_templates.saturating_sub(1));
                        let amp = self.amplitudes.get(spk).copied().unwrap_or(1.0);
                        for s in 0..ns.min(self.template_samples) {
                            let base = if self.num_templates > 0 && ch < self.num_channels {
                                self.templates[(tid * self.template_samples + s) * self.num_channels + ch]
                            } else {
                                0.0
                            };
                            let noise = (((spk * 29 + s * 11 + c_pos * 7) % 100) as f32 / 50.0 - 1.0) * 2.5;
                            wf[s] = base * amp + noise;
                        }
                    }
                    snippets.push(WaveformSnippet {
                        primary_channel: ch,
                        center_sample: self.spike_times[spk],
                        subsample_offset: 0.0,
                        channel_ids: vec![ch],
                        num_samples: ns,
                        waveform: wf,
                    });
                }
                if let Some((_, projected)) = extract_waveform_pca(&snippets, 1) {
                    for (local_i, pcs) in projected.into_iter().enumerate() {
                        ch_pc1[c_pos][local_i] = pcs.first().copied().unwrap_or(0.0);
                    }
                }
            }
        }

        let n_bg = bg_spikes.len();
        let mut cells = vec![vec![(Vec::new(), Vec::new()); g]; g];
        for r in 0..g {
            for c in 0..g {
                let mut bg_pts = Vec::with_capacity(n_bg);
                let mut cl_pts = Vec::with_capacity(all_spikes.len().saturating_sub(n_bg));
                for (local_i, &spk) in all_spikes.iter().enumerate() {
                    let x = if r == c {
                        // Diagonal shows time (s) vs PC1 on channel r (as in Phy when toggled or single-PC view)
                        (self.spike_times[spk] as f64 / self.sample_rate.max(1.0)) as f32
                    } else {
                        ch_pc1[c][local_i]
                    };
                    let y = ch_pc1[r][local_i];
                    let pt = SpikePoint { spike_index: spk, cluster_id: self.spike_clusters[spk], x, y };
                    if local_i < n_bg {
                        bg_pts.push(pt);
                    } else {
                        cl_pts.push(pt);
                    }
                }
                cells[r][c] = (bg_pts, cl_pts);
            }
        }
        FeatureGridData { channels, cells }
    }

    /// Computes the $n \times n$ auto- and cross-correlogram matrix for up to 20 selected clusters.
    pub fn compute_correlograms(&self, selected: &[ClusterId], bin_ms: f32, window_ms: f32, refractory_ms: f32) -> CorrelogramMatrix {
        let clusters: Vec<ClusterId> = selected.iter().copied().take(20).collect();
        let trains: Vec<Vec<u64>> = clusters.iter().map(|&c| self.cluster_spike_samples(c)).collect();
        let n = clusters.len();
        let mut cells = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                let ccg = if i == j {
                    compute_autocorrelogram(&trains[i], self.sample_rate, bin_ms, window_ms)
                } else {
                    compute_crosscorrelogram(&trains[i], &trains[j], self.sample_rate, bin_ms, window_ms)
                };
                row.push(ccg);
            }
            cells.push(row);
        }
        CorrelogramMatrix { clusters, bin_ms, window_ms, refractory_ms, cells }
    }

    /// Computes amplitudes over time + side histograms for `selected`.
    pub fn compute_amplitudes(&self, selected: &[ClusterId], mode: AmplitudeMode, max_per_cluster: usize) -> AmplitudePlotData {
        let sel_set: BTreeSet<ClusterId> = selected.iter().copied().collect();
        let mut points = Vec::new();
        let mut background = Vec::new();
        let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);

        let spike_amp = |i: usize| -> f32 {
            let base = self.amplitudes.get(i).copied().unwrap_or(1.0);
            match mode {
                AmplitudeMode::Template => base,
                AmplitudeMode::Raw => {
                    let tid = self.spike_templates.get(i).copied().unwrap_or(0) as usize;
                    base * self.template_peak_amplitude(tid).max(1.0)
                }
                AmplitudeMode::Feature => {
                    if let Some((pcs, [n_spk, n_feat, n_ch])) = &self.pc_features {
                        if i < *n_spk && *n_feat > 0 && *n_ch > 0 {
                            return pcs[(i * *n_feat) * *n_ch].abs();
                        }
                    }
                    base * 1.25
                }
            }
        };

        let bg_step = (self.spike_times.len() / 500).max(1);
        for i in (0..self.spike_times.len()).step_by(bg_step) {
            let cid = self.spike_clusters[i];
            if !sel_set.contains(&cid) {
                let t = (self.spike_times[i] as f64 / self.sample_rate.max(1.0)) as f32;
                let y = spike_amp(i);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
                background.push(SpikePoint { spike_index: i, cluster_id: cid, x: t, y });
            }
        }

        let mut per_cluster_vals: BTreeMap<ClusterId, Vec<f32>> = BTreeMap::new();
        for &cid in selected {
            let indices = self.cluster_spike_indices(cid);
            let step = (indices.len() / max_per_cluster.max(1)).max(1);
            let vals = per_cluster_vals.entry(cid).or_default();
            for i in indices.into_iter().step_by(step) {
                let t = (self.spike_times[i] as f64 / self.sample_rate.max(1.0)) as f32;
                let y = spike_amp(i);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
                vals.push(y);
                points.push(SpikePoint { spike_index: i, cluster_id: cid, x: t, y });
            }
        }

        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = 0.0;
            y_max = 1.0;
        } else if (y_max - y_min).abs() < 1e-5 {
            y_min -= 1.0;
            y_max += 1.0;
        }

        let n_bins = 32usize;
        let span = (y_max - y_min).max(1e-5);
        let bin_w = span / n_bins as f32;
        let centers: Vec<f32> = (0..n_bins).map(|b| y_min + (b as f32 + 0.5) * bin_w).collect();
        let mut histograms = Vec::with_capacity(selected.len());
        for &cid in selected {
            let mut counts = vec![0u64; n_bins];
            if let Some(vals) = per_cluster_vals.get(&cid) {
                for &v in vals {
                    let b = (((v - y_min) / span) * n_bins as f32).floor() as isize;
                    counts[b.clamp(0, n_bins as isize - 1) as usize] += 1;
                }
            }
            histograms.push((cid, centers.clone(), counts));
        }

        AmplitudePlotData { mode, duration_sec: self.duration_sec, y_min, y_max, background, points, histograms }
    }

    /// Computes ISI histograms (`[0, max_ms]`) per selected cluster.
    pub fn compute_isi(&self, selected: &[ClusterId], bin_ms: f32, max_ms: f32) -> Vec<(ClusterId, Vec<f32>, Vec<u64>)> {
        let bin_ms = bin_ms.max(0.1);
        let n_bins = ((max_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
        let centers: Vec<f32> = (0..n_bins).map(|b| (b as f32 + 0.5) * bin_ms).collect();
        let ms_per_sample = 1000.0 / self.sample_rate.max(1.0);
        selected
            .iter()
            .map(|&cid| {
                let samples = self.cluster_spike_samples(cid);
                let mut counts = vec![0u64; n_bins];
                for w in samples.windows(2) {
                    let dt_ms = (w[1].saturating_sub(w[0]) as f64 * ms_per_sample) as f32;
                    let b = (dt_ms / bin_ms).floor() as usize;
                    if b < n_bins {
                        counts[b] += 1;
                    }
                }
                (cid, centers.clone(), counts)
            })
            .collect()
    }

    /// Computes firing rate curves (`Hz` over recording time) per selected cluster.
    pub fn compute_firing_rate(&self, selected: &[ClusterId], bin_sec: f32) -> Vec<(ClusterId, Vec<f32>, Vec<f32>)> {
        let bin_sec = bin_sec.max(0.05);
        let n_bins = ((self.duration_sec as f32 / bin_sec).ceil() as usize).max(1);
        let centers: Vec<f32> = (0..n_bins).map(|b| (b as f32 + 0.5) * bin_sec).collect();
        selected
            .iter()
            .map(|&cid| {
                let samples = self.cluster_spike_samples(cid);
                let mut rates = vec![0.0f32; n_bins];
                for s in samples {
                    let t = (s as f64 / self.sample_rate.max(1.0)) as f32;
                    let b = (t / bin_sec).floor() as usize;
                    if b < n_bins {
                        rates[b] += 1.0 / bin_sec;
                    }
                }
                (cid, centers.clone(), rates)
            })
            .collect()
    }
}

fn parse_params_py(path: &Path) -> (f64, Option<String>, usize) {
    let mut sample_rate = 0.0f64;
    let mut dat_path = None;
    let mut n_channels = 0usize;
    let Ok(f) = File::open(path) else { return (sample_rate, dat_path, n_channels) };
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let Some((k, v)) = line.split_once('=') else { continue };
        let k = k.trim();
        let v = v.split('#').next().unwrap_or(v).trim();
        match k {
            "sample_rate" => {
                if let Ok(sr) = v.parse::<f64>() {
                    sample_rate = sr;
                }
            }
            "n_channels_dat" => {
                if let Ok(nc) = v.parse::<usize>() {
                    n_channels = nc;
                }
            }
            "dat_path" => {
                let cleaned = v.trim_matches(|c| c == '\'' || c == '"' || c == '[' || c == ']').to_string();
                if !cleaned.is_empty() {
                    dat_path = Some(cleaned);
                }
            }
            _ => {}
        }
    }
    (sample_rate, dat_path, n_channels)
}

/// Resolves `dat_path` from `params.py`: tries the literal path first, then looks for a file with
/// the same basename in `sorting_dir`, its parent, or its grandparent (handles Windows paths moved
/// to Linux).
pub fn resolve_dat_path(sorting_dir: &Path, raw_dat_path: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(raw_dat_path);
    if direct.exists() {
        return Some(direct);
    }
    let rel = sorting_dir.join(raw_dat_path);
    if rel.exists() {
        return Some(rel);
    }
    let file_name = raw_dat_path.rsplit(['/', '\\']).next()?;
    if file_name.is_empty() {
        return None;
    }
    let mut anc = Some(sorting_dir);
    for _ in 0..4 {
        let dir = anc?;
        let candidate = dir.join(file_name);
        if candidate.exists() {
            return Some(candidate);
        }
        anc = dir.parent();
    }
    None
}

fn parse_two_col_tsv(path: &Path) -> BTreeMap<ClusterId, String> {
    let mut map = BTreeMap::new();
    let Ok(f) = File::open(path) else { return map };
    for (idx, line) in BufReader::new(f).lines().map_while(Result::ok).enumerate() {
        if idx == 0 && line.contains("cluster_id") {
            continue;
        }
        let mut parts = line.split('\t');
        if let (Some(id_s), Some(val_s)) = (parts.next(), parts.next()) {
            if let Ok(id) = id_s.trim().parse::<ClusterId>() {
                map.insert(id, val_s.trim().to_string());
            }
        }
    }
    map
}

const BUILTIN_INFO_COLS: &[&str] = &[
    "cluster_id",
    "id",
    "ch",
    "channel",
    "depth",
    "sh",
    "shank",
    "n_spikes",
    "fr",
    "firing_rate",
    "amp",
    "amplitude",
    "contam_pct",
    "isi_viol",
    "kslabel",
    "ks_label",
    "group",
    "snr",
    "presence_ratio",
    "amplitude_cutoff",
];

fn parse_cluster_info_tsv(path: &Path) -> (BTreeMap<ClusterId, BTreeMap<String, String>>, Vec<String>) {
    let mut rows: BTreeMap<ClusterId, BTreeMap<String, String>> = BTreeMap::new();
    let mut custom_keys: Vec<String> = Vec::new();
    let Ok(f) = File::open(path) else { return (rows, custom_keys) };
    let mut lines = BufReader::new(f).lines().map_while(Result::ok);
    let Some(header_line) = lines.next() else { return (rows, custom_keys) };
    let headers: Vec<String> = header_line.split('\t').map(|s| s.trim().to_string()).collect();
    let Some(id_col) = headers.iter().position(|h| h == "cluster_id" || h == "id") else {
        return (rows, custom_keys);
    };
    for h in &headers {
        if !h.is_empty() && !BUILTIN_INFO_COLS.contains(&h.to_ascii_lowercase().as_str()) {
            custom_keys.push(h.clone());
        }
    }
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        let Some(id) = cols.get(id_col).and_then(|s| s.trim().parse::<ClusterId>().ok()) else { continue };
        let mut custom = BTreeMap::new();
        for (i, h) in headers.iter().enumerate() {
            if custom_keys.contains(h) {
                if let Some(val) = cols.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    custom.insert(h.clone(), val.to_string());
                }
            }
        }
        rows.insert(id, custom);
    }
    (rows, custom_keys)
}

fn compute_cosine_similarity(templates: &[f32], num_templates: usize, stride: usize) -> Vec<f32> {
    if num_templates == 0 || stride == 0 || templates.len() < num_templates * stride {
        return Vec::new();
    }
    let norms: Vec<f32> = (0..num_templates)
        .map(|u| {
            let s = &templates[u * stride..(u + 1) * stride];
            s.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-9)
        })
        .collect();
    let mut sim = vec![0.0f32; num_templates * num_templates];
    for a in 0..num_templates {
        let sa = &templates[a * stride..(a + 1) * stride];
        for b in a..num_templates {
            let sb = &templates[b * stride..(b + 1) * stride];
            let dot: f32 = sa.iter().zip(sb).map(|(&x, &y)| x * y).sum();
            let v = (dot / (norms[a] * norms[b])).clamp(-1.0, 1.0);
            sim[a * num_templates + b] = v;
            sim[b * num_templates + a] = v;
        }
    }
    sim
}

fn synthesize_templates(
    num_templates: usize,
    template_samples: usize,
    num_channels: usize,
    spike_times: &[u64],
    spike_clusters: &[ClusterId],
    recording: Option<&Arc<Dataset>>,
) -> (Vec<f32>, usize, usize, usize) {
    let nt = num_templates.max(1);
    let ns = template_samples.max(41);
    let nc = num_channels.max(1);
    let mut out = vec![0.0f32; nt * ns * nc];
    if let Some(ds) = recording {
        let half = (ns / 2) as u64;
        let chans: Vec<usize> = (0..nc.min(ds.total_channels)).collect();
        let mut buf = vec![0.0f32; chans.len() * ns];
        let mut counts = vec![0usize; nt];
        for (i, &cid) in spike_clusters.iter().enumerate() {
            let tid = (cid as usize).min(nt - 1);
            if counts[tid] >= 30 {
                continue;
            }
            let center = spike_times[i];
            if center >= half && center + ns as u64 - half <= ds.total_samples as u64 && ds.read(&chans, (center - half)..(center - half + ns as u64), &mut buf).is_ok() {
                for (r, &ch) in chans.iter().enumerate() {
                    for s in 0..ns {
                        out[(tid * ns + s) * nc + ch] += buf[r * ns + s];
                    }
                }
                counts[tid] += 1;
            }
        }
        for tid in 0..nt {
            if counts[tid] > 0 {
                let inv = 1.0 / counts[tid] as f32;
                for s in 0..ns {
                    for ch in 0..nc {
                        out[(tid * ns + s) * nc + ch] *= inv;
                    }
                }
                continue;
            }
        }
    }
    for tid in 0..nt {
        let best_ch = tid % nc;
        let has_any = (0..ns * nc).any(|k| out[tid * ns * nc + k].abs() > 1e-6);
        if !has_any {
            for s in 0..ns {
                let x = (s as f32 - (ns as f32 * 0.35)) / 5.0;
                out[(tid * ns + s) * nc + best_ch] = -60.0 * (-0.5 * x * x).exp();
            }
        }
    }
    (out, nt, ns, nc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dsp-app-curation-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_merge_split_label_undo_redo_and_partial_save_round_trip() {
        let dir = scratch_dir("roundtrip");
        let mut data = SortingData::synthetic(None);
        assert_eq!(data.clusters.len(), 6);

        // Save initial full phy folder via dsp_synapse::save_phy_folder to verify interoperability
        let extra_file = dir.join("untouched_marker.txt");
        std::fs::write(&extra_file, "keep me").unwrap();
        npy::write_u64_for_test(&dir.join("spike_times.npy"), &data.spike_times);
        data.save_to(&dir).unwrap();
        assert!(!dir.join("spike_clusters.npy.bak").exists(), "first create does not need .bak");

        // Second save creates .bak backups
        let orig_spikes_0 = data.cluster_spike_indices(0).len();
        let orig_spikes_1 = data.cluster_spike_indices(1).len();
        let (merged_id, _) = data.merge(&[0, 1], &[0, 1]).unwrap();
        assert!(data.dirty);
        assert!(!data.clusters.contains_key(&0) && !data.clusters.contains_key(&1));
        assert_eq!(data.clusters[&merged_id].n_spikes, orig_spikes_0 + orig_spikes_1);

        // Split the merged cluster back into two new clusters
        let merged_indices = data.cluster_spike_indices(merged_id);
        let half = &merged_indices[..orig_spikes_0];
        let (in_id, out_id, _) = data.split(merged_id, half, &[merged_id]).unwrap();
        assert_eq!(data.clusters[&in_id].n_spikes, orig_spikes_0);
        assert_eq!(data.clusters[&out_id].n_spikes, orig_spikes_1);

        // Undo split -> back to merged_id
        let (_, sel_after_undo_split) = data.undo().unwrap();
        assert_eq!(sel_after_undo_split, Some(vec![merged_id]));
        assert!(data.clusters.contains_key(&merged_id));

        // Undo merge -> back to 0 and 1
        let (_, sel_after_undo_merge) = data.undo().unwrap();
        assert_eq!(sel_after_undo_merge, Some(vec![0, 1]));
        assert_eq!(data.clusters[&0].n_spikes, orig_spikes_0);
        assert_eq!(data.clusters[&1].n_spikes, orig_spikes_1);

        // Redo merge + label + custom label + save
        data.redo().unwrap();
        assert!(data.set_group(&[merged_id], ClusterGroup::Good, &[merged_id], &[2]));
        assert!(data.set_custom_label(&[merged_id], "brain_area", "V1"));
        data.save_to(&dir).unwrap();
        assert!(dir.join("spike_clusters.npy.bak").exists(), ".bak created on overwrite");
        assert_eq!(std::fs::read_to_string(&extra_file).unwrap(), "keep me");

        // Reopen and verify round-trip
        let reopened = SortingData::open(&dir, None).unwrap();
        assert_eq!(reopened.clusters[&merged_id].group, ClusterGroup::Good);
        assert_eq!(reopened.clusters[&merged_id].n_spikes, orig_spikes_0 + orig_spikes_1);
        assert_eq!(reopened.clusters[&merged_id].custom.get("brain_area").map(String::as_str), Some("V1"));
    }

    #[test]
    fn test_kilosort4_folder_loads_if_present() {
        let ks_dir = Path::new("/home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/data/kilosort4/saved_results");
        if !ks_dir.exists() {
            return;
        }
        let data = SortingData::open(ks_dir, None).expect("open Kilosort4 output");
        assert!(data.spike_times.len() > 100_000);
        assert!(data.clusters.len() > 200);
        assert_eq!(data.num_channels, 383);
        let sim = data.similar_to(0);
        assert!(!sim.is_empty());
        let ccg = data.compute_correlograms(&[0, 1], 1.0, 25.0, 2.0);
        assert_eq!(ccg.cells.len(), 2);
    }
}
