//! A sorting as phy sees it: one row per spike (time, cluster, template, amplitude, position),
//! dense templates, template similarity, optional PC and template features, the cluster tables
//! (`cluster_group.tsv`, `cluster_KSLabel.tsv`, …, `cluster_info.tsv`) and `params.py`.
//!
//! [`PhySorting::load`] reads a phy / Kilosort 1–4 folder; [`load_spikes`] reads that or any
//! format [`super::load_sorting`] knows (`.sorting.zarr`, NWB `/units`), converted to this
//! spike-level form. [`PhySorting::save_curation`] writes back only what curation changes
//! (`spike_clusters.npy`, `cluster_group.tsv`, `cluster_info.tsv`), keeping a `.bak` of each the
//! first time.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use dsp_base::resampler::minmax::peak_to_peak;
use dsp_core::{DspError, DspResult, Position3D, SensorLayout, SensorSite};

use super::npy::{read_npy, write_npy, NpyArray, NpyElement};
use crate::core::{SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate};
use crate::sorting::similarity::compute_template_similarity_matrix;

/// Cluster (and template) ids as phy stores them.
pub type ClusterId = u32;

/// `params.py` of a phy folder.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhyParams {
    pub sample_rate: f64,
    /// As written (may be a path from another machine; see [`resolve_dat_path`]).
    pub dat_path: Option<String>,
    pub n_channels_dat: usize,
    pub dtype: Option<String>,
    pub offset: u64,
    pub hp_filtered: bool,
}

impl PhyParams {
    pub fn read(path: &Path) -> Self {
        let mut p = Self::default();
        let Ok(f) = File::open(path) else { return p };
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            let Some((k, v)) = line.split_once('=') else { continue };
            let v = v.split('#').next().unwrap_or(v).trim();
            // `'a.bin'`, `r"C:\a.bin"` or `['a.bin']`
            let text = || {
                let v = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(v).trim();
                let v = v.strip_prefix('r').filter(|s| s.starts_with(['\'', '"'])).unwrap_or(v);
                v.trim_matches(|c| c == '\'' || c == '"').to_string()
            };
            match k.trim() {
                "sample_rate" => p.sample_rate = v.parse().unwrap_or(0.0),
                "n_channels_dat" => p.n_channels_dat = v.parse().unwrap_or(0),
                "offset" => p.offset = v.parse().unwrap_or(0),
                "hp_filtered" => p.hp_filtered = v == "True",
                "dtype" => p.dtype = Some(text()).filter(|s| !s.is_empty()),
                "dat_path" => p.dat_path = Some(text()).filter(|s| !s.is_empty()),
                _ => {}
            }
        }
        p
    }
}

/// The recording `dat_path` names: as written, relative to the folder, else a file of that name in
/// the folder or up to three folders above it (paths written on another machine, e.g. Windows).
pub fn resolve_dat_path(sorting_dir: &Path, dat_path: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(dat_path);
    if direct.is_file() {
        return Some(direct);
    }
    let relative = sorting_dir.join(dat_path);
    if relative.is_file() {
        return Some(relative);
    }
    let name = dat_path.rsplit(['/', '\\']).next().filter(|n| !n.is_empty())?;
    sorting_dir.ancestors().take(4).map(|d| d.join(name)).find(|p| p.is_file())
}

/// Dense templates `[count, samples, channels]` (C order), as `templates.npy`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhyTemplates {
    pub data: Vec<f32>,
    pub count: usize,
    pub samples: usize,
    pub channels: usize,
}

impl PhyTemplates {
    /// Template `t` on `channel` over time.
    pub fn trace(&self, t: usize, channel: usize) -> Vec<f32> {
        (0..self.samples).map(|s| self.data[(t * self.samples + s) * self.channels + channel]).collect()
    }

    /// Peak-to-peak of template `t` on each channel.
    pub fn peak_to_peak(&self, t: usize) -> Vec<f32> {
        if t >= self.count {
            return vec![0.0; self.channels];
        }
        (0..self.channels).map(|c| peak_to_peak(&self.trace(t, c))).collect()
    }

    /// Channel where template `t` is largest (peak-to-peak).
    pub fn best_channel(&self, t: usize) -> usize {
        self.top_channels(t, 1).first().copied().unwrap_or(0)
    }

    /// The `k` channels where template `t` is largest, largest first.
    pub fn top_channels(&self, t: usize, k: usize) -> Vec<usize> {
        let ptp = self.peak_to_peak(t);
        let mut order: Vec<usize> = (0..self.channels).collect();
        order.sort_by(|&a, &b| ptp[b].total_cmp(&ptp[a]).then(a.cmp(&b)));
        order.truncate(k.min(self.channels));
        order
    }

    /// Largest peak-to-peak of template `t` over its channels.
    pub fn amplitude(&self, t: usize) -> f32 {
        self.peak_to_peak(t).into_iter().fold(0.0, f32::max)
    }

    /// Template `t` as a [`WaveformTemplate`] on the channels where it is not zero.
    pub fn waveform(&self, t: usize) -> WaveformTemplate {
        let channels: Vec<usize> = (0..self.channels).filter(|&c| self.trace(t, c).iter().any(|v| *v != 0.0)).collect();
        let mean: Vec<f32> = channels.iter().flat_map(|&c| self.trace(t, c)).collect();
        let std = vec![0.0; mean.len()];
        WaveformTemplate::new(channels, self.samples, mean, std)
    }
}

/// The cluster tables of a phy folder, by cluster id (values as written).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClusterTables {
    /// `cluster_group.tsv`: good / mua / noise / unsorted (curated).
    pub group: BTreeMap<ClusterId, String>,
    /// `cluster_KSLabel.tsv`: the sorter's own label.
    pub ks_label: BTreeMap<ClusterId, String>,
    /// `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv`.
    pub amplitude: BTreeMap<ClusterId, f32>,
    pub contam_pct: BTreeMap<ClusterId, f32>,
    /// `cluster_info.tsv`: every column of every row, and the column order.
    pub info: BTreeMap<ClusterId, BTreeMap<String, String>>,
    pub info_columns: Vec<String>,
}

impl ClusterTables {
    fn read(dir: &Path) -> Self {
        let number = |m: BTreeMap<ClusterId, String>| m.into_iter().filter_map(|(k, v)| Some((k, v.parse().ok()?))).collect();
        let (info, info_columns) = read_table(&dir.join("cluster_info.tsv"));
        Self {
            group: read_two_columns(&dir.join("cluster_group.tsv")),
            ks_label: read_two_columns(&dir.join("cluster_KSLabel.tsv")),
            amplitude: number(read_two_columns(&dir.join("cluster_Amplitude.tsv"))),
            contam_pct: number(read_two_columns(&dir.join("cluster_ContamPct.tsv"))),
            info,
            info_columns,
        }
    }
}

/// `cluster_id \t value` rows (header skipped).
fn read_two_columns(path: &Path) -> BTreeMap<ClusterId, String> {
    let Ok(f) = File::open(path) else { return BTreeMap::new() };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| {
            let (id, value) = line.split_once('\t')?;
            Some((id.trim().parse().ok()?, value.trim().to_string()))
        })
        .collect()
}

/// A tab-separated table keyed by its `cluster_id` (or `id`) column, and its column names.
fn read_table(path: &Path) -> (BTreeMap<ClusterId, BTreeMap<String, String>>, Vec<String>) {
    let Ok(f) = File::open(path) else { return Default::default() };
    let mut lines = BufReader::new(f).lines().map_while(Result::ok);
    let Some(header) = lines.next() else { return Default::default() };
    let columns: Vec<String> = header.split('\t').map(|s| s.trim().to_string()).collect();
    let Some(id_col) = columns.iter().position(|h| h == "cluster_id" || h == "id") else { return Default::default() };
    let rows = lines
        .filter_map(|line| {
            let cells: Vec<&str> = line.split('\t').collect();
            let id = cells.get(id_col)?.trim().parse().ok()?;
            let row = columns.iter().zip(&cells).filter(|(_, v)| !v.trim().is_empty()).map(|(k, v)| (k.clone(), v.trim().to_string())).collect();
            Some((id, row))
        })
        .collect();
    (rows, columns)
}

/// An optional array of the folder, read as `T` (absent or unreadable: `None`, with a warning for
/// unreadable ones).
fn optional<T: NpyElement>(dir: &Path, name: &str) -> Option<NpyArray<T>> {
    let path = dir.join(name);
    if !path.exists() {
        return None;
    }
    read_npy::<T>(&path).map_err(|e| tracing::warn!("ignoring {name}: {e}")).ok()
}

/// Rows of an `[n, ≥2]` array as `[x, y]` (first two columns).
fn xy_rows(a: &NpyArray<f32>) -> Option<Vec<[f32; 2]>> {
    let cols = *a.shape.get(1)?;
    (a.shape.len() == 2 && cols >= 2).then(|| a.data.chunks_exact(cols).map(|r| [r[0], r[1]]).collect())
}

/// A whole sorting, spike by spike. See the module docs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhySorting {
    /// Folder it was read from (phy folders).
    pub folder: Option<PathBuf>,
    pub params: PhyParams,
    pub spike_times: Vec<u64>,
    pub spike_clusters: Vec<ClusterId>,
    /// Template each spike was matched with (its cluster when the sorter saved none).
    pub spike_templates: Vec<ClusterId>,
    pub amplitudes: Vec<f32>,
    /// `[x, y]` µm per spike (empty when not saved).
    pub spike_positions: Vec<[f32; 2]>,
    /// Recording channel of each sorted channel.
    pub channel_map: Vec<usize>,
    /// `[x, y]` µm per sorted channel.
    pub channel_positions: Vec<[f32; 2]>,
    pub channel_shanks: Vec<usize>,
    pub templates: Option<PhyTemplates>,
    pub templates_std: Option<Vec<f32>>,
    pub templates_se: Option<Vec<f32>>,
    /// `[templates, templates]` similarity (as `similar_templates.npy`, else computed from the
    /// templates).
    pub similar_templates: Option<Vec<f32>>,
    /// `pc_features.npy` `[spikes, pcs, channels]` and `pc_feature_ind.npy` `[templates, channels]`.
    pub pc_features: Option<(Vec<f32>, [usize; 3])>,
    pub pc_feature_ind: Option<(Vec<usize>, [usize; 2])>,
    /// `template_features.npy` `[spikes, k]` and `template_feature_ind.npy` `[templates, k]`.
    pub template_features: Option<(Vec<f32>, [usize; 2])>,
    pub template_feature_ind: Option<(Vec<usize>, [usize; 2])>,
    pub tables: ClusterTables,
}

impl PhySorting {
    /// Reads a phy / Kilosort 1–4 folder (`spike_times.npy` and `spike_clusters.npy` required).
    pub fn load(dir: &Path) -> DspResult<Self> {
        let need = |name: &str| {
            let p = dir.join(name);
            if p.exists() { Ok(p) } else { Err(DspError::UnsupportedFormat(format!("{} has no {name}: not a phy / Kilosort folder", dir.display()))) }
        };
        let spike_times = read_npy::<u64>(&need("spike_times.npy")?)?.data;
        let clusters = read_npy::<i64>(&need("spike_clusters.npy")?)?.data;
        if let Some(bad) = clusters.iter().find(|&&c| c < 0) {
            return Err(DspError::UnsupportedFormat(format!("{}: negative cluster id {bad} in spike_clusters.npy", dir.display())));
        }
        let n = spike_times.len().min(clusters.len());
        let spike_clusters: Vec<ClusterId> = clusters[..n].iter().map(|&c| c as ClusterId).collect();
        let spike_times = spike_times[..n].to_vec();
        let per_spike = |v: Option<Vec<ClusterId>>| v.filter(|v| v.len() >= n).map(|mut v| { v.truncate(n); v });
        let spike_templates = per_spike(optional::<ClusterId>(dir, "spike_templates.npy").map(|a| a.data)).unwrap_or_else(|| spike_clusters.clone());
        let amplitudes = optional::<f32>(dir, "amplitudes.npy").map(|a| a.data).filter(|a| a.len() >= n).map(|mut a| { a.truncate(n); a }).unwrap_or_default();
        let spike_positions = optional::<f32>(dir, "spike_positions.npy").as_ref().and_then(xy_rows).filter(|p| p.len() >= n).map(|mut p| { p.truncate(n); p }).unwrap_or_default();

        let params = PhyParams::read(&dir.join("params.py"));
        let channel_positions = optional::<f32>(dir, "channel_positions.npy").as_ref().and_then(xy_rows).unwrap_or_default();
        let channel_map = optional::<usize>(dir, "channel_map.npy").map(|a| a.data).unwrap_or_default();
        let channel_shanks = optional::<usize>(dir, "channel_shanks.npy").map(|a| a.data).unwrap_or_default();
        let channels = channel_positions.len().max(channel_map.len());

        let templates = optional::<f32>(dir, "templates.npy").filter(|a| a.shape.len() == 3).map(|a| {
            let (count, samples, stored) = (a.shape[0], a.shape[1], a.shape[2]);
            // Sparse templates (Kilosort 1–3): `templates_ind.npy` names each stored column's channel
            let ind = optional::<usize>(dir, "templates_ind.npy").filter(|i| i.shape == [count, stored]);
            match ind {
                Some(ind) if stored < channels => {
                    let mut dense = vec![0.0f32; count * samples * channels];
                    for t in 0..count {
                        for k in 0..stored {
                            let ch = ind.data[t * stored + k];
                            if ch < channels {
                                for s in 0..samples {
                                    dense[(t * samples + s) * channels + ch] = a.data[(t * samples + s) * stored + k];
                                }
                            }
                        }
                    }
                    PhyTemplates { data: dense, count, samples, channels }
                }
                _ => PhyTemplates { data: a.data, count, samples, channels: stored },
            }
        });
        let same_shape = |name: &str| optional::<f32>(dir, name).filter(|a| templates.as_ref().is_some_and(|t| a.data.len() == t.data.len())).map(|a| a.data);
        let (templates_std, templates_se) = (same_shape("templates_std.npy"), same_shape("templates_se.npy"));
        let similar_templates = optional::<f32>(dir, "similar_templates.npy")
            .filter(|a| templates.as_ref().is_some_and(|t| a.shape == [t.count, t.count]))
            .map(|a| a.data);
        let three = |a: NpyArray<f32>| (a.shape.len() == 3).then(|| (a.data, [a.shape[0], a.shape[1], a.shape[2]]));
        let two = |a: NpyArray<f32>| (a.shape.len() == 2).then(|| (a.data, [a.shape[0], a.shape[1]]));
        let two_ind = |a: NpyArray<usize>| (a.shape.len() == 2).then(|| (a.data, [a.shape[0], a.shape[1]]));

        let mut sorting = Self {
            folder: Some(dir.to_path_buf()),
            params,
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            spike_positions,
            channel_map,
            channel_positions,
            channel_shanks,
            templates,
            templates_std,
            templates_se,
            similar_templates,
            pc_features: optional::<f32>(dir, "pc_features.npy").and_then(three),
            pc_feature_ind: optional::<usize>(dir, "pc_feature_ind.npy").and_then(two_ind),
            template_features: optional::<f32>(dir, "template_features.npy").and_then(two),
            template_feature_ind: optional::<usize>(dir, "template_feature_ind.npy").and_then(two_ind),
            tables: ClusterTables::read(dir),
        };
        sorting.fill_defaults();
        Ok(sorting)
    }

    /// The spike-level form of a [`SortingOutput`] (templates densified on the probe's channels;
    /// labels as `cluster_group` values).
    pub fn from_sorting_output(so: &SortingOutput) -> Self {
        let (spike_times, clusters, amplitudes, locations) = so.flattened_spikes();
        let spike_clusters: Vec<ClusterId> = clusters.iter().map(|&c| c.max(0) as ClusterId).collect();
        let channels = so.probe.as_ref().map(|p| p.total_channels()).unwrap_or_else(|| {
            so.units.iter().filter_map(|u| u.template.as_ref()).flat_map(|t| t.channel_ids.iter().copied()).max().map_or(0, |c| c + 1)
        });
        let samples = so.units.iter().filter_map(|u| u.template.as_ref()).map(|t| t.num_samples).max().unwrap_or(0);
        let count = so.units.iter().map(|u| u.unit_id + 1).max().unwrap_or(0);
        let templates = (samples > 0 && channels > 0).then(|| {
            let mut data = vec![0.0f32; count * samples * channels];
            for u in &so.units {
                let Some(t) = &u.template else { continue };
                for (r, &ch) in t.channel_ids.iter().enumerate().filter(|(_, c)| **c < channels) {
                    for (s, &v) in t.row(r).iter().enumerate().take(samples) {
                        data[(u.unit_id * samples + s) * channels + ch] = v;
                    }
                }
            }
            PhyTemplates { data, count, samples, channels }
        });
        let label = |q: UnitQualityLabel| match q {
            UnitQualityLabel::SingleUnit => "good",
            UnitQualityLabel::MultiUnit => "mua",
            UnitQualityLabel::Noise => "noise",
        };
        let group: BTreeMap<ClusterId, String> = so.units.iter().map(|u| (u.unit_id as ClusterId, label(u.quality_label).to_string())).collect();
        let sites = so.probe.as_ref().map(|p| p.sites().to_vec()).unwrap_or_default();
        let mut sorting = Self {
            params: PhyParams { sample_rate: so.sample_rate_hz, ..Default::default() },
            spike_templates: spike_clusters.clone(),
            spike_clusters,
            spike_times,
            amplitudes,
            spike_positions: locations.iter().map(|l| [l[0], l[1]]).collect(),
            channel_map: sites.iter().map(|s| s.channel_id).collect(),
            channel_positions: sites.iter().map(|s| [s.position.x_um, s.position.y_um]).collect(),
            channel_shanks: sites.iter().map(|s| s.shank_id).collect(),
            templates,
            tables: ClusterTables { ks_label: group.clone(), group, ..Default::default() },
            ..Default::default()
        };
        sorting.fill_defaults();
        sorting
    }

    /// Identity channel map, zero shanks, similarity from the templates, where missing.
    fn fill_defaults(&mut self) {
        let channels = self.channels();
        if self.channel_map.len() != channels {
            self.channel_map = (0..channels).collect();
        }
        if self.channel_shanks.len() != channels {
            self.channel_shanks = vec![0; channels];
        }
        if self.similar_templates.is_none() {
            self.similar_templates = self.templates.as_ref().filter(|t| t.count > 0).map(|t| {
                let waveforms: Vec<WaveformTemplate> = (0..t.count).map(|i| t.waveform(i)).collect();
                // Lag-maximized (±5 samples), as Kilosort's own similarity
                compute_template_similarity_matrix(&waveforms, 5)
            });
        }
    }

    /// Sorted channels (templates, positions or channel map, whichever says).
    pub fn channels(&self) -> usize {
        self.templates.as_ref().map_or(0, |t| t.channels).max(self.channel_positions.len()).max(self.channel_map.len())
    }

    pub fn sample_rate(&self) -> f64 {
        self.params.sample_rate
    }

    /// The recording `params.py` names, if it can be found.
    pub fn recording_path(&self) -> Option<PathBuf> {
        resolve_dat_path(self.folder.as_deref()?, self.params.dat_path.as_deref()?)
    }

    /// Similarity of templates `a` and `b` (0 without templates).
    pub fn similarity(&self, a: usize, b: usize) -> f32 {
        let n = self.templates.as_ref().map_or(0, |t| t.count);
        match &self.similar_templates {
            Some(s) if a < n && b < n => s[a * n + b],
            _ => 0.0,
        }
    }

    /// Back to a unit-by-unit [`SortingOutput`] (`unsorted` and unknown groups become noise, as
    /// before).
    pub fn to_sorting_output(&self) -> SortingOutput {
        let rate = self.params.sample_rate;
        let total_samples = self.spike_times.iter().copied().max().unwrap_or(0);
        let mut by_cluster: BTreeMap<ClusterId, Vec<usize>> = BTreeMap::new();
        for (i, &c) in self.spike_clusters.iter().enumerate() {
            by_cluster.entry(c).or_default().push(i);
        }
        let units = by_cluster
            .into_iter()
            .map(|(id, idx)| {
                let template = self.templates.as_ref().filter(|t| (id as usize) < t.count).map(|t| {
                    let tid = id as usize;
                    let channels: Vec<usize> = (0..t.channels).collect();
                    let at = |data: &[f32], c: usize, s: usize| data[(tid * t.samples + s) * t.channels + c];
                    let mean: Vec<f32> = channels.iter().flat_map(|&c| (0..t.samples).map(move |s| (c, s))).map(|(c, s)| at(&t.data, c, s)).collect();
                    let std = match &self.templates_std {
                        Some(sd) => channels.iter().flat_map(|&c| (0..t.samples).map(move |s| (c, s))).map(|(c, s)| at(sd, c, s)).collect(),
                        None => vec![1.0; mean.len()],
                    };
                    let mut w = WaveformTemplate::with_count(channels.clone(), t.samples, idx.len(), mean, std);
                    if let Some(se) = &self.templates_se {
                        w.se = channels.iter().flat_map(|&c| (0..t.samples).map(move |s| (c, s))).map(|(c, s)| at(se, c, s)).collect();
                    }
                    w
                });
                let times = idx.iter().map(|&i| self.spike_times[i]).collect();
                let amps = idx.iter().filter_map(|&i| self.amplitudes.get(i).copied()).collect();
                let locs = idx.iter().filter_map(|&i| self.spike_positions.get(i)).map(|p| [p[0], p[1], 0.0]).collect();
                let mut unit = SortedUnit::from_spikes(id as usize, 0, times, amps, locs, template, rate, total_samples, 10.0);
                if let Some(g) = self.tables.group.get(&id).or_else(|| self.tables.ks_label.get(&id)) {
                    unit.quality_label = match g.to_ascii_lowercase().as_str() {
                        "good" => UnitQualityLabel::SingleUnit,
                        "mua" => UnitQualityLabel::MultiUnit,
                        _ => UnitQualityLabel::Noise,
                    };
                }
                unit
            })
            .collect();
        let probe = (!self.channel_positions.is_empty()).then(|| {
            let contacts = self
                .channel_positions
                .iter()
                .enumerate()
                .map(|(i, p)| SensorSite::new(self.channel_map.get(i).copied().unwrap_or(i), Position3D::new(p[0], p[1], 0.0), self.channel_shanks.get(i).copied().unwrap_or(0)))
                .collect();
            SensorLayout::new("phy_probe", contacts)
        });
        SortingOutput::new("kilosort_phy", rate, total_samples, probe, units, None)
    }

    /// Writes curation results into `dir`: `spike_clusters.npy` (from `spike_clusters`),
    /// `cluster_group.tsv` (from `groups`) and `cluster_info.tsv` (`columns`, then one row per
    /// entry of `rows`). The first time a file is replaced its original is kept as `<name>.bak`;
    /// nothing else in the folder is touched.
    pub fn save_curation(
        dir: &Path,
        spike_clusters: &[ClusterId],
        groups: &BTreeMap<ClusterId, String>,
        columns: &[String],
        rows: &BTreeMap<ClusterId, BTreeMap<String, String>>,
    ) -> DspResult<()> {
        let io = |e: std::io::Error| DspError::Io(format!("{}: {e}", dir.display()));
        std::fs::create_dir_all(dir).map_err(io)?;
        for name in ["spike_clusters.npy", "cluster_group.tsv", "cluster_info.tsv"] {
            let (file, bak) = (dir.join(name), dir.join(format!("{name}.bak")));
            if file.exists() && !bak.exists() {
                std::fs::copy(&file, &bak).map_err(io)?;
            }
        }
        let clusters: Vec<i32> = spike_clusters.iter().map(|&c| c as i32).collect();
        write_npy(&dir.join("spike_clusters.npy"), &clusters, &[clusters.len()])?;

        let mut f = File::create(dir.join("cluster_group.tsv")).map_err(io)?;
        writeln!(f, "cluster_id\tgroup").map_err(io)?;
        for (id, g) in groups {
            writeln!(f, "{id}\t{g}").map_err(io)?;
        }

        let mut f = File::create(dir.join("cluster_info.tsv")).map_err(io)?;
        writeln!(f, "{}", columns.join("\t")).map_err(io)?;
        for (id, row) in rows {
            let cells: Vec<String> = columns
                .iter()
                .map(|c| if c == "cluster_id" || c == "id" { id.to_string() } else { row.get(c).cloned().unwrap_or_default() })
                .collect();
            writeln!(f, "{}", cells.join("\t")).map_err(io)?;
        }
        Ok(())
    }
}

/// Spikes of a sorting at `path`: a phy / Kilosort folder (or a file in one), else any format
/// [`super::load_sorting`] reads.
pub fn load_spikes(path: &Path) -> DspResult<PhySorting> {
    let folder = if path.is_dir() { Some(path) } else { path.parent().filter(|_| path.is_file()) };
    if let Some(dir) = folder.filter(|d| d.join("spike_times.npy").exists()) {
        return PhySorting::load(dir);
    }
    Ok(PhySorting::from_sorting_output(&super::load_sorting(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dsp_physorting_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A Kilosort4-like folder: int64 times, 2-column positions, float64 channel positions.
    fn write_folder(dir: &Path) {
        write_npy(&dir.join("spike_times.npy"), &[10i64, 20, 30, 40, 55], &[5]).unwrap();
        write_npy(&dir.join("spike_clusters.npy"), &[0i32, 1, 0, 1, 1], &[5]).unwrap();
        write_npy(&dir.join("spike_templates.npy"), &[0u32, 1, 0, 1, 1], &[5]).unwrap();
        write_npy(&dir.join("amplitudes.npy"), &[1.0f64, 2.0, 1.5, 2.5, 3.0], &[5]).unwrap();
        write_npy(&dir.join("spike_positions.npy"), &[0.0f32, 10.0, 0.0, 30.0, 0.0, 10.0, 0.0, 30.0, 0.0, 30.0], &[5, 2]).unwrap();
        write_npy(&dir.join("channel_positions.npy"), &[0.0f64, 0.0, 0.0, 20.0, 0.0, 40.0], &[3, 2]).unwrap();
        write_npy(&dir.join("channel_map.npy"), &[0i64, 1, 2], &[3]).unwrap();
        // Template 0 peaks on channel 2, template 1 on channel 0
        // [template, sample, channel] in a 2 × 4 × 3 array
        let at = |t: usize, s: usize, c: usize| (t * 4 + s) * 3 + c;
        let mut t = vec![0.0f32; 2 * 4 * 3];
        t[at(0, 1, 2)] = -50.0;
        t[at(0, 2, 2)] = 10.0;
        t[at(1, 1, 0)] = -80.0;
        write_npy(&dir.join("templates.npy"), &t, &[2, 4, 3]).unwrap();
        std::fs::write(dir.join("params.py"), "dat_path = 'C:/Users/x/rec.bin'\nn_channels_dat = 3\ndtype = 'int16'\noffset = 0\nsample_rate = 30000.0\nhp_filtered = False\n").unwrap();
        std::fs::write(dir.join("cluster_KSLabel.tsv"), "cluster_id\tKSLabel\n0\tgood\n1\tmua\n").unwrap();
        std::fs::write(dir.join("cluster_ContamPct.tsv"), "cluster_id\tContamPct\n0\t1.5\n1\t20.0\n").unwrap();
        std::fs::write(dir.join("cluster_info.tsv"), "cluster_id\tch\tquality\n0\t2\tnice\n1\t0\t\n").unwrap();
        std::fs::write(dir.join("rec.bin"), [0u8; 4]).unwrap();
    }

    #[test]
    fn test_loads_a_kilosort4_like_folder() {
        let dir = scratch("load");
        write_folder(&dir);
        let s = PhySorting::load(&dir).unwrap();
        assert_eq!(s.spike_times, vec![10, 20, 30, 40, 55]);
        assert_eq!(s.spike_clusters, vec![0, 1, 0, 1, 1]);
        assert_eq!(s.amplitudes, vec![1.0, 2.0, 1.5, 2.5, 3.0]);
        assert_eq!(s.spike_positions[1], [0.0, 30.0], "two-column positions are kept");
        assert_eq!(s.channel_positions, vec![[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]]);
        assert_eq!(s.params.sample_rate, 30_000.0);
        assert_eq!(s.params.dat_path.as_deref(), Some("C:/Users/x/rec.bin"));
        assert_eq!(s.recording_path(), Some(dir.join("rec.bin")), "a Windows path falls back to the file's name");
        let t = s.templates.as_ref().unwrap();
        assert_eq!((t.best_channel(0), t.best_channel(1)), (2, 0));
        assert_eq!(t.top_channels(0, 2)[0], 2);
        assert_eq!(t.amplitude(0), 60.0);
        assert!(s.similar_templates.is_some(), "similarity computed when the file is missing");
        assert!((s.similarity(0, 0) - 1.0).abs() < 1e-5);
        assert_eq!(s.tables.ks_label[&1], "mua");
        assert_eq!(s.tables.contam_pct[&1], 20.0);
        assert_eq!(s.tables.info[&0]["quality"], "nice");
        assert_eq!(s.tables.info_columns, vec!["cluster_id", "ch", "quality"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_save_curation_touches_three_files_and_keeps_backups() {
        let dir = scratch("save");
        write_folder(&dir);
        let groups = BTreeMap::from([(0, "good".to_string()), (2, "noise".to_string())]);
        let columns = vec!["cluster_id".to_string(), "group".to_string()];
        let rows = BTreeMap::from([(0, BTreeMap::from([("group".to_string(), "good".to_string())])), (2, BTreeMap::new())]);
        PhySorting::save_curation(&dir, &[0, 2, 0, 2, 2], &groups, &columns, &rows).unwrap();
        let s = PhySorting::load(&dir).unwrap();
        assert_eq!(s.spike_clusters, vec![0, 2, 0, 2, 2]);
        assert_eq!(s.tables.group[&2], "noise");
        assert_eq!(s.tables.info[&0]["group"], "good");
        assert!(dir.join("spike_clusters.npy.bak").exists() && dir.join("cluster_info.tsv.bak").exists());
        assert!(!dir.join("cluster_group.tsv.bak").exists(), "no original to keep");
        // A second save keeps the first backups (the originals)
        PhySorting::save_curation(&dir, &[1, 1, 1, 1, 1], &groups, &columns, &rows).unwrap();
        let original = read_npy::<i32>(&dir.join("spike_clusters.npy.bak")).unwrap().data;
        assert_eq!(original, vec![0, 1, 0, 1, 1]);
        assert!(dir.join("amplitudes.npy").exists() && !dir.join("amplitudes.npy.bak").exists(), "other files untouched");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_sorting_output_round_trip_and_load_spikes() {
        let dir = scratch("convert");
        write_folder(&dir);
        let s = PhySorting::load(&dir).unwrap();
        let so = s.to_sorting_output();
        assert_eq!(so.units.len(), 2);
        let back = PhySorting::from_sorting_output(&so);
        assert_eq!(back.spike_times, s.spike_times);
        assert_eq!(back.spike_clusters, s.spike_clusters);
        assert_eq!(back.templates.as_ref().unwrap().best_channel(0), 2);
        assert_eq!(load_spikes(&dir.join("params.py")).unwrap().spike_times, s.spike_times, "a file inside the folder opens the folder");
        assert!(load_spikes(&dir.join("missing")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_rejects_folders_without_spikes() {
        let dir = scratch("empty");
        assert!(PhySorting::load(&dir).unwrap_err().to_string().contains("spike_times.npy"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The user's Kilosort4 output, when present: `DSP_KS4_FOLDER=<saved_results>`.
    #[test]
    #[ignore]
    fn test_real_kilosort4_folder() {
        let Some(dir) = std::env::var_os("DSP_KS4_FOLDER") else { return };
        let s = PhySorting::load(Path::new(&dir)).unwrap();
        let t = s.templates.as_ref().unwrap();
        println!("{} spikes, {} templates × {} samples × {} channels, {} clusters, recording {:?}", s.spike_times.len(), t.count, t.samples, t.channels, s.tables.ks_label.len(), s.recording_path());
        assert_eq!(s.spike_times.len(), s.spike_clusters.len());
        assert!(s.similar_templates.is_some());
    }
}
