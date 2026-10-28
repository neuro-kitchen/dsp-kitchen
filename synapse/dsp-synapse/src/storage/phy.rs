//! Phy / Kilosort folders ↔ [`SortingOutput`]. The folder itself (files, tables, `params.py`) is
//! read and written by [`dsp_io::neuro::phy::PhyFolder`]; this module converts and fills what the
//! files may lack (template similarity).

use std::collections::BTreeMap;
use std::path::Path;

use dsp_core::{DspError, DspResult};
use dsp_io::neuro::phy::{ClusterId, ClusterTables, PhyFolder, PhyParams};
use dsp_io::neuro::probe::SensorLayout;
use dsp_io::neuro::templates::{DenseTemplates, TemplateAxisOrder};

use crate::core::{
    dense_waveform, pack_templates, unpack_template, RecordingMeta, SortedUnit, SortingOutput,
    UnitQualityLabel, WaveformTemplate,
};
use crate::metrics::QualityCriteria;
use crate::sorting::similarity::compute_template_similarity_matrix;

/// Lag window (± samples) of the template similarity, as Kilosort's own.
const SIMILARITY_MAX_LAG: usize = 5;

/// `cluster_info.tsv` columns written for a [`SortingOutput`].
const INFO_COLUMNS: [&str; 9] =
    ["cluster_id", "ch", "firing_rate", "snr", "isi_viol", "presence_ratio", "amplitude_cutoff", "composite_score", "group"];

/// Computes `similar_templates` from the templates when the folder has none.
pub fn fill_similarity(folder: &mut PhyFolder) {
    if folder.similar_templates.is_some() {
        return;
    }
    folder.similar_templates = folder.templates.as_ref().filter(|t| t.count > 0).map(|t| {
        let waveforms: Vec<WaveformTemplate> = (0..t.count).map(|i| dense_waveform(t, i)).collect();
        compute_template_similarity_matrix(&waveforms, SIMILARITY_MAX_LAG)
    });
}

/// The folder form of a [`SortingOutput`]: templates densified on the probe's channels (slot =
/// unit id), labels as `cluster_group` and `cluster_KSLabel`, unit metrics in `cluster_info`,
/// similarity computed.
pub fn from_sorting_output(so: &SortingOutput) -> PhyFolder {
    let (spike_times, clusters, amplitudes, locations) = so.flattened_spikes();
    let spike_clusters: Vec<ClusterId> = clusters.iter().map(|&c| c.max(0) as ClusterId).collect();
    let channels = so.probe.as_ref().map(|p| p.total_channels()).unwrap_or_else(|| {
        so.units.iter().filter_map(|u| u.template.as_ref()).flat_map(|t| t.channel_ids.iter().copied()).max().map_or(0, |c| c + 1)
    });
    let (templates, templates_std, templates_se) = match pack_templates(
        so.units.iter().map(|u| (u.unit_id, u.template.as_ref())),
        channels,
        TemplateAxisOrder::SamplesChannels,
    ) {
        Some(([count, samples, ch], t_mean, t_std, t_se)) => {
            (Some(DenseTemplates { data: t_mean, count, samples, channels: ch }), Some(t_std), Some(t_se))
        }
        None => (None, None, None),
    };

    let group: BTreeMap<ClusterId, String> =
        so.units.iter().map(|u| (u.unit_id as ClusterId, u.quality_label.as_str().to_string())).collect();
    let info: BTreeMap<ClusterId, BTreeMap<String, String>> = so
        .units
        .iter()
        .map(|u| {
            let cells = [
                format!("{}", u.primary_channel),
                format!("{:.2}", u.firing_rate_hz),
                format!("{:.2}", u.snr),
                format!("{:.4}", u.isi_violation_ratio),
                format!("{:.3}", u.presence_ratio),
                format!("{:.3}", u.amplitude_cutoff),
                format!("{:.3}", u.composite_score),
                u.quality_label.as_str().to_string(),
            ];
            let row = INFO_COLUMNS[1..].iter().map(|c| c.to_string()).zip(cells).collect();
            (u.unit_id as ClusterId, row)
        })
        .collect();

    let (channel_map, channel_positions, channel_shanks) =
        so.probe.as_ref().map(SensorLayout::to_channel_arrays).unwrap_or_default();

    let m = &so.recording_meta;
    let params = PhyParams {
        sample_rate: so.sample_rate_hz,
        dat_path: m.dat_path.clone(),
        n_channels_dat: m.n_channels_dat.unwrap_or(channels.max(1)),
        dtype: m.dtype.clone(),
        offset: m.offset,
        hp_filtered: m.hp_filtered,
    };

    let mut folder = PhyFolder {
        params,
        spike_templates: spike_clusters.clone(),
        spike_clusters,
        spike_times,
        amplitudes,
        spike_positions: locations.iter().map(|l| [l[0], l[1]]).collect(),
        channel_map,
        channel_positions,
        channel_shanks,
        templates,
        templates_std,
        templates_se,
        tables: ClusterTables {
            ks_label: group.clone(),
            group,
            info,
            info_columns: INFO_COLUMNS.iter().map(|c| c.to_string()).collect(),
            ..Default::default()
        },
        ..Default::default()
    };
    folder.fill_channel_defaults();
    fill_similarity(&mut folder);
    folder
}

/// A folder as a unit-by-unit [`SortingOutput`] (stored metrics and labels win over computed ones).
/// The recording length is the binary recording's ([`PhyFolder::recording_samples`]) when found,
/// else just past the last spike.
pub fn to_sorting_output(folder: &PhyFolder) -> SortingOutput {
    let rate = folder.params.sample_rate;
    let past_last_spike = folder.spike_times.iter().max().map_or(0, |&t| t + 1);
    let total_samples = folder.recording_samples().map_or(past_last_spike, |n| n.max(past_last_spike));
    let mut by_cluster: BTreeMap<ClusterId, Vec<usize>> = BTreeMap::new();
    for (i, &c) in folder.spike_clusters.iter().enumerate() {
        by_cluster.entry(c).or_default().push(i);
    }
    let units = by_cluster
        .into_iter()
        .map(|(id, idx)| {
            let rep_tid = idx.first().and_then(|&i| folder.spike_templates.get(i).copied()).unwrap_or(id) as usize;
            let template = folder.templates.as_ref().and_then(|t| {
                let tid = if (id as usize) < t.count {
                    id as usize
                } else if rep_tid < t.count {
                    rep_tid
                } else {
                    return None;
                };
                unpack_template(
                    tid,
                    [t.count, t.samples, t.channels],
                    TemplateAxisOrder::SamplesChannels,
                    &t.data,
                    folder.templates_std.as_deref(),
                    folder.templates_se.as_deref(),
                    idx.len(),
                )
            });

            let info_row = folder.tables.info.get(&id);
            let primary_channel =
                info_row.and_then(|r| r.get("ch").or_else(|| r.get("chan"))).and_then(|v| v.parse::<usize>().ok());

            let times = idx.iter().map(|&i| folder.spike_times[i]).collect();
            let amps = idx.iter().filter_map(|&i| folder.amplitudes.get(i).copied()).collect();
            let locs = idx.iter().filter_map(|&i| folder.spike_positions.get(i)).map(|p| [p[0], p[1], 0.0]).collect();

            let mut unit = SortedUnit::from_spikes_with(
                id as usize,
                primary_channel,
                times,
                amps,
                locs,
                template,
                rate,
                total_samples,
                None,
                QualityCriteria::PHY,
            );

            if let Some(row) = info_row {
                if let Some(snr) = row.get("snr").and_then(|v| v.parse::<f32>().ok()) {
                    unit.snr = snr;
                }
                if let Some(fr) = row.get("firing_rate").or_else(|| row.get("fr")).and_then(|v| v.parse::<f64>().ok()) {
                    unit.firing_rate_hz = fr;
                }
                if let Some(isi) = row.get("isi_viol").and_then(|v| v.parse::<f64>().ok()) {
                    unit.isi_violation_ratio = isi;
                }
                if let Some(pr) = row.get("presence_ratio").and_then(|v| v.parse::<f64>().ok()) {
                    unit.presence_ratio = pr;
                }
                if let Some(ac) = row.get("amplitude_cutoff").and_then(|v| v.parse::<f64>().ok()) {
                    unit.amplitude_cutoff = ac;
                }
            }

            if let Some(g) = folder
                .tables
                .group
                .get(&id)
                .or_else(|| info_row.and_then(|r| r.get("group")))
                .or_else(|| folder.tables.ks_label.get(&id))
            {
                unit.quality_label = UnitQualityLabel::parse(g);
            }
            unit
        })
        .collect();

    let probe =
        SensorLayout::from_channel_arrays("phy_probe", &folder.channel_map, &folder.channel_positions, &folder.channel_shanks);

    let recording_meta = RecordingMeta {
        dat_path: folder.params.dat_path.clone(),
        dtype: folder.params.dtype.clone(),
        offset: folder.params.offset,
        hp_filtered: folder.params.hp_filtered,
        n_channels_dat: (folder.params.n_channels_dat > 0).then_some(folder.params.n_channels_dat),
    };

    SortingOutput::new("kilosort_phy", rate, total_samples, probe, units, None).with_recording_meta(recording_meta)
}

/// Saves a [`SortingOutput`] as a Phy / Kilosort folder.
///
/// # Errors
///
/// Fails ([`DspError::Io`]) when the path cannot be written.
pub fn save_phy_folder(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    from_sorting_output(sorting).write(dir)
}

/// Loads a Phy / Kilosort folder as a [`SortingOutput`].
///
/// # Errors
///
/// [`DspError::Io`] when `dir` is not a directory; fails when its files cannot be read.
pub fn load_phy_folder(dir: &Path) -> DspResult<SortingOutput> {
    if !dir.is_dir() {
        return Err(DspError::Io(format!("{} is not a directory", dir.display())));
    }
    Ok(to_sorting_output(&PhyFolder::read(dir)?))
}

/// Spikes of a sorting at `path`: a phy / Kilosort folder (or a file in one), else any format
/// [`super::load_sorting`] reads, in folder form; similarity filled when missing.
///
/// # Errors
///
/// As [`super::load_sorting`].
pub fn load_spikes(path: &Path) -> DspResult<PhyFolder> {
    let folder = if path.is_dir() { Some(path) } else { path.parent().filter(|_| path.is_file()) };
    let mut spikes = match folder.filter(|d| PhyFolder::is_phy_folder(d)) {
        Some(dir) => PhyFolder::read(dir)?,
        None => {
            let mut f = from_sorting_output(&super::load_sorting(path)?);
            f.folder = folder.map(Path::to_path_buf);
            f
        }
    };
    fill_similarity(&mut spikes);
    Ok(spikes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_io::container::npy::write_npy;
    use dsp_io::neuro::probe::{Position3D, SensorSite};
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dsp_synapse_phy_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Two units, templates peaking on channels 2 and 0.
    fn write_folder(dir: &Path) {
        write_npy(&dir.join("spike_times.npy"), &[10i64, 20, 30, 40, 55], &[5]).unwrap();
        write_npy(&dir.join("spike_clusters.npy"), &[0i32, 1, 0, 1, 1], &[5]).unwrap();
        let at = |t: usize, s: usize, c: usize| (t * 4 + s) * 3 + c;
        let mut t = vec![0.0f32; 2 * 4 * 3];
        t[at(0, 1, 2)] = -50.0;
        t[at(0, 2, 2)] = 10.0;
        t[at(1, 1, 0)] = -80.0;
        write_npy(&dir.join("templates.npy"), &t, &[2, 4, 3]).unwrap();
        std::fs::write(dir.join("params.py"), "sample_rate = 30000.0\nn_channels_dat = 3\n").unwrap();
    }

    #[test]
    fn sorting_output_round_trip_and_load_spikes() {
        let dir = scratch("convert");
        write_folder(&dir);
        let folder = PhyFolder::read(&dir).unwrap();
        let so = to_sorting_output(&folder);
        assert_eq!(so.units.len(), 2);
        let back = from_sorting_output(&so);
        assert_eq!(back.spike_times, folder.spike_times);
        assert_eq!(back.spike_clusters, folder.spike_clusters);
        assert_eq!(back.templates.as_ref().unwrap().best_channel(0), 2);
        let spikes = load_spikes(&dir.join("params.py")).unwrap();
        assert_eq!(spikes.spike_times, folder.spike_times, "a file inside the folder opens the folder");
        assert!(spikes.similar_templates.is_some(), "similarity computed when the file is missing");
        assert!((spikes.similarity(0, 0) - 1.0).abs() < 1e-5);
        assert!(load_spikes(&dir.join("missing")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_and_load_phy_folder() {
        let dir = scratch("save");
        let times = vec![1000u64, 2500, 4200, 6800];
        let amps = vec![120.0f32, 135.0, 110.0, 140.0];
        let locs = vec![[10.0f32, 20.0, 15.0]; 4];
        let template = WaveformTemplate::with_count(vec![0, 1], 30, 4, vec![0.0; 60], vec![1.0; 60]);
        let unit = SortedUnit::from_spikes(0, 0, times, amps, locs, Some(template), 30_000.0, 10_000, 8.0);
        let contacts = vec![
            SensorSite::new(0, Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, Position3D::new(0.0, 20.0, 0.0), 0),
        ];
        let orig = SortingOutput::new("test_sorter", 30_000.0, 10_000, Some(SensorLayout::new("2ch", contacts)), vec![unit], None);

        save_phy_folder(&orig, &dir).unwrap();
        for f in ["spike_times.npy", "spike_clusters.npy", "templates.npy", "params.py", "cluster_info.tsv", "cluster_group.tsv"] {
            assert!(dir.join(f).exists(), "{f}");
        }
        let loaded = load_phy_folder(&dir).unwrap();
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);
        assert_eq!(loaded.units[0].amplitudes_uv, orig.units[0].amplitudes_uv);
        assert_eq!(loaded.sample_rate_hz, 30_000.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
