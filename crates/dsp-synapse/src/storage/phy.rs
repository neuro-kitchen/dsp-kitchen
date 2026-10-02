//! Phy / Kilosort Flat Folder Sorter Output Persistence (`phy.rs`).
//!
//! Saves and loads [`SortingOutput`] to/from a standard Phy/Kilosort directory:
//! - `spike_times.npy` (`u64` sample timestamps)
//! - `spike_clusters.npy` (`i32` cluster assignment)
//! - `amplitudes.npy` (`f32` peak amplitude/scaling)
//! - `spike_positions.npy` (`[N, 3]` f32 3D coordinates in $\mu\text{m}$)
//! - `templates.npy` (`[N_units, N_samples, N_channels]` f32 mean waveforms)
//! - `templates_std.npy` & `templates_se.npy`
//! - `channel_map.npy` & `channel_positions.npy`
//! - `cluster_group.tsv` (`cluster_id \t group`)
//! - `cluster_info.tsv` (full quality metrics table)
//! - `params.py` (`sample_rate`, `n_channels_dat`, `dtype`, `hp_filtered`)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use dsp_core::{DspError, DspResult, SensorLayout, SensorSite};
use crate::core::{SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate};
use super::npy::{
    read_npy_f32_1d, read_npy_f32_2d, read_npy_f32_3d, read_npy_i32_1d, read_npy_u64_1d,
    write_npy_f32_1d, write_npy_f32_2d, write_npy_f32_3d, write_npy_i32_1d, write_npy_u64_1d,
};

/// Saves a [`SortingOutput`] to a Phy/Kilosort directory.
pub fn save_phy_folder(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    std::fs::create_dir_all(dir).map_err(|e| DspError::Io(e.to_string()))?;

    let (samples, clusters, amps, locs) = sorting.flattened_spikes();
    write_npy_u64_1d(&dir.join("spike_times.npy"), &samples)?;
    write_npy_i32_1d(&dir.join("spike_clusters.npy"), &clusters)?;
    write_npy_f32_1d(&dir.join("amplitudes.npy"), &amps)?;

    if !locs.is_empty() {
        let flat_locs: Vec<f32> = locs.iter().flatten().copied().collect();
        write_npy_f32_2d(&dir.join("spike_positions.npy"), &flat_locs, [locs.len(), 3])?;
    }

    // Determine common template shape [N_units, N_samples, N_channels]
    let num_units = sorting.units.len();
    let num_channels = sorting
        .probe
        .as_ref()
        .map(|p| p.total_channels())
        .unwrap_or_else(|| {
            sorting
                .units
                .iter()
                .filter_map(|u| u.template.as_ref())
                .map(|t| t.num_channels)
                .max()
                .unwrap_or(1)
        });

    let template_samples = sorting
        .units
        .iter()
        .filter_map(|u| u.template.as_ref())
        .map(|t| t.num_samples)
        .max()
        .unwrap_or(0);

    if num_units > 0 && template_samples > 0 && num_channels > 0 {
        let total_t_elems = num_units * template_samples * num_channels;
        let mut t_mean = vec![0.0f32; total_t_elems];
        let mut t_std = vec![0.0f32; total_t_elems];
        let mut t_se = vec![0.0f32; total_t_elems];

        for (u_idx, unit) in sorting.units.iter().enumerate() {
            if let Some(t) = &unit.template {
                let off = u_idx * template_samples * num_channels;
                for (r, &ch) in t.channel_ids.iter().enumerate() {
                    if ch < num_channels {
                        for s in 0..t.num_samples.min(template_samples) {
                            let idx = off + s * num_channels + ch;
                            t_mean[idx] = t.row(r)[s];
                            t_std[idx] = t.std[r * t.num_samples + s];
                            t_se[idx] = t.se[r * t.num_samples + s];
                        }
                    }
                }
            }
        }

        let shape = [num_units, template_samples, num_channels];
        write_npy_f32_3d(&dir.join("templates.npy"), &t_mean, shape)?;
        write_npy_f32_3d(&dir.join("templates_std.npy"), &t_std, shape)?;
        write_npy_f32_3d(&dir.join("templates_se.npy"), &t_se, shape)?;
    }

    // Probe geometry: channel_map.npy & channel_positions.npy
    if let Some(probe) = &sorting.probe {
        let ch_ids: Vec<i32> = probe.contacts.iter().map(|c| c.channel_id as i32).collect();
        write_npy_i32_1d(&dir.join("channel_map.npy"), &ch_ids)?;
        let mut pos = Vec::with_capacity(probe.contacts.len() * 2);
        for c in &probe.contacts {
            pos.push(c.position.x_um);
            pos.push(c.position.y_um);
        }
        write_npy_f32_2d(&dir.join("channel_positions.npy"), &pos, [probe.contacts.len(), 2])?;
    }

    // cluster_group.tsv and cluster_info.tsv
    let mut grp_file = File::create(dir.join("cluster_group.tsv")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(grp_file, "cluster_id\tgroup").map_err(|e| DspError::Io(e.to_string()))?;
    for u in &sorting.units {
        let label_str = match u.quality_label {
            UnitQualityLabel::SingleUnit => "good",
            UnitQualityLabel::MultiUnit => "mua",
            UnitQualityLabel::Noise => "noise",
        };
        writeln!(grp_file, "{}\t{}", u.unit_id, label_str).map_err(|e| DspError::Io(e.to_string()))?;
    }

    let mut info_file = File::create(dir.join("cluster_info.tsv")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(
        info_file,
        "cluster_id\tch\tfiring_rate\tsnr\tisi_viol\tpresence_ratio\tamplitude_cutoff\tgroup"
    )
    .map_err(|e| DspError::Io(e.to_string()))?;
    for u in &sorting.units {
        let label_str = match u.quality_label {
            UnitQualityLabel::SingleUnit => "good",
            UnitQualityLabel::MultiUnit => "mua",
            UnitQualityLabel::Noise => "noise",
        };
        writeln!(
            info_file,
            "{}\t{}\t{:.2}\t{:.2}\t{:.4}\t{:.3}\t{:.3}\t{}",
            u.unit_id,
            u.primary_channel,
            u.firing_rate_hz,
            u.snr,
            u.isi_violation_ratio,
            u.presence_ratio,
            u.amplitude_cutoff,
            label_str
        )
        .map_err(|e| DspError::Io(e.to_string()))?;
    }

    // params.py
    let mut params_file = File::create(dir.join("params.py")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "# Generated by dsp-kitchen").map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "dat_path = 'recording.dat'").map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "n_channels_dat = {}", num_channels).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "dtype = 'int16'").map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "offset = 0").map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "sample_rate = {}", sorting.sample_rate_hz).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "hp_filtered = True").map_err(|e| DspError::Io(e.to_string()))?;

    Ok(())
}

/// Loads a Phy / Kilosort folder into a [`SortingOutput`].
pub fn load_phy_folder(dir: &Path) -> DspResult<SortingOutput> {
    if !dir.is_dir() {
        return Err(DspError::Io(format!("{} is not a directory", dir.display())));
    }

    let spike_times_path = dir.join("spike_times.npy");
    let spike_clusters_path = dir.join("spike_clusters.npy");
    if !spike_times_path.exists() || !spike_clusters_path.exists() {
        return Err(DspError::UnsupportedFormat(format!(
            "Phy folder {} missing spike_times.npy or spike_clusters.npy",
            dir.display()
        )));
    }

    let spike_times = read_npy_u64_1d(&spike_times_path)?;
    let spike_clusters = read_npy_i32_1d(&spike_clusters_path)?;
    let amplitudes = dir
        .join("amplitudes.npy")
        .exists()
        .then(|| read_npy_f32_1d(&dir.join("amplitudes.npy")).ok())
        .flatten()
        .unwrap_or_default();

    let spike_positions = dir
        .join("spike_positions.npy")
        .exists()
        .then(|| read_npy_f32_2d(&dir.join("spike_positions.npy")).ok())
        .flatten()
        .and_then(|(data, shape)| {
            if shape[1] == 3 {
                Some(
                    data.chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect::<Vec<[f32; 3]>>(),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();

    // Parse params.py for sample_rate
    let mut sample_rate_hz = 30000.0f64;
    if let Ok(params_file) = File::open(dir.join("params.py")) {
        for line in BufReader::new(params_file).lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.starts_with("sample_rate") {
                if let Some(val_str) = trimmed.split('=').nth(1) {
                    if let Ok(v) = val_str.trim().parse::<f64>() {
                        sample_rate_hz = v;
                    }
                }
            }
        }
    }

    // Optional cluster_group labels
    let mut labels: BTreeMap<usize, UnitQualityLabel> = BTreeMap::new();
    let group_path = if dir.join("cluster_group.tsv").exists() {
        dir.join("cluster_group.tsv")
    } else {
        dir.join("cluster_KSLabel.tsv")
    };
    if let Ok(f) = File::open(group_path) {
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            let mut parts = line.split('\t');
            if let (Some(id_str), Some(grp)) = (parts.next(), parts.next()) {
                if let Ok(id) = id_str.trim().parse::<usize>() {
                    let q = match grp.trim().to_ascii_lowercase().as_str() {
                        "good" => UnitQualityLabel::SingleUnit,
                        "mua" => UnitQualityLabel::MultiUnit,
                        _ => UnitQualityLabel::Noise,
                    };
                    labels.insert(id, q);
                }
            }
        }
    }

    // Optional templates [num_units, num_samples, num_channels]
    let loaded_templates = dir
        .join("templates.npy")
        .exists()
        .then(|| read_npy_f32_3d(&dir.join("templates.npy")).ok())
        .flatten();
    let loaded_std = dir
        .join("templates_std.npy")
        .exists()
        .then(|| read_npy_f32_3d(&dir.join("templates_std.npy")).ok())
        .flatten();
    let loaded_se = dir
        .join("templates_se.npy")
        .exists()
        .then(|| read_npy_f32_3d(&dir.join("templates_se.npy")).ok())
        .flatten();

    // Optional probe geometry
    let probe = if dir.join("channel_positions.npy").exists() {
        let (pos, shape) = read_npy_f32_2d(&dir.join("channel_positions.npy"))?;
        let contacts = (0..shape[0])
            .map(|ch| {
                let x = pos[ch * 2];
                let y = pos[ch * 2 + 1];
                SensorSite::new(ch, dsp_core::Position3D::new(x, y, 0.0), 0)
            })
            .collect();
        Some(SensorLayout::new("phy_probe", contacts))
    } else {
        None
    };

    let total_samples = spike_times.last().copied().unwrap_or(0);

    // Group spikes by cluster ID
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let n = spike_times.len().min(spike_clusters.len());
    for i in 0..n {
        let cl = spike_clusters[i];
        if cl >= 0 {
            groups.entry(cl as usize).or_default().push(i);
        }
    }

    let mut units = Vec::with_capacity(groups.len());
    for (unit_id, indices) in groups {
        let mut u_times = Vec::with_capacity(indices.len());
        let mut u_amps = Vec::with_capacity(indices.len());
        let mut u_locs = Vec::new();
        for &idx in &indices {
            u_times.push(spike_times[idx]);
            if let Some(&a) = amplitudes.get(idx) {
                u_amps.push(a);
            }
            if let Some(&l) = spike_positions.get(idx) {
                u_locs.push(l);
            }
        }

        // Reconstruct template if present
        let template = if let Some((t_data, [n_u, t_samples, n_c])) = &loaded_templates {
            if unit_id < *n_u {
                let off = unit_id * t_samples * n_c;
                let mut mean = vec![0.0f32; n_c * t_samples];
                let mut std = vec![1.0f32; n_c * t_samples];
                let mut se = vec![0.0f32; n_c * t_samples];
                for s in 0..*t_samples {
                    for c in 0..*n_c {
                        let src_idx = off + s * n_c + c;
                        let dst_idx = c * t_samples + s;
                        mean[dst_idx] = t_data[src_idx];
                        if let Some((std_data, _)) = &loaded_std {
                            std[dst_idx] = std_data[src_idx];
                        }
                        if let Some((se_data, _)) = &loaded_se {
                            se[dst_idx] = se_data[src_idx];
                        }
                    }
                }
                let mut t = WaveformTemplate::with_count((0..*n_c).collect(), *t_samples, indices.len(), mean, std);
                if loaded_se.is_some() {
                    t.se = se;
                }
                Some(t)
            } else {
                None
            }
        } else {
            None
        };

        let mut unit = SortedUnit::from_spikes(
            unit_id,
            0,
            u_times,
            u_amps,
            u_locs,
            template,
            sample_rate_hz,
            total_samples,
            10.0,
        );
        if let Some(&q) = labels.get(&unit_id) {
            unit.quality_label = q;
        }
        units.push(unit);
    }

    Ok(SortingOutput::new(
        "kilosort_phy",
        sample_rate_hz,
        total_samples,
        probe,
        units,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phy_folder_save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dsp_phy_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let times = vec![1000u64, 2500, 4200, 6800];
        let amps = vec![120.0f32, 135.0, 110.0, 140.0];
        let locs = vec![[10.0f32, 20.0, 15.0]; 4];
        let t_mean = vec![0.0f32; 2 * 30];
        let template = WaveformTemplate::with_count(vec![0, 1], 30, 4, t_mean.clone(), vec![1.0; 60]);

        let unit0 = SortedUnit::from_spikes(0, 0, times, amps, locs, Some(template), 30_000.0, 10_000, 8.0);
        let contacts = vec![
            SensorSite::new(0, dsp_core::Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, dsp_core::Position3D::new(0.0, 20.0, 0.0), 0),
        ];
        let probe = SensorLayout::new("2ch", contacts);
        let orig = SortingOutput::new("test_sorter", 30_000.0, 10_000, Some(probe), vec![unit0], None);

        save_phy_folder(&orig, &dir).unwrap();
        assert!(dir.join("spike_times.npy").exists());
        assert!(dir.join("spike_clusters.npy").exists());
        assert!(dir.join("templates.npy").exists());
        assert!(dir.join("params.py").exists());
        assert!(dir.join("cluster_info.tsv").exists());

        let loaded = load_phy_folder(&dir).unwrap();
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);
        assert_eq!(loaded.units[0].amplitudes_uv, orig.units[0].amplitudes_uv);
        assert_eq!(loaded.sample_rate_hz, 30_000.0);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
