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

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dsp_core::{DspError, DspResult};
use crate::core::{SortingOutput, UnitQualityLabel};
use super::npy::{write_npy_f32_1d, write_npy_f32_2d, write_npy_f32_3d, write_npy_i32_1d, write_npy_u64_1d};
use super::phy_sorting::PhySorting;

/// Saves a [`SortingOutput`] to a Phy/Kilosort directory.
pub fn save_phy_folder(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    std::fs::create_dir_all(dir).map_err(|e| DspError::Io(e.to_string()))?;

    let (samples, clusters, amps, locs) = sorting.flattened_spikes();
    write_npy_u64_1d(&dir.join("spike_times.npy"), &samples)?;
    write_npy_i32_1d(&dir.join("spike_clusters.npy"), &clusters)?;
    write_npy_i32_1d(&dir.join("spike_templates.npy"), &clusters)?;
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

        // Compute and write similar_templates.npy [num_units, num_units]
        let mut sim_matrix = vec![0.0f32; num_units * num_units];
        let t_len = template_samples * num_channels;
        let mut norms = vec![0.0f32; num_units];
        for u in 0..num_units {
            let u_slice = &t_mean[u * t_len..(u + 1) * t_len];
            let norm = u_slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
            norms[u] = if norm > 1e-9 { norm } else { 1.0 };
        }
        for u1 in 0..num_units {
            let s1 = &t_mean[u1 * t_len..(u1 + 1) * t_len];
            for u2 in 0..num_units {
                let s2 = &t_mean[u2 * t_len..(u2 + 1) * t_len];
                let dot: f32 = s1.iter().zip(s2).map(|(&a, &b)| a * b).sum();
                sim_matrix[u1 * num_units + u2] = dot / (norms[u1] * norms[u2]);
            }
        }
        write_npy_f32_2d(&dir.join("similar_templates.npy"), &sim_matrix, [num_units, num_units])?;
    }

    // Probe geometry: channel_map.npy, channel_positions.npy, and channel_shanks.npy
    if let Some(probe) = &sorting.probe {
        let ch_ids: Vec<i32> = probe.contacts.iter().map(|c| c.channel_id as i32).collect();
        write_npy_i32_1d(&dir.join("channel_map.npy"), &ch_ids)?;
        let shanks: Vec<i32> = probe.contacts.iter().map(|c| c.shank_id as i32).collect();
        write_npy_i32_1d(&dir.join("channel_shanks.npy"), &shanks)?;
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

/// Loads a Phy / Kilosort folder into a [`SortingOutput`] (through [`PhySorting::load`], which
/// reads every array and table; see there).
pub fn load_phy_folder(dir: &Path) -> DspResult<SortingOutput> {
    if !dir.is_dir() {
        return Err(DspError::Io(format!("{} is not a directory", dir.display())));
    }
    Ok(PhySorting::load(dir)?.to_sorting_output())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{SortedUnit, WaveformTemplate};
    use dsp_core::{SensorLayout, SensorSite};

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
