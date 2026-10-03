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
use crate::core::{DenseTemplates, SortingOutput, TemplateAxisOrder, WaveformTemplate};
use crate::sorting::similarity::compute_template_similarity_matrix;
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

    let num_channels = sorting
        .probe
        .as_ref()
        .map(|p| p.total_channels())
        .unwrap_or_else(|| {
            sorting
                .units
                .iter()
                .filter_map(|u| u.template.as_ref())
                .flat_map(|t| t.channel_ids.iter().copied())
                .max()
                .map_or(1, |c| c + 1)
        });

    if let Some((shape, t_mean, t_std, t_se)) = DenseTemplates::pack_units(
        sorting.units.iter().map(|u| (u.unit_id, u.template.as_ref())),
        num_channels,
        TemplateAxisOrder::SamplesChannels,
    ) {
        let [count, template_samples, channels] = shape;
        write_npy_f32_3d(&dir.join("templates.npy"), &t_mean, shape)?;
        write_npy_f32_3d(&dir.join("templates_std.npy"), &t_std, shape)?;
        write_npy_f32_3d(&dir.join("templates_se.npy"), &t_se, shape)?;

        let dense = DenseTemplates {
            data: t_mean,
            count,
            samples: template_samples,
            channels,
        };
        let waveforms: Vec<WaveformTemplate> = (0..count).map(|i| dense.waveform(i)).collect();
        let sim_matrix = compute_template_similarity_matrix(&waveforms, 5);
        write_npy_f32_2d(&dir.join("similar_templates.npy"), &sim_matrix, [count, count])?;
    }

    // Probe geometry: channel_map.npy, channel_positions.npy, and channel_shanks.npy
    if let Some(probe) = &sorting.probe {
        let (ch_map, ch_pos, ch_shanks) = probe.to_channel_arrays();
        let ch_ids: Vec<i32> = ch_map.into_iter().map(|c| c as i32).collect();
        write_npy_i32_1d(&dir.join("channel_map.npy"), &ch_ids)?;
        let shanks: Vec<i32> = ch_shanks.into_iter().map(|s| s as i32).collect();
        write_npy_i32_1d(&dir.join("channel_shanks.npy"), &shanks)?;
        let pos: Vec<f32> = ch_pos.iter().flatten().copied().collect();
        write_npy_f32_2d(&dir.join("channel_positions.npy"), &pos, [ch_pos.len(), 2])?;
    }

    // cluster_group.tsv and cluster_info.tsv
    let mut grp_file = File::create(dir.join("cluster_group.tsv")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(grp_file, "cluster_id\tgroup").map_err(|e| DspError::Io(e.to_string()))?;
    for u in &sorting.units {
        writeln!(grp_file, "{}\t{}", u.unit_id, u.quality_label.as_str()).map_err(|e| DspError::Io(e.to_string()))?;
    }

    let mut info_file = File::create(dir.join("cluster_info.tsv")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(
        info_file,
        "cluster_id\tch\tfiring_rate\tsnr\tisi_viol\tpresence_ratio\tamplitude_cutoff\tgroup"
    )
    .map_err(|e| DspError::Io(e.to_string()))?;
    for u in &sorting.units {
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
            u.quality_label.as_str()
        )
        .map_err(|e| DspError::Io(e.to_string()))?;
    }

    // params.py
    let mut params_file = File::create(dir.join("params.py")).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "# Generated by dsp-kitchen").map_err(|e| DspError::Io(e.to_string()))?;
    let meta = &sorting.recording_meta;
    if let Some(dat_path) = meta.dat_path.as_deref() {
        writeln!(params_file, "dat_path = '{dat_path}'").map_err(|e| DspError::Io(e.to_string()))?;
    }
    let n_ch_dat = meta.n_channels_dat.unwrap_or(num_channels);
    writeln!(params_file, "n_channels_dat = {n_ch_dat}").map_err(|e| DspError::Io(e.to_string()))?;
    let dtype = meta.dtype.as_deref().unwrap_or("int16");
    writeln!(params_file, "dtype = '{dtype}'").map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "offset = {}", meta.offset).map_err(|e| DspError::Io(e.to_string()))?;
    writeln!(params_file, "sample_rate = {}", sorting.sample_rate_hz).map_err(|e| DspError::Io(e.to_string()))?;
    let hp_str = if meta.hp_filtered { "True" } else { "False" };
    writeln!(params_file, "hp_filtered = {hp_str}").map_err(|e| DspError::Io(e.to_string()))?;

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
