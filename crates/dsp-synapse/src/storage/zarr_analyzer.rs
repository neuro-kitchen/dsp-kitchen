//! Self-Contained Zarr Sorting Analyzer Store (`.sorting.zarr`, `zarr_analyzer.rs`).
//!
//! Stores [`SortingOutput`] as a modular Zarr v3 directory bundling:
//! - Root `zarr.json` with metadata (`sorter_name`, `sample_rate_hz`, `total_samples`,
//!   `units_summary`, `probe`, and `drift`).
//! - Subgroup `spikes/` with `times.npy`, `clusters.npy`, `amplitudes.npy`, `locations.npy`.
//! - Subgroup `templates/` with `mean.npy`, `std.npy`, `se.npy`, and `channel_ids.npy`.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dsp_core::{DspError, DspResult, SensorLayout};
use serde::{Deserialize, Serialize};

use crate::core::{SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate};
use crate::spatial::DriftEstimate;
use super::npy::{
    read_npy_f32_1d, read_npy_f32_2d, read_npy_f32_3d, read_npy_i32_1d, read_npy_u64_1d,
    write_npy_f32_1d, write_npy_f32_2d, write_npy_f32_3d, write_npy_i32_1d, write_npy_u64_1d,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UnitMetadataEntry {
    unit_id: usize,
    primary_channel: usize,
    quality_label: UnitQualityLabel,
    #[serde(default, deserialize_with = "crate::core::sorting_output::serde_nan::deserialize_f32")]
    snr: f32,
    #[serde(default, deserialize_with = "crate::core::sorting_output::serde_nan::deserialize_f64")]
    firing_rate_hz: f64,
    #[serde(default, deserialize_with = "crate::core::sorting_output::serde_nan::deserialize_f64")]
    isi_violation_ratio: f64,
    #[serde(default, deserialize_with = "crate::core::sorting_output::serde_nan::deserialize_f64")]
    presence_ratio: f64,
    #[serde(default, deserialize_with = "crate::core::sorting_output::serde_nan::deserialize_f64")]
    amplitude_cutoff: f64,
    num_spikes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SortingZarrManifest {
    zarr_format: u32,
    node_type: String,
    dsp_format: String,
    sorter_name: String,
    sample_rate_hz: f64,
    total_samples: u64,
    probe: Option<SensorLayout>,
    drift: Option<DriftEstimate>,
    units: Vec<UnitMetadataEntry>,
}

/// Saves a [`SortingOutput`] to a `.sorting.zarr` directory.
pub fn save_sorting_zarr(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    std::fs::create_dir_all(dir).map_err(|e| DspError::Io(e.to_string()))?;
    let spikes_dir = dir.join("spikes");
    let templates_dir = dir.join("templates");
    std::fs::create_dir_all(&spikes_dir).map_err(|e| DspError::Io(e.to_string()))?;
    std::fs::create_dir_all(&templates_dir).map_err(|e| DspError::Io(e.to_string()))?;

    // 1. Spikes arrays
    let (samples, clusters, amps, locs) = sorting.flattened_spikes();
    write_npy_u64_1d(&spikes_dir.join("times.npy"), &samples)?;
    write_npy_i32_1d(&spikes_dir.join("clusters.npy"), &clusters)?;
    write_npy_f32_1d(&spikes_dir.join("amplitudes.npy"), &amps)?;
    if !locs.is_empty() {
        let flat_locs: Vec<f32> = locs.iter().flatten().copied().collect();
        write_npy_f32_2d(&spikes_dir.join("locations.npy"), &flat_locs, [locs.len(), 3])?;
    }

    // 2. Templates arrays [N_units, N_channels, N_samples]
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
        let total_t = num_units * num_channels * template_samples;
        let mut t_mean = vec![0.0f32; total_t];
        let mut t_std = vec![0.0f32; total_t];
        let mut t_se = vec![0.0f32; total_t];

        for (u_idx, unit) in sorting.units.iter().enumerate() {
            if let Some(t) = &unit.template {
                let off = u_idx * num_channels * template_samples;
                for (r, &ch) in t.channel_ids.iter().enumerate() {
                    if ch < num_channels {
                        let dst_off = off + ch * template_samples;
                        let src_slice = t.row(r);
                        let n_s = src_slice.len().min(template_samples);
                        t_mean[dst_off..dst_off + n_s].copy_from_slice(&src_slice[..n_s]);
                        t_std[dst_off..dst_off + n_s].copy_from_slice(&t.std[r * t.num_samples..r * t.num_samples + n_s]);
                        t_se[dst_off..dst_off + n_s].copy_from_slice(&t.se[r * t.num_samples..r * t.num_samples + n_s]);
                    }
                }
            }
        }
        let shape = [num_units, num_channels, template_samples];
        write_npy_f32_3d(&templates_dir.join("mean.npy"), &t_mean, shape)?;
        write_npy_f32_3d(&templates_dir.join("std.npy"), &t_std, shape)?;
        write_npy_f32_3d(&templates_dir.join("se.npy"), &t_se, shape)?;
    }

    // 3. Manifest metadata in zarr.json
    let unit_entries: Vec<UnitMetadataEntry> = sorting
        .units
        .iter()
        .map(|u| UnitMetadataEntry {
            unit_id: u.unit_id,
            primary_channel: u.primary_channel,
            quality_label: u.quality_label,
            snr: u.snr,
            firing_rate_hz: u.firing_rate_hz,
            isi_violation_ratio: u.isi_violation_ratio,
            presence_ratio: u.presence_ratio,
            amplitude_cutoff: u.amplitude_cutoff,
            num_spikes: u.spike_samples.len(),
        })
        .collect();

    let manifest = SortingZarrManifest {
        zarr_format: 3,
        node_type: "group".into(),
        dsp_format: "sorting_analyzer_v1".into(),
        sorter_name: sorting.sorter_name.clone(),
        sample_rate_hz: sorting.sample_rate_hz,
        total_samples: sorting.total_samples,
        probe: sorting.probe.clone(),
        drift: sorting.drift.clone(),
        units: unit_entries,
    };

    let json_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| DspError::Io(e.to_string()))?;
    let mut zarr_file = File::create(dir.join("zarr.json")).map_err(|e| DspError::Io(e.to_string()))?;
    zarr_file.write_all(&json_bytes).map_err(|e| DspError::Io(e.to_string()))?;

    Ok(())
}

/// Loads a `.sorting.zarr` directory into a [`SortingOutput`].
pub fn load_sorting_zarr(dir: &Path) -> DspResult<SortingOutput> {
    let manifest_path = dir.join("zarr.json");
    if !manifest_path.exists() {
        return Err(DspError::UnsupportedFormat(format!(
            "Missing zarr.json in {}",
            dir.display()
        )));
    }

    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| DspError::Io(e.to_string()))?;
    let manifest: SortingZarrManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|e| DspError::Io(e.to_string()))?;

    let spikes_dir = dir.join("spikes");
    let times = read_npy_u64_1d(&spikes_dir.join("times.npy"))?;
    let clusters = read_npy_i32_1d(&spikes_dir.join("clusters.npy"))?;
    let amps = spikes_dir
        .join("amplitudes.npy")
        .exists()
        .then(|| read_npy_f32_1d(&spikes_dir.join("amplitudes.npy")).ok())
        .flatten()
        .unwrap_or_default();

    let locs = spikes_dir
        .join("locations.npy")
        .exists()
        .then(|| read_npy_f32_2d(&spikes_dir.join("locations.npy")).ok())
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

    let templates_dir = dir.join("templates");
    let loaded_t = templates_dir
        .join("mean.npy")
        .exists()
        .then(|| read_npy_f32_3d(&templates_dir.join("mean.npy")).ok())
        .flatten();
    let loaded_std = templates_dir
        .join("std.npy")
        .exists()
        .then(|| read_npy_f32_3d(&templates_dir.join("std.npy")).ok())
        .flatten();
    let loaded_se = templates_dir
        .join("se.npy")
        .exists()
        .then(|| read_npy_f32_3d(&templates_dir.join("se.npy")).ok())
        .flatten();

    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, (&t, &c)) in times.iter().zip(&clusters).enumerate() {
        if c >= 0 {
            let _ = t;
            groups.entry(c as usize).or_default().push(i);
        }
    }

    let mut units = Vec::with_capacity(manifest.units.len());
    for meta in &manifest.units {
        let indices = groups.remove(&meta.unit_id).unwrap_or_default();
        let u_times: Vec<u64> = indices.iter().map(|&idx| times[idx]).collect();
        let u_amps: Vec<f32> = indices
            .iter()
            .filter_map(|&idx| amps.get(idx).copied())
            .collect();
        let u_locs: Vec<[f32; 3]> = indices
            .iter()
            .filter_map(|&idx| locs.get(idx).copied())
            .collect();

        let template = if let Some((t_mean, [n_u, n_c, n_s])) = &loaded_t {
            if meta.unit_id < *n_u {
                let off = meta.unit_id * n_c * n_s;
                let mean = t_mean[off..off + n_c * n_s].to_vec();
                let std = loaded_std
                    .as_ref()
                    .map(|(s, _)| s[off..off + n_c * n_s].to_vec())
                    .unwrap_or_else(|| vec![1.0; n_c * n_s]);
                let se = loaded_se
                    .as_ref()
                    .map(|(s, _)| s[off..off + n_c * n_s].to_vec())
                    .unwrap_or_else(|| vec![0.0; n_c * n_s]);

                let mut t = WaveformTemplate::with_count(
                    (0..*n_c).collect(),
                    *n_s,
                    meta.num_spikes.max(indices.len()),
                    mean,
                    std,
                );
                t.se = se;
                Some(t)
            } else {
                None
            }
        } else {
            None
        };

        units.push(SortedUnit {
            unit_id: meta.unit_id,
            primary_channel: meta.primary_channel,
            spike_samples: u_times,
            amplitudes_uv: u_amps,
            locations_um: u_locs,
            template,
            quality_label: meta.quality_label,
            snr: meta.snr,
            firing_rate_hz: meta.firing_rate_hz,
            isi_violation_ratio: meta.isi_violation_ratio,
            presence_ratio: meta.presence_ratio,
            amplitude_cutoff: meta.amplitude_cutoff,
        });
    }

    Ok(SortingOutput::new(
        manifest.sorter_name,
        manifest.sample_rate_hz,
        manifest.total_samples,
        manifest.probe,
        units,
        manifest.drift,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::SensorSite;

    #[test]
    fn test_zarr_analyzer_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dsp_zarr_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let times = vec![500u64, 1200, 2900];
        let amps = vec![95.0f32, 105.0, 100.0];
        let locs = vec![[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]];
        let t_mean = vec![0.0f32; 2 * 20];
        let template = WaveformTemplate::with_count(vec![0, 1], 20, 3, t_mean, vec![1.0; 40]);

        let unit0 = SortedUnit::from_spikes(0, 0, times, amps, locs, Some(template), 30_000.0, 5000, 10.0);
        let contacts = vec![
            SensorSite::new(0, dsp_core::Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, dsp_core::Position3D::new(0.0, 20.0, 0.0), 0),
        ];
        let probe = SensorLayout::new("2ch", contacts);
        let orig = SortingOutput::new("zarr_test_sorter", 30_000.0, 5000, Some(probe), vec![unit0], None);

        save_sorting_zarr(&orig, &dir).unwrap();
        assert!(dir.join("zarr.json").exists());
        assert!(dir.join("spikes").join("times.npy").exists());
        assert!(dir.join("templates").join("mean.npy").exists());

        let loaded = load_sorting_zarr(&dir).unwrap();
        assert_eq!(loaded.sorter_name, "zarr_test_sorter");
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);
        assert_eq!(loaded.units[0].locations_um, orig.units[0].locations_um);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
