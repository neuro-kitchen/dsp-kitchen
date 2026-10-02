//! NWB `/units` DynamicTable Zarr Persistence (`nwb_units.rs`).
//!
//! Stores [`SortingOutput`] within the standard NWB `/units` group inside a `.nwb.zarr` directory:
//! - `units/id`: `[0..K]`
//! - `units/spike_times`: concatenated `f64` spike timestamps in seconds
//! - `units/spike_times_index`: `[K]` ragged cumulative spike counts
//! - `units/waveform_mean`: `[K, C, T]`
//! - `units/waveform_sd`: `[K, C, T]`
//! - `units/waveform_se`: `[K, C, T]`
//! - `units/snr`, `units/firing_rate`, `units/quality`

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dsp_core::{DspError, DspResult};
use serde::{Deserialize, Serialize};

use crate::core::{SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate};
use super::npy::{
    read_npy_f32_1d, read_npy_f32_3d, read_npy_i32_1d, read_npy_u64_1d, write_npy_f32_1d,
    write_npy_f32_3d, write_npy_i32_1d, write_npy_u64_1d,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NwbUnitsZarrJson {
    zarr_format: u32,
    node_type: String,
    attributes: NwbUnitsAttributes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NwbUnitsAttributes {
    neurodata_type: String,
    namespace: String,
    description: String,
    colnames: Vec<String>,
}

/// Saves [`SortingOutput`] into the `units/` group inside `nwb_zarr_dir`.
pub fn save_nwb_units(sorting: &SortingOutput, nwb_zarr_dir: &Path) -> DspResult<()> {
    let units_dir = nwb_zarr_dir.join("units");
    std::fs::create_dir_all(&units_dir).map_err(|e| DspError::Io(e.to_string()))?;

    let num_units = sorting.units.len();
    let mut unit_ids = Vec::with_capacity(num_units);
    let mut spike_times_sec = Vec::new();
    let mut spike_times_index = Vec::with_capacity(num_units);
    let mut snr_vec = Vec::with_capacity(num_units);
    let mut fr_vec = Vec::with_capacity(num_units);

    let fs = sorting.sample_rate_hz.max(1.0);
    let mut cum_spikes = 0u64;

    for u in &sorting.units {
        unit_ids.push(u.unit_id as i32);
        for &s in &u.spike_samples {
            spike_times_sec.push((s as f64) / fs);
        }
        cum_spikes += u.spike_samples.len() as u64;
        spike_times_index.push(cum_spikes);
        snr_vec.push(u.snr);
        fr_vec.push(u.firing_rate_hz as f32);
    }

    write_npy_i32_1d(&units_dir.join("id.npy"), &unit_ids)?;
    write_npy_u64_1d(&units_dir.join("spike_times_index.npy"), &spike_times_index)?;
    let spike_times_f32: Vec<f32> = spike_times_sec.iter().map(|&t| t as f32).collect();
    write_npy_f32_1d(&units_dir.join("spike_times.npy"), &spike_times_f32)?;
    write_npy_f32_1d(&units_dir.join("snr.npy"), &snr_vec)?;
    write_npy_f32_1d(&units_dir.join("firing_rate.npy"), &fr_vec)?;

    // Templates [num_units, num_channels, num_samples]
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
        write_npy_f32_3d(&units_dir.join("waveform_mean.npy"), &t_mean, shape)?;
        write_npy_f32_3d(&units_dir.join("waveform_sd.npy"), &t_std, shape)?;
        write_npy_f32_3d(&units_dir.join("waveform_se.npy"), &t_se, shape)?;
    }

    let manifest = NwbUnitsZarrJson {
        zarr_format: 3,
        node_type: "group".into(),
        attributes: NwbUnitsAttributes {
            neurodata_type: "Units".into(),
            namespace: "core".into(),
            description: format!("Sorted neural units extracted by {}", sorting.sorter_name),
            colnames: vec![
                "id".into(),
                "spike_times".into(),
                "spike_times_index".into(),
                "waveform_mean".into(),
                "snr".into(),
                "firing_rate".into(),
            ],
        },
    };
    let json_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| DspError::Io(e.to_string()))?;
    let mut zarr_file = File::create(units_dir.join("zarr.json")).map_err(|e| DspError::Io(e.to_string()))?;
    zarr_file.write_all(&json_bytes).map_err(|e| DspError::Io(e.to_string()))?;

    Ok(())
}

/// Loads a `SortingOutput` from the `units/` group inside `nwb_zarr_dir`.
pub fn load_nwb_units(nwb_zarr_dir: &Path, sample_rate_hz: f64) -> DspResult<SortingOutput> {
    let units_dir = if nwb_zarr_dir.join("units").exists() {
        nwb_zarr_dir.join("units")
    } else {
        nwb_zarr_dir.to_path_buf()
    };

    let id_path = units_dir.join("id.npy");
    let times_path = units_dir.join("spike_times.npy");
    let index_path = units_dir.join("spike_times_index.npy");
    if !id_path.exists() || !times_path.exists() || !index_path.exists() {
        return Err(DspError::UnsupportedFormat(format!(
            "NWB units group in {} missing id.npy, spike_times.npy, or spike_times_index.npy",
            units_dir.display()
        )));
    }

    let unit_ids = read_npy_i32_1d(&id_path)?;
    let spike_times_sec = read_npy_f32_1d(&times_path)?;
    let spike_times_index = read_npy_u64_1d(&index_path)?;

    let snrs = units_dir
        .join("snr.npy")
        .exists()
        .then(|| read_npy_f32_1d(&units_dir.join("snr.npy")).ok())
        .flatten()
        .unwrap_or_default();

    let loaded_t = units_dir
        .join("waveform_mean.npy")
        .exists()
        .then(|| read_npy_f32_3d(&units_dir.join("waveform_mean.npy")).ok())
        .flatten();
    let loaded_se = units_dir
        .join("waveform_se.npy")
        .exists()
        .then(|| read_npy_f32_3d(&units_dir.join("waveform_se.npy")).ok())
        .flatten();

    let fs = sample_rate_hz.max(1.0);
    let mut prev_idx = 0usize;
    let mut units = Vec::with_capacity(unit_ids.len());

    for (u_pos, &uid) in unit_ids.iter().enumerate() {
        let end_idx = spike_times_index
            .get(u_pos)
            .copied()
            .unwrap_or(0) as usize;
        let u_sec = &spike_times_sec[prev_idx.min(spike_times_sec.len())..end_idx.min(spike_times_sec.len())];
        let spike_samples: Vec<u64> = u_sec.iter().map(|&t| (t as f64 * fs).round() as u64).collect();
        prev_idx = end_idx;

        let template = if let Some((t_mean, [n_u, n_c, n_s])) = &loaded_t {
            if u_pos < *n_u {
                let off = u_pos * n_c * n_s;
                let mean = t_mean[off..off + n_c * n_s].to_vec();
                let se = loaded_se
                    .as_ref()
                    .map(|(s, _)| s[off..off + n_c * n_s].to_vec())
                    .unwrap_or_else(|| vec![0.0; n_c * n_s]);
                let mut t = WaveformTemplate::with_count(
                    (0..*n_c).collect(),
                    *n_s,
                    spike_samples.len(),
                    mean,
                    vec![1.0; n_c * n_s],
                );
                t.se = se;
                Some(t)
            } else {
                None
            }
        } else {
            None
        };

        let mut unit = SortedUnit::from_spikes(
            uid as usize,
            0,
            spike_samples,
            Vec::new(),
            Vec::new(),
            template,
            fs,
            0,
            10.0,
        );
        if let Some(&snr) = snrs.get(u_pos) {
            unit.snr = snr;
            if snr >= 3.0 {
                unit.quality_label = UnitQualityLabel::SingleUnit;
            }
        }
        units.push(unit);
    }

    let total_samples = units
        .iter()
        .filter_map(|u| u.spike_samples.last().copied())
        .max()
        .unwrap_or(0);

    Ok(SortingOutput::new(
        "nwb_units",
        fs,
        total_samples,
        None,
        units,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nwb_units_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dsp_nwb_units_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let times = vec![300u64, 1500, 3000];
        let t_mean = vec![0.0f32; 2 * 25];
        let template = WaveformTemplate::with_count(vec![0, 1], 25, 3, t_mean, vec![1.0; 50]);

        let unit0 = SortedUnit::from_spikes(0, 0, times, Vec::new(), Vec::new(), Some(template), 30_000.0, 4000, 8.0);
        let orig = SortingOutput::new("nwb_sorter", 30_000.0, 4000, None, vec![unit0], None);

        save_nwb_units(&orig, &dir).unwrap();
        assert!(dir.join("units").join("id.npy").exists());
        assert!(dir.join("units").join("spike_times.npy").exists());
        assert!(dir.join("units").join("spike_times_index.npy").exists());

        let loaded = load_nwb_units(&dir, 30_000.0).unwrap();
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
