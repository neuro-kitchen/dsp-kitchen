//! Self-Contained Zarr v3 Sorting Analyzer Store (`.sorting.zarr`, `zarr_analyzer.rs`).
//!
//! Stores [`SortingOutput`] as a modular Zarr v3 directory bundling:
//! - Root `zarr.json` with metadata (`sorter_name`, `sample_rate_hz`, `total_samples`,
//!   `recording_meta`, `probe`, `drift`, and `units`).
//! - Subgroup `/spikes` with Zarr v3 arrays `times`, `clusters`, `amplitudes`, `locations`.
//! - Subgroup `/templates` with Zarr v3 arrays `mean`, `std`, `se` (`[N_units, N_channels, N_samples]`).

use std::path::Path;

use dsp_core::{DspError, DspResult, SensorLayout};
use serde::{Deserialize, Serialize};
use serde_json::Map;

use crate::core::{
    DenseTemplates, RecordingMeta, SortedUnit, SortingOutput, TemplateAxisOrder, UnitQualityLabel,
};
use crate::spatial::DriftEstimate;
use super::zarr_store::{
    open_rw_store, read_array, read_node_json, read_optional_array, write_array_f32,
    write_array_i32, write_array_u64, write_group,
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
    dsp_format: String,
    sorter_name: String,
    sample_rate_hz: f64,
    total_samples: u64,
    #[serde(default)]
    recording_meta: RecordingMeta,
    probe: Option<SensorLayout>,
    drift: Option<DriftEstimate>,
    units: Vec<UnitMetadataEntry>,
}

/// Saves a [`SortingOutput`] to a `.sorting.zarr` Zarr v3 directory.
pub fn save_sorting_zarr(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    let store = open_rw_store(dir)?;

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
        dsp_format: "sorting_analyzer_v1".into(),
        sorter_name: sorting.sorter_name.clone(),
        sample_rate_hz: sorting.sample_rate_hz,
        total_samples: sorting.total_samples,
        recording_meta: sorting.recording_meta.clone(),
        probe: sorting.probe.clone(),
        drift: sorting.drift.clone(),
        units: unit_entries,
    };

    let root_attrs = serde_json::to_value(&manifest)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    write_group(&store, dir, "/", root_attrs)?;
    write_group(&store, dir, "/spikes", Map::new())?;
    write_group(&store, dir, "/templates", Map::new())?;

    // 1. Spikes arrays
    let (samples, clusters, amps, locs) = sorting.flattened_spikes();
    write_array_u64(&store, dir, "/spikes/times", &samples, &[samples.len()], &["spikes"], Map::new())?;
    write_array_i32(&store, dir, "/spikes/clusters", &clusters, &[clusters.len()], &["spikes"], Map::new())?;
    write_array_f32(&store, dir, "/spikes/amplitudes", &amps, &[amps.len()], &["spikes"], Map::new())?;
    if !locs.is_empty() {
        let flat_locs: Vec<f32> = locs.iter().flatten().copied().collect();
        write_array_f32(
            &store,
            dir,
            "/spikes/locations",
            &flat_locs,
            &[locs.len(), 3],
            &["spikes", "xyz"],
            Map::new(),
        )?;
    }

    // 2. Templates arrays [N_units, N_channels, N_samples] via DenseTemplates
    let probe_channels = sorting.probe.as_ref().map_or(0, |p| p.total_channels());
    let unit_templates = sorting
        .units
        .iter()
        .enumerate()
        .map(|(slot, u)| (slot, u.template.as_ref()));
    if let Some((shape, t_mean, t_std, t_se)) =
        DenseTemplates::pack_units(unit_templates, probe_channels, TemplateAxisOrder::ChannelsSamples)
    {
        let dims = ["units", "channels", "samples"];
        write_array_f32(&store, dir, "/templates/mean", &t_mean, &shape, &dims, Map::new())?;
        write_array_f32(&store, dir, "/templates/std", &t_std, &shape, &dims, Map::new())?;
        write_array_f32(&store, dir, "/templates/se", &t_se, &shape, &dims, Map::new())?;
    }

    Ok(())
}

/// Loads a `.sorting.zarr` Zarr v3 directory into a [`SortingOutput`].
pub fn load_sorting_zarr(dir: &Path) -> DspResult<SortingOutput> {
    let root_json = read_node_json(dir, "/").ok_or_else(|| {
        DspError::UnsupportedFormat(format!("Missing zarr.json in {}", dir.display()))
    })?;
    let manifest_val = root_json
        .get("attributes")
        .filter(|v| v.get("dsp_format").is_some())
        .unwrap_or(&root_json);
    let manifest: SortingZarrManifest = serde_json::from_value(manifest_val.clone())
        .map_err(|e| DspError::Io(format!("{}: {e}", dir.display())))?;

    let times = read_array::<u64>(dir, "/spikes/times")?.data;
    let clusters = read_array::<i32>(dir, "/spikes/clusters")?.data;
    let amps = read_optional_array::<f32>(dir, "/spikes/amplitudes")
        .map(|a| a.data)
        .unwrap_or_default();
    let locs = read_optional_array::<f32>(dir, "/spikes/locations")
        .and_then(|a| {
            (a.shape.get(1) == Some(&3)).then(|| {
                a.data
                    .chunks_exact(3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect::<Vec<[f32; 3]>>()
            })
        })
        .unwrap_or_default();

    let loaded_t = read_optional_array::<f32>(dir, "/templates/mean")
        .and_then(|a| a.dims::<3>(dir).ok().map(|dims| (a.data, dims)));
    let loaded_std = read_optional_array::<f32>(dir, "/templates/std").map(|a| a.data);
    let loaded_se = read_optional_array::<f32>(dir, "/templates/se").map(|a| a.data);

    let mut groups = SortingOutput::group_spikes_by_cluster(&times, &clusters, &amps, &locs);
    let mut units = Vec::with_capacity(manifest.units.len());

    for (slot, meta) in manifest.units.iter().enumerate() {
        let (u_times, u_amps, u_locs) = groups.remove(&meta.unit_id).unwrap_or_default();
        let template = loaded_t.as_ref().and_then(|(t_mean, shape)| {
            // Support both slot-indexed (v3) and legacy unit_id-indexed templates
            let idx = if slot < shape[0] && manifest.units.len() == shape[0] {
                slot
            } else {
                meta.unit_id
            };
            DenseTemplates::unpack_unit(
                idx,
                *shape,
                TemplateAxisOrder::ChannelsSamples,
                t_mean,
                loaded_std.as_deref(),
                loaded_se.as_deref(),
                meta.num_spikes.max(u_times.len()),
            )
        });

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
    )
    .with_recording_meta(manifest.recording_meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WaveformTemplate;
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

        let unit0 = SortedUnit::from_spikes(5, 1, times, amps, locs, Some(template), 30_000.0, 5000, 10.0);
        let contacts = vec![
            SensorSite::new(0, dsp_core::Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, dsp_core::Position3D::new(0.0, 20.0, 0.0), 0),
        ];
        let probe = SensorLayout::new("2ch", contacts);
        let orig = SortingOutput::new("zarr_test_sorter", 30_000.0, 5000, Some(probe), vec![unit0], None);

        save_sorting_zarr(&orig, &dir).unwrap();
        assert!(dir.join("zarr.json").exists());
        assert!(dir.join("spikes").join("times").join("zarr.json").exists());
        assert!(dir.join("templates").join("mean").join("zarr.json").exists());

        let loaded = load_sorting_zarr(&dir).unwrap();
        assert_eq!(loaded.sorter_name, "zarr_test_sorter");
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].unit_id, 5);
        assert_eq!(loaded.units[0].primary_channel, 1);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);
        assert_eq!(loaded.units[0].locations_um, orig.units[0].locations_um);
        assert!(loaded.units[0].template.is_some());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
