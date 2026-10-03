//! NWB `/units` DynamicTable Zarr v3 Persistence (`nwb_units.rs`).
//!
//! Stores and loads [`SortingOutput`] within the standard NWB `/units` group inside a `.nwb.zarr`
//! store (compatible with `neuro-convert` and `hdmf-zarr`):
//! - `/units/id`: `[K]` (`int64`)
//! - `/units/spike_times`: concatenated `float64` spike timestamps in seconds (with `resolution = 1 / fs`)
//! - `/units/spike_times_index`: `[K]` (`uint64`) ragged cumulative spike counts
//! - `/units/waveform_mean`, `/units/waveform_sd`, `/units/waveform_se`: `[K, C, T]` (or `[K, T]`)
//! - `/units/snr`, `/units/firing_rate`, `/units/primary_channel`

use std::path::Path;

use dsp_core::{DspError, DspResult, SensorLayout};
use serde_json::{Map, Value, json};

use crate::core::{
    DenseTemplates, RecordingMeta, SortedUnit, SortingOutput, TemplateAxisOrder, UnitQualityLabel,
    WaveformTemplate,
};
use crate::metrics::QualityCriteria;
use super::zarr_store::{
    has_array, infer_nwb_sample_rate, open_rw_store, read_array, read_node_attributes,
    read_optional_array, write_array_f32, write_array_f64, write_array_i64, write_array_u64,
    write_group,
};

fn col_attrs(description: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("neurodata_type".into(), json!("VectorData"));
    m.insert("namespace".into(), json!("hdmf-common"));
    m.insert("description".into(), json!(description));
    m
}

/// Saves [`SortingOutput`] into the `/units` Zarr v3 group inside `nwb_zarr_dir`.
pub fn save_nwb_units(sorting: &SortingOutput, nwb_zarr_dir: &Path) -> DspResult<()> {
    let store = open_rw_store(nwb_zarr_dir)?;
    let fs = sorting.sample_rate_hz.max(1.0);

    let (unit_ids, spike_times_sec, spike_times_index) = sorting.to_ragged_spikes();
    let num_units = sorting.units.len();
    let snr_vec: Vec<f32> = sorting.units.iter().map(|u| u.snr).collect();
    let fr_vec: Vec<f32> = sorting.units.iter().map(|u| u.firing_rate_hz as f32).collect();
    let ch_vec: Vec<i64> = sorting.units.iter().map(|u| u.primary_channel as i64).collect();
    let quality_labels: Vec<&str> = sorting.units.iter().map(|u| u.quality_label.as_str()).collect();

    let total_spikes = spike_times_sec.len();
    let has_all_amps = total_spikes > 0
        && sorting
            .units
            .iter()
            .all(|u| u.amplitudes_uv.len() == u.spike_samples.len());
    let has_all_locs = total_spikes > 0
        && sorting
            .units
            .iter()
            .all(|u| u.locations_um.len() == u.spike_samples.len());

    let probe_channels = sorting.probe.as_ref().map_or(0, |p| p.total_channels());
    let packed_templates = DenseTemplates::pack_units(
        sorting.units.iter().enumerate().map(|(slot, u)| (slot, u.template.as_ref())),
        probe_channels,
        TemplateAxisOrder::ChannelsSamples,
    );

    let mut colnames = vec![
        "spike_times",
        "snr",
        "firing_rate",
        "primary_channel",
    ];
    if packed_templates.is_some() {
        colnames.push("waveform_mean");
        colnames.push("waveform_sd");
        colnames.push("waveform_se");
    }
    if has_all_amps {
        colnames.push("spike_amplitudes");
    }
    if has_all_locs {
        colnames.push("spike_locations");
    }

    let mut group_attrs = Map::new();
    group_attrs.insert("neurodata_type".into(), json!("Units"));
    group_attrs.insert("namespace".into(), json!("core"));
    group_attrs.insert(
        "description".into(),
        json!(format!("Sorted neural units extracted by {}", sorting.sorter_name)),
    );
    group_attrs.insert("colnames".into(), json!(colnames));
    group_attrs.insert("sample_rate_hz".into(), json!(fs));
    group_attrs.insert("total_samples".into(), json!(sorting.total_samples));
    group_attrs.insert("sorter_name".into(), json!(sorting.sorter_name));
    group_attrs.insert("quality_labels".into(), json!(quality_labels));
    if let Some(probe) = &sorting.probe
        && let Ok(val) = serde_json::to_value(probe)
    {
        group_attrs.insert("probe".into(), val);
    }
    write_group(&store, nwb_zarr_dir, "/units", group_attrs)?;

    let mut id_attrs = Map::new();
    id_attrs.insert("neurodata_type".into(), json!("ElementIdentifiers"));
    id_attrs.insert("namespace".into(), json!("hdmf-common"));
    write_array_i64(&store, nwb_zarr_dir, "/units/id", &unit_ids, &[num_units], &["num_rows"], id_attrs)?;

    let mut st_attrs = col_attrs("the spike times for each unit in seconds");
    st_attrs.insert("resolution".into(), json!(1.0 / fs));
    st_attrs.insert("unit".into(), json!("seconds"));
    write_array_f64(
        &store,
        nwb_zarr_dir,
        "/units/spike_times",
        &spike_times_sec,
        &[spike_times_sec.len()],
        &["num_spikes"],
        st_attrs,
    )?;

    let mut idx_attrs = Map::new();
    idx_attrs.insert("neurodata_type".into(), json!("VectorIndex"));
    idx_attrs.insert("namespace".into(), json!("hdmf-common"));
    idx_attrs.insert("description".into(), json!("Index for VectorData 'spike_times'"));
    idx_attrs.insert(
        "target".into(),
        json!({ "_REFERENCE": { "source": ".", "path": "/units/spike_times" } }),
    );
    write_array_u64(
        &store,
        nwb_zarr_dir,
        "/units/spike_times_index",
        &spike_times_index,
        &[num_units],
        &["num_rows"],
        idx_attrs,
    )?;

    if has_all_amps {
        let ragged_amps: Vec<f32> = sorting
            .units
            .iter()
            .flat_map(|u| u.amplitudes_uv.iter().copied())
            .collect();
        write_array_f32(
            &store,
            nwb_zarr_dir,
            "/units/spike_amplitudes",
            &ragged_amps,
            &[ragged_amps.len()],
            &["num_spikes"],
            col_attrs("per-spike peak amplitudes in microvolts"),
        )?;
    }

    if has_all_locs {
        let ragged_locs: Vec<f32> = sorting
            .units
            .iter()
            .flat_map(|u| u.locations_um.iter().flatten().copied())
            .collect();
        write_array_f32(
            &store,
            nwb_zarr_dir,
            "/units/spike_locations",
            &ragged_locs,
            &[total_spikes, 3],
            &["num_spikes", "xyz"],
            col_attrs("per-spike 3D coordinates in micrometers"),
        )?;
    }

    write_array_f32(&store, nwb_zarr_dir, "/units/snr", &snr_vec, &[num_units], &["num_rows"], col_attrs("unit peak-to-noise ratio"))?;
    write_array_f32(&store, nwb_zarr_dir, "/units/firing_rate", &fr_vec, &[num_units], &["num_rows"], col_attrs("mean firing rate in Hz"))?;
    write_array_i64(&store, nwb_zarr_dir, "/units/primary_channel", &ch_vec, &[num_units], &["num_rows"], col_attrs("primary recording channel"))?;

    if let Some((shape, t_mean, t_std, t_se)) = packed_templates {
        let mut wm_attrs = col_attrs("the spike waveform mean for each spike unit");
        wm_attrs.insert("sampling_rate".into(), json!(fs));
        wm_attrs.insert("unit".into(), json!("microvolts"));
        let dims = ["num_units", "num_channels", "num_samples"];
        write_array_f32(&store, nwb_zarr_dir, "/units/waveform_mean", &t_mean, &shape, &dims, wm_attrs)?;
        write_array_f32(&store, nwb_zarr_dir, "/units/waveform_sd", &t_std, &shape, &dims, col_attrs("spike waveform standard deviation"))?;
        write_array_f32(&store, nwb_zarr_dir, "/units/waveform_se", &t_se, &shape, &dims, col_attrs("spike waveform standard error"))?;
    }

    Ok(())
}

/// Loads a [`SortingOutput`] from the `/units` group inside `nwb_zarr_dir`.
///
/// `sample_rate_hz` may be passed explicitly (`Some(rate)` or `rate`) or omitted (`None`), in
/// which case the sample rate is inferred from `/units` metadata (`sample_rate_hz`,
/// `waveform_mean.sampling_rate`, `1.0 / spike_times.resolution`, or `/acquisition` series rate).
pub fn load_nwb_units(
    nwb_zarr_dir: &Path,
    sample_rate_hz: impl Into<Option<f64>>,
) -> DspResult<SortingOutput> {
    let (base_dir, prefix) = if nwb_zarr_dir.join("units").exists() {
        (nwb_zarr_dir, "/units")
    } else {
        (nwb_zarr_dir, "")
    };
    let node = |name: &str| {
        if prefix.is_empty() {
            format!("/{name}")
        } else {
            format!("{prefix}/{name}")
        }
    };

    if !has_array(base_dir, &node("id"))
        || !has_array(base_dir, &node("spike_times"))
        || !has_array(base_dir, &node("spike_times_index"))
    {
        return Err(DspError::UnsupportedFormat(format!(
            "NWB units group in {} missing id, spike_times, or spike_times_index",
            nwb_zarr_dir.display()
        )));
    }

    let root_nwb = if nwb_zarr_dir.join("acquisition").is_dir() {
        Some(nwb_zarr_dir)
    } else {
        nwb_zarr_dir.parent().filter(|p| p.join("acquisition").is_dir())
    };

    let explicit_sr = sample_rate_hz.into().filter(|&r| r > 0.0);
    let inferred_sr = infer_nwb_sample_rate(root_nwb.unwrap_or(nwb_zarr_dir));
    let fs = explicit_sr
        .or(inferred_sr)
        .ok_or_else(|| {
            DspError::InvalidConfig(format!(
                "{}: sample_rate_hz not found in NWB metadata and none was provided",
                nwb_zarr_dir.display()
            ))
        })?;

    let group_attrs = read_node_attributes(base_dir, prefix);
    let sorter_name = group_attrs
        .as_ref()
        .and_then(|a| a.get("sorter_name"))
        .and_then(Value::as_str)
        .unwrap_or("nwb_units")
        .to_string();
    let stored_total_samples = group_attrs
        .as_ref()
        .and_then(|a| a.get("total_samples"))
        .and_then(Value::as_u64);
    let probe: Option<SensorLayout> = group_attrs
        .as_ref()
        .and_then(|a| a.get("probe"))
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let quality_labels: Vec<String> = group_attrs
        .as_ref()
        .and_then(|a| a.get("quality_labels"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let unit_ids = read_array::<i64>(base_dir, &node("id"))?.data;
    let spike_times_sec = read_array::<f64>(base_dir, &node("spike_times"))?.data;
    let spike_times_index = read_array::<u64>(base_dir, &node("spike_times_index"))?.data;

    let ragged_amps = read_optional_array::<f32>(base_dir, &node("spike_amplitudes")).map(|a| a.data);
    let ragged_locs = read_optional_array::<f32>(base_dir, &node("spike_locations"))
        .filter(|a| a.shape.len() == 2 && a.shape[1] >= 3)
        .map(|a| {
            let cols = a.shape[1];
            a.data
                .chunks_exact(cols)
                .map(|r| [r[0], r[1], r[2]])
                .collect::<Vec<[f32; 3]>>()
        });

    let snrs = read_optional_array::<f32>(base_dir, &node("snr"))
        .map(|a| a.data)
        .unwrap_or_default();
    let primary_channels = read_optional_array::<usize>(base_dir, &node("primary_channel"))
        .or_else(|| read_optional_array::<usize>(base_dir, &node("source_channel")))
        .or_else(|| read_optional_array::<usize>(base_dir, &node("electrodes")))
        .map(|a| a.data)
        .unwrap_or_default();

    let loaded_t = read_optional_array::<f32>(base_dir, &node("waveform_mean"));
    let loaded_sd = read_optional_array::<f32>(base_dir, &node("waveform_sd")).map(|a| a.data);
    let loaded_se = read_optional_array::<f32>(base_dir, &node("waveform_se")).map(|a| a.data);

    let unpacked = SortingOutput::unpack_ragged_spikes(&unit_ids, &spike_times_sec, &spike_times_index, fs);
    let total_samples = stored_total_samples.unwrap_or_else(|| {
        unpacked
            .iter()
            .filter_map(|(_, s)| s.last().copied())
            .max()
            .unwrap_or(0)
    });

    let criteria = QualityCriteria::default();
    let mut units = Vec::with_capacity(unpacked.len());
    let mut prev_idx = 0usize;

    for (u_pos, (uid, spike_samples)) in unpacked.into_iter().enumerate() {
        let end_idx = spike_times_index
            .get(u_pos)
            .copied()
            .unwrap_or(prev_idx as u64) as usize;
        let start_idx = prev_idx.min(end_idx);
        prev_idx = end_idx;

        let u_amps = ragged_amps
            .as_ref()
            .filter(|a| end_idx <= a.len())
            .map(|a| a[start_idx..end_idx].to_vec())
            .unwrap_or_default();
        let u_locs = ragged_locs
            .as_ref()
            .filter(|l| end_idx <= l.len())
            .map(|l| l[start_idx..end_idx].to_vec())
            .unwrap_or_default();

        let primary_ch = primary_channels.get(u_pos).copied();
        let template = loaded_t.as_ref().and_then(|arr| match arr.shape.as_slice() {
            // 3D: [num_units, num_channels, num_samples]
            &[n_u, n_c, n_s] => DenseTemplates::unpack_unit(
                u_pos,
                [n_u, n_c, n_s],
                TemplateAxisOrder::ChannelsSamples,
                &arr.data,
                loaded_sd.as_deref(),
                loaded_se.as_deref(),
                spike_samples.len(),
            ),
            // 2D (neuro-convert single-channel snippet units): [num_units, num_samples]
            &[n_u, n_s] if u_pos < n_u && n_s > 0 => {
                let off = u_pos * n_s;
                let mean = arr.data[off..off + n_s].to_vec();
                let std = loaded_sd
                    .as_ref()
                    .filter(|s| s.len() == arr.data.len())
                    .map(|s| s[off..off + n_s].to_vec())
                    .unwrap_or_else(|| vec![1.0; n_s]);
                let ch = primary_ch.unwrap_or(0);
                Some(WaveformTemplate::with_count(vec![ch], n_s, spike_samples.len(), mean, std))
            }
            _ => None,
        });

        let mut unit = SortedUnit::from_spikes_with(
            uid,
            primary_ch,
            spike_samples,
            u_amps,
            u_locs,
            template,
            fs,
            total_samples,
            None,
            criteria,
        );
        if let Some(&snr) = snrs.get(u_pos) {
            unit.snr = snr;
            unit.quality_label = criteria.classify(unit.num_spikes(), snr, unit.isi_violation_ratio);
        }
        if let Some(q_str) = quality_labels.get(u_pos) {
            unit.quality_label = UnitQualityLabel::parse(q_str);
        }
        units.push(unit);
    }

    let mut out = SortingOutput::new(
        sorter_name,
        fs,
        total_samples,
        probe,
        units,
        None,
    );
    if let Some(root) = root_nwb {
        out = out.with_recording_meta(RecordingMeta {
            dat_path: Some(root.to_string_lossy().into_owned()),
            ..Default::default()
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nwb_units_roundtrip_and_inferred_rate() {
        let dir = std::env::temp_dir().join(format!("dsp_nwb_units_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Include a timestamp past 2^24 samples to verify f64 precision is preserved
        let times = vec![300u64, 1500, 20_000_003];
        let t_mean = vec![0.0f32; 2 * 25];
        let template = WaveformTemplate::with_count(vec![0, 1], 25, 3, t_mean, vec![1.0; 50]);

        let unit0 = SortedUnit::from_spikes(0, 1, times.clone(), Vec::new(), Vec::new(), Some(template), 20_000.0, 20_001_000, 8.0);
        let orig = SortingOutput::new("nwb_sorter", 20_000.0, 20_001_000, None, vec![unit0], None);

        save_nwb_units(&orig, &dir).unwrap();
        assert!(dir.join("units").join("zarr.json").exists());
        assert!(dir.join("units").join("id").join("zarr.json").exists());
        assert!(dir.join("units").join("spike_times").join("zarr.json").exists());
        assert!(dir.join("units").join("spike_times_index").join("zarr.json").exists());

        // Load with None: sample rate (20_000.0 Hz) is inferred from Zarr metadata
        let loaded = load_nwb_units(&dir, None).unwrap();
        assert_eq!(loaded.sample_rate_hz, 20_000.0);
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].primary_channel, 1);
        assert_eq!(loaded.units[0].spike_samples, times);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
