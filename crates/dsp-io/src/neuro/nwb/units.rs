//! NWB `/units` DynamicTable in a `.nwb.zarr` store (hdmf-zarr layout, as `neuro-convert` writes):
//! - `id` `[K]` (`int64`)
//! - `spike_times` concatenated seconds (`float64`, `resolution = 1 / fs`) and `spike_times_index`
//!   `[K]` (`uint64`, cumulative end of each unit's spikes)
//! - optional `spike_amplitudes` / `spike_locations` (per spike, aligned with `spike_times`)
//! - optional `snr`, `firing_rate`, `primary_channel` (read also as `source_channel` /
//!   `electrodes`) `[K]`
//! - optional `waveform_mean`, `waveform_sd`, `waveform_se` `[K, C, T]` (or `[K, T]`)
//! - group attributes `sorter_name`, `sample_rate_hz`, `total_samples`, `quality_labels`, `probe`
//!   (written by this crate; read when present).
//!
//! [`NwbUnitsTable`] holds the columns as stored; nothing is computed.

use std::ops::Range;
use std::path::{Path, PathBuf};

use dsp_core::{DspError, DspResult};
use serde_json::{json, Map, Value};

use crate::container::npy::NpyArray;
use crate::container::zarr::{
    has_array, open_rw_store, read_array, read_node_attributes, read_optional_array, write_array_f32,
    write_array_f64, write_array_i64, write_array_u64, write_group,
};
use crate::neuro::probe::SensorLayout;

/// The `/units` table of an NWB store, column by column. See the module docs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NwbUnitsTable {
    /// NWB root (the store holding `/acquisition`) the table was read from, if any.
    pub nwb_root: Option<PathBuf>,
    pub sample_rate_hz: f64,
    pub sorter_name: Option<String>,
    pub total_samples: Option<u64>,
    pub ids: Vec<i64>,
    /// All units' spike times (s), unit after unit.
    pub spike_times_sec: Vec<f64>,
    /// End of each unit's spikes in [`Self::spike_times_sec`].
    pub spike_times_index: Vec<u64>,
    /// µV per spike.
    pub spike_amplitudes: Option<Vec<f32>>,
    /// `[x, y, z]` µm per spike.
    pub spike_locations: Option<Vec<[f32; 3]>>,
    pub snr: Option<Vec<f32>>,
    pub firing_rate: Option<Vec<f32>>,
    pub primary_channel: Option<Vec<usize>>,
    /// `[K, C, T]` or `[K, T]`.
    pub waveform_mean: Option<NpyArray<f32>>,
    /// Same shape as `waveform_mean`.
    pub waveform_sd: Option<Vec<f32>>,
    pub waveform_se: Option<Vec<f32>>,
    /// Quality label per unit (`good`, `mua`, …).
    pub quality_labels: Vec<String>,
    pub probe: Option<SensorLayout>,
}

/// Attribute `key` deserialized as `T` (missing or another type: `None`).
fn from_attr<T: serde::de::DeserializeOwned>(attrs: &Map<String, Value>, key: &str) -> Option<T> {
    attrs.get(key).and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn column_attrs(description: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("neurodata_type".into(), json!("VectorData"));
    m.insert("namespace".into(), json!("hdmf-common"));
    m.insert("description".into(), json!(description));
    m
}

impl NwbUnitsTable {
    /// Rows (units).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether `dir` holds a units table (an NWB root with `/units`, or the `units` group itself).
    pub fn is_units_table(dir: &Path) -> bool {
        has_array(dir, "/units/spike_times_index") || (has_array(dir, "/spike_times") && has_array(dir, "/spike_times_index"))
    }

    /// Spikes of row `u` (indices into the per-spike columns).
    pub fn spikes_of(&self, u: usize) -> Range<usize> {
        let end = self.spike_times_index.get(u).map_or(0, |&e| e as usize);
        let start = if u == 0 { 0 } else { self.spike_times_index.get(u - 1).map_or(0, |&e| e as usize) };
        start.min(end)..end
    }

    /// Reads the table at `dir` (an NWB root with `/units`, or the `units` group). The sample rate
    /// is `sample_rate_hz` when given, else inferred ([`infer_nwb_sample_rate`]); neither is an
    /// error.
    pub fn read(dir: &Path, sample_rate_hz: Option<f64>) -> DspResult<Self> {
        let prefix = if dir.join("units").exists() { "/units" } else { "" };
        let node = |name: &str| format!("{prefix}/{name}");
        if !has_array(dir, &node("id")) || !has_array(dir, &node("spike_times")) || !has_array(dir, &node("spike_times_index")) {
            return Err(DspError::UnsupportedFormat(format!(
                "NWB units group in {} missing id, spike_times, or spike_times_index",
                dir.display()
            )));
        }
        let nwb_root = if dir.join("acquisition").is_dir() {
            Some(dir.to_path_buf())
        } else {
            dir.parent().filter(|p| p.join("acquisition").is_dir()).map(Path::to_path_buf)
        };
        let sample_rate_hz = sample_rate_hz
            .filter(|&r| r > 0.0)
            .or_else(|| infer_nwb_sample_rate(nwb_root.as_deref().unwrap_or(dir)))
            .ok_or_else(|| {
                DspError::InvalidConfig(format!("{}: sample rate not found in NWB metadata and none was provided", dir.display()))
            })?;

        let attrs = read_node_attributes(dir, prefix).unwrap_or_default();
        let per_unit = |name: &str| read_optional_array::<f32>(dir, &node(name)).map(|a| a.data);
        let waveform_mean = read_optional_array::<f32>(dir, &node("waveform_mean"));
        let same_shape = |name: &str| per_unit(name).filter(|v| waveform_mean.as_ref().is_some_and(|m| m.data.len() == v.len()));

        Ok(Self {
            sample_rate_hz,
            sorter_name: from_attr(&attrs, "sorter_name"),
            total_samples: from_attr(&attrs, "total_samples"),
            ids: read_array::<i64>(dir, &node("id"))?.data,
            spike_times_sec: read_array::<f64>(dir, &node("spike_times"))?.data,
            spike_times_index: read_array::<u64>(dir, &node("spike_times_index"))?.data,
            spike_amplitudes: per_unit("spike_amplitudes"),
            spike_locations: read_optional_array::<f32>(dir, &node("spike_locations"))
                .filter(|a| a.shape.len() == 2 && a.shape[1] >= 3)
                .map(|a| a.data.chunks_exact(a.shape[1]).map(|r| [r[0], r[1], r[2]]).collect()),
            snr: per_unit("snr"),
            firing_rate: per_unit("firing_rate"),
            primary_channel: ["primary_channel", "source_channel", "electrodes"]
                .iter()
                .find_map(|n| read_optional_array::<usize>(dir, &node(n)))
                .map(|a| a.data),
            waveform_sd: same_shape("waveform_sd"),
            waveform_se: same_shape("waveform_se"),
            waveform_mean,
            quality_labels: from_attr(&attrs, "quality_labels").unwrap_or_default(),
            probe: from_attr(&attrs, "probe"),
            nwb_root,
        })
    }

    /// Writes the table as the `/units` group of the NWB store `nwb_zarr_dir` (created if needed).
    /// Per-spike columns are written only when they cover every spike.
    pub fn write(&self, nwb_zarr_dir: &Path) -> DspResult<()> {
        let dir = nwb_zarr_dir;
        let store = open_rw_store(dir)?;
        let fs = self.sample_rate_hz;
        let (rows, spikes) = (self.len(), self.spike_times_sec.len());
        let amps = self.spike_amplitudes.as_ref().filter(|a| a.len() == spikes && spikes > 0);
        let locs = self.spike_locations.as_ref().filter(|l| l.len() == spikes && spikes > 0);
        let per_row = |c: &Option<Vec<f32>>| c.as_ref().filter(|v| v.len() == rows).cloned();
        let (snr, firing_rate) = (per_row(&self.snr), per_row(&self.firing_rate));
        let primary = self.primary_channel.as_ref().filter(|v| v.len() == rows);

        let mut colnames = vec!["spike_times"];
        for (present, name) in [
            (snr.is_some(), "snr"),
            (firing_rate.is_some(), "firing_rate"),
            (primary.is_some(), "primary_channel"),
            (self.waveform_mean.is_some(), "waveform_mean"),
            (self.waveform_sd.is_some(), "waveform_sd"),
            (self.waveform_se.is_some(), "waveform_se"),
            (amps.is_some(), "spike_amplitudes"),
            (locs.is_some(), "spike_locations"),
        ] {
            if present {
                colnames.push(name);
            }
        }

        let sorter = self.sorter_name.as_deref().unwrap_or("unknown");
        let mut group = Map::new();
        group.insert("neurodata_type".into(), json!("Units"));
        group.insert("namespace".into(), json!("core"));
        group.insert("description".into(), json!(format!("Sorted neural units extracted by {sorter}")));
        group.insert("colnames".into(), json!(colnames));
        group.insert("sample_rate_hz".into(), json!(fs));
        group.insert("sorter_name".into(), json!(sorter));
        if let Some(total) = self.total_samples {
            group.insert("total_samples".into(), json!(total));
        }
        if !self.quality_labels.is_empty() {
            group.insert("quality_labels".into(), json!(self.quality_labels));
        }
        if let Some(probe) = &self.probe
            && let Ok(v) = serde_json::to_value(probe)
        {
            group.insert("probe".into(), v);
        }
        write_group(&store, dir, "/units", group)?;

        let mut id_attrs = Map::new();
        id_attrs.insert("neurodata_type".into(), json!("ElementIdentifiers"));
        id_attrs.insert("namespace".into(), json!("hdmf-common"));
        write_array_i64(&store, dir, "/units/id", &self.ids, &[rows], &["num_rows"], id_attrs)?;

        let mut st_attrs = column_attrs("the spike times for each unit in seconds");
        st_attrs.insert("resolution".into(), json!(1.0 / fs));
        st_attrs.insert("unit".into(), json!("seconds"));
        write_array_f64(&store, dir, "/units/spike_times", &self.spike_times_sec, &[spikes], &["num_spikes"], st_attrs)?;

        let mut idx_attrs = Map::new();
        idx_attrs.insert("neurodata_type".into(), json!("VectorIndex"));
        idx_attrs.insert("namespace".into(), json!("hdmf-common"));
        idx_attrs.insert("description".into(), json!("Index for VectorData 'spike_times'"));
        idx_attrs.insert("target".into(), json!({ "_REFERENCE": { "source": ".", "path": "/units/spike_times" } }));
        write_array_u64(&store, dir, "/units/spike_times_index", &self.spike_times_index, &[rows], &["num_rows"], idx_attrs)?;

        if let Some(a) = amps {
            write_array_f32(&store, dir, "/units/spike_amplitudes", a, &[spikes], &["num_spikes"], column_attrs("per-spike peak amplitudes in microvolts"))?;
        }
        if let Some(l) = locs {
            let flat: Vec<f32> = l.iter().flatten().copied().collect();
            write_array_f32(&store, dir, "/units/spike_locations", &flat, &[spikes, 3], &["num_spikes", "xyz"], column_attrs("per-spike 3D coordinates in micrometers"))?;
        }
        if let Some(v) = &snr {
            write_array_f32(&store, dir, "/units/snr", v, &[rows], &["num_rows"], column_attrs("unit peak-to-noise ratio"))?;
        }
        if let Some(v) = &firing_rate {
            write_array_f32(&store, dir, "/units/firing_rate", v, &[rows], &["num_rows"], column_attrs("mean firing rate in Hz"))?;
        }
        if let Some(v) = primary {
            let v: Vec<i64> = v.iter().map(|&c| c as i64).collect();
            write_array_i64(&store, dir, "/units/primary_channel", &v, &[rows], &["num_rows"], column_attrs("primary recording channel"))?;
        }
        if let Some(mean) = &self.waveform_mean {
            let dims: &[&str] = if mean.shape.len() == 3 { &["num_units", "num_channels", "num_samples"] } else { &["num_units", "num_samples"] };
            let mut wm_attrs = column_attrs("the spike waveform mean for each spike unit");
            wm_attrs.insert("sampling_rate".into(), json!(fs));
            wm_attrs.insert("unit".into(), json!("microvolts"));
            write_array_f32(&store, dir, "/units/waveform_mean", &mean.data, &mean.shape, dims, wm_attrs)?;
            for (name, buf, what) in [
                ("/units/waveform_sd", &self.waveform_sd, "spike waveform standard deviation"),
                ("/units/waveform_se", &self.waveform_se, "spike waveform standard error"),
            ] {
                if let Some(b) = buf.as_ref().filter(|b| b.len() == mean.data.len()) {
                    write_array_f32(&store, dir, name, b, &mean.shape, dims, column_attrs(what))?;
                }
            }
        }
        Ok(())
    }
}

/// Infers the recording sample rate (Hz) from an NWB Zarr store without hardcoding 30 kHz:
/// 1. `/units` `sample_rate_hz` or `sampling_rate` attribute
/// 2. `/units/waveform_mean` `sampling_rate` attribute
/// 3. `1.0 / resolution` on `/units/spike_times` (written by `neuro-convert`)
/// 4. Any `/acquisition/<series>/starting_time` `rate` attribute in the parent NWB store
pub fn infer_nwb_sample_rate(nwb_or_units_dir: &Path) -> Option<f64> {
    let (root_dir, units_prefix) = if nwb_or_units_dir.join("units").is_dir() {
        (nwb_or_units_dir, "/units")
    } else {
        (nwb_or_units_dir.parent().unwrap_or(nwb_or_units_dir), "")
    };
    let base = if units_prefix.is_empty() { nwb_or_units_dir } else { root_dir };

    if let Some(attrs) = read_node_attributes(base, units_prefix) {
        if let Some(sr) = attrs.get("sample_rate_hz").or_else(|| attrs.get("sampling_rate")).and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some(sr);
        }
    }
    let wm_path = format!("{units_prefix}/waveform_mean");
    if let Some(attrs) = read_node_attributes(base, &wm_path) {
        if let Some(sr) = attrs.get("sampling_rate").and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some(sr);
        }
    }
    let st_path = format!("{units_prefix}/spike_times");
    if let Some(attrs) = read_node_attributes(base, &st_path) {
        if let Some(res) = attrs.get("resolution").and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some((1.0 / res).round());
        }
    }
    if let Ok(entries) = std::fs::read_dir(root_dir.join("acquisition")) {
        for entry in entries.flatten() {
            let st = format!("/acquisition/{}/starting_time", entry.file_name().to_string_lossy());
            if let Some(attrs) = read_node_attributes(root_dir, &st) {
                if let Some(rate) = attrs.get("rate").and_then(Value::as_f64).filter(|&r| r > 0.0) {
                    return Some(rate);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trips() {
        let dir = std::env::temp_dir().join(format!("dsp_nwb_units_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let table = NwbUnitsTable {
            sample_rate_hz: 30_000.0,
            sorter_name: Some("test".into()),
            total_samples: Some(90_000),
            ids: vec![3, 7],
            spike_times_sec: vec![0.1, 0.5, 0.2],
            spike_times_index: vec![2, 3],
            spike_amplitudes: Some(vec![-50.0, -60.0, -80.0]),
            snr: Some(vec![5.0, 9.0]),
            primary_channel: Some(vec![0, 1]),
            waveform_mean: Some(NpyArray { data: vec![0.0, -1.0, 0.0, -2.0], shape: vec![2, 1, 2] }),
            quality_labels: vec!["good".into(), "mua".into()],
            ..Default::default()
        };
        table.write(&dir).unwrap();
        assert!(NwbUnitsTable::is_units_table(&dir));
        let back = NwbUnitsTable::read(&dir, None).unwrap();
        assert_eq!(back.spikes_of(1), 2..3);
        assert_eq!(back, table);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
