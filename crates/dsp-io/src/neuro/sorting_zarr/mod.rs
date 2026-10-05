//! `.sorting.zarr`: dsp-kitchen's own self-contained sorting store (Zarr v3). Not a SpikeInterface
//! layout.
//!
//! - root `zarr.json` attributes: the [`SortingZarrManifest`] (`dsp_format`, sorter, sample rate,
//!   length, recording provenance, probe, drift, one entry per unit)
//! - `/spikes`: `times` (`uint64` samples), `clusters` (`int32`), `amplitudes` (`float32`),
//!   optional `locations` `[spikes, 3]`
//! - `/templates`: optional `mean`, `std`, `se` `[units, channels, samples]` (unit row order;
//!   older stores indexed by unit id)
//!
//! [`SortingZarr`] holds the store as written; nothing is computed.

use std::path::Path;

use dsp_core::{DspError, DspResult};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::container::npy::NpyArray;
use crate::container::zarr::{
    has_array, open_rw_store, read_array, read_node_json, read_optional_array, write_array_f32, write_array_i32,
    write_array_u64, write_group,
};
use crate::neuro::probe::SensorLayout;

/// `dsp_format` written to the manifest.
pub const SORTING_ZARR_FORMAT: &str = "sorting_analyzer_v1";

/// `null` (how JSON stores NaN) read back as NaN.
fn nan_f32<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::NAN))
}

fn nan_f64<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// One unit of the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortingZarrUnit {
    pub unit_id: usize,
    pub primary_channel: usize,
    /// As written by the sorter (e.g. `SingleUnit`, `good`, `mua`).
    pub quality_label: String,
    #[serde(default, deserialize_with = "nan_f32")]
    pub snr: f32,
    #[serde(default, deserialize_with = "nan_f64")]
    pub firing_rate_hz: f64,
    #[serde(default, deserialize_with = "nan_f64")]
    pub isi_violation_ratio: f64,
    #[serde(default, deserialize_with = "nan_f64")]
    pub presence_ratio: f64,
    #[serde(default, deserialize_with = "nan_f64")]
    pub amplitude_cutoff: f64,
    pub num_spikes: usize,
}

/// Root attributes of a `.sorting.zarr` store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortingZarrManifest {
    pub dsp_format: String,
    pub sorter_name: String,
    pub sample_rate_hz: f64,
    pub total_samples: u64,
    /// Recording provenance (`dat_path`, `dtype`, …), kept as written.
    #[serde(default)]
    pub recording_meta: Value,
    pub probe: Option<SensorLayout>,
    /// Drift estimate, kept as written.
    pub drift: Option<Value>,
    pub units: Vec<SortingZarrUnit>,
}

/// A `.sorting.zarr` store. See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct SortingZarr {
    pub manifest: SortingZarrManifest,
    /// Per spike: sample, cluster (unit id), amplitude (µV, may be empty), location (µm, may be empty).
    pub spike_times: Vec<u64>,
    pub spike_clusters: Vec<i32>,
    pub amplitudes: Vec<f32>,
    pub locations: Vec<[f32; 3]>,
    /// `[units, channels, samples]`.
    pub templates_mean: Option<NpyArray<f32>>,
    /// Same shape as `templates_mean`.
    pub templates_std: Option<Vec<f32>>,
    pub templates_se: Option<Vec<f32>>,
}

impl SortingZarr {
    /// Whether `dir` is a `.sorting.zarr` store.
    pub fn is_sorting_zarr(dir: &Path) -> bool {
        dir.join("zarr.json").is_file() && has_array(dir, "/spikes/times")
    }

    /// Reads the store at `dir`.
    pub fn read(dir: &Path) -> DspResult<Self> {
        let root = read_node_json(dir, "/").ok_or_else(|| DspError::UnsupportedFormat(format!("Missing zarr.json in {}", dir.display())))?;
        let manifest = root.get("attributes").filter(|v| v.get("dsp_format").is_some()).unwrap_or(&root);
        let manifest: SortingZarrManifest =
            serde_json::from_value(manifest.clone()).map_err(|e| DspError::UnsupportedFormat(format!("{}: {e}", dir.display())))?;

        let templates_mean = read_optional_array::<f32>(dir, "/templates/mean").filter(|a| a.shape.len() == 3);
        let same_shape = |name: &str| {
            read_optional_array::<f32>(dir, name).map(|a| a.data).filter(|v| templates_mean.as_ref().is_some_and(|m| m.data.len() == v.len()))
        };
        Ok(Self {
            manifest,
            spike_times: read_array::<u64>(dir, "/spikes/times")?.data,
            spike_clusters: read_array::<i32>(dir, "/spikes/clusters")?.data,
            amplitudes: read_optional_array::<f32>(dir, "/spikes/amplitudes").map(|a| a.data).unwrap_or_default(),
            locations: read_optional_array::<f32>(dir, "/spikes/locations")
                .filter(|a| a.shape.get(1) == Some(&3))
                .map(|a| a.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
                .unwrap_or_default(),
            templates_std: same_shape("/templates/std"),
            templates_se: same_shape("/templates/se"),
            templates_mean,
        })
    }

    /// Writes the store into `dir` (created if needed).
    pub fn write(&self, dir: &Path) -> DspResult<()> {
        let store = open_rw_store(dir)?;
        let root = match serde_json::to_value(&self.manifest) {
            Ok(Value::Object(m)) => m,
            Ok(_) => Map::new(),
            Err(e) => return Err(DspError::InvalidConfig(format!("{}: manifest: {e}", dir.display()))),
        };
        write_group(&store, dir, "/", root)?;
        write_group(&store, dir, "/spikes", Map::new())?;
        write_group(&store, dir, "/templates", Map::new())?;

        let n = self.spike_times.len();
        write_array_u64(&store, dir, "/spikes/times", &self.spike_times, &[n], &["spikes"], Map::new())?;
        write_array_i32(&store, dir, "/spikes/clusters", &self.spike_clusters, &[self.spike_clusters.len()], &["spikes"], Map::new())?;
        write_array_f32(&store, dir, "/spikes/amplitudes", &self.amplitudes, &[self.amplitudes.len()], &["spikes"], Map::new())?;
        if !self.locations.is_empty() {
            let flat: Vec<f32> = self.locations.iter().flatten().copied().collect();
            write_array_f32(&store, dir, "/spikes/locations", &flat, &[self.locations.len(), 3], &["spikes", "xyz"], Map::new())?;
        }

        if let Some(mean) = &self.templates_mean {
            let dims = ["units", "channels", "samples"];
            write_array_f32(&store, dir, "/templates/mean", &mean.data, &mean.shape, &dims, Map::new())?;
            for (name, buf) in [("/templates/std", &self.templates_std), ("/templates/se", &self.templates_se)] {
                if let Some(b) = buf.as_ref().filter(|b| b.len() == mean.data.len()) {
                    write_array_f32(&store, dir, name, b, &mean.shape, &dims, Map::new())?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trips() {
        let dir = std::env::temp_dir().join(format!("dsp_sorting_zarr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SortingZarr {
            manifest: SortingZarrManifest {
                dsp_format: SORTING_ZARR_FORMAT.into(),
                sorter_name: "test".into(),
                sample_rate_hz: 30_000.0,
                total_samples: 5_000,
                recording_meta: Value::Null,
                probe: None,
                drift: None,
                units: vec![SortingZarrUnit {
                    unit_id: 5,
                    primary_channel: 1,
                    quality_label: "SingleUnit".into(),
                    snr: 10.0,
                    firing_rate_hz: 3.0,
                    isi_violation_ratio: 0.0,
                    presence_ratio: 1.0,
                    amplitude_cutoff: 0.0,
                    num_spikes: 2,
                }],
            },
            spike_times: vec![500, 1200],
            spike_clusters: vec![5, 5],
            amplitudes: vec![95.0, 105.0],
            locations: vec![[0.0, 10.0, 20.0]; 2],
            templates_mean: Some(NpyArray { data: vec![0.0, -1.0, 0.0, -2.0], shape: vec![1, 2, 2] }),
            templates_std: Some(vec![1.0; 4]),
            templates_se: None,
        };
        store.write(&dir).unwrap();
        assert!(SortingZarr::is_sorting_zarr(&dir));
        assert_eq!(SortingZarr::read(&dir).unwrap(), store);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
