//! PyO3 bindings for sorting output storage (`SortingOutput`, Phy, Zarr, NWB `/units`).

use std::path::Path;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{to_numpy, to_numpy_u64, F32Array};
use dsp_synapse::core::SortingOutput;
use dsp_synapse::sorting::MotorUnitPulseTrain;
use dsp_synapse::storage::{
    load_nwb_units as rust_load_nwb, load_phy_folder as rust_load_phy,
    load_sorting as rust_load_sorting, save_nwb_units as rust_save_nwb,
    save_phy_folder as rust_save_phy, save_sorting as rust_save_sorting, SortingFormat,
};

use super::probe::PyProbeLayout;

fn parse_format(fmt: Option<&str>) -> PyResult<Option<SortingFormat>> {
    match fmt {
        None => Ok(None),
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "phy" | "kilosort" => Ok(Some(SortingFormat::Phy)),
            "zarr" | "sorting.zarr" | "sorting_zarr" | "zarr_analyzer" => {
                Ok(Some(SortingFormat::ZarrAnalyzer))
            }
            "nwb" | "nwb.zarr" | "nwb_units" => Ok(Some(SortingFormat::NwbUnits)),
            other => Err(PyValueError::new_err(format!(
                "Unknown format '{other}'. Expected 'phy', 'zarr', or 'nwb'."
            ))),
        },
    }
}

/// Unified, format-agnostic container holding spike trains, waveform templates, and quality metrics.
#[pyclass(name = "SortingOutput")]
pub struct PySortingOutput {
    pub(crate) inner: SortingOutput,
}

impl PySortingOutput {
    pub fn new(inner: SortingOutput) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &SortingOutput {
        &self.inner
    }
}

#[pymethods]
impl PySortingOutput {
    #[getter]
    pub fn sorter_name(&self) -> &str {
        &self.inner.sorter_name
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    #[getter]
    pub fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }

    #[getter]
    pub fn num_units(&self) -> usize {
        self.inner.num_units()
    }

    #[getter]
    pub fn total_spikes(&self) -> usize {
        self.inner.total_spikes()
    }

    #[getter]
    pub fn probe(&self) -> Option<PyProbeLayout> {
        self.inner.probe.as_ref().map(|p| PyProbeLayout {
            inner: p.clone(),
        })
    }

    /// List of all integer unit IDs.
    pub fn unit_ids(&self) -> Vec<usize> {
        self.inner.units.iter().map(|u| u.unit_id).collect()
    }

    /// Returns spike timestamps in sample indices for `unit_id`.
    pub fn spike_train<'py>(&self, py: Python<'py>, unit_id: usize) -> PyResult<Bound<'py, PyAny>> {
        let unit = self
            .inner
            .unit(unit_id)
            .ok_or_else(|| PyValueError::new_err(format!("Unit ID {unit_id} not found")))?;
        let n = unit.spike_samples.len();
        to_numpy_u64(py, unit.spike_samples.clone(), &[n])
    }

    /// Returns spike amplitudes in $\mu\text{V}$ for `unit_id`.
    pub fn spike_amplitudes<'py>(
        &self,
        py: Python<'py>,
        unit_id: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let unit = self
            .inner
            .unit(unit_id)
            .ok_or_else(|| PyValueError::new_err(format!("Unit ID {unit_id} not found")))?;
        let n = unit.amplitudes_uv.len();
        to_numpy(py, unit.amplitudes_uv.clone(), &[n])
    }

    /// Returns 3D spike coordinates `[N, 3]` in $\mu\text{m}$ for `unit_id`.
    pub fn spike_locations<'py>(
        &self,
        py: Python<'py>,
        unit_id: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let unit = self
            .inner
            .unit(unit_id)
            .ok_or_else(|| PyValueError::new_err(format!("Unit ID {unit_id} not found")))?;
        let n = unit.locations_um.len();
        let flat: Vec<f32> = unit.locations_um.iter().flatten().copied().collect();
        to_numpy(py, flat, &[n, 3])
    }

    /// Returns the multi-channel waveform template (`mean`, `std`, `se`, `channel_ids`).
    pub fn unit_template<'py>(
        &self,
        py: Python<'py>,
        unit_id: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let unit = self
            .inner
            .unit(unit_id)
            .ok_or_else(|| PyValueError::new_err(format!("Unit ID {unit_id} not found")))?;
        let Some(t) = &unit.template else {
            return Ok(None);
        };

        let dict = PyDict::new(py);
        let shape = &[t.num_channels, t.num_samples];
        dict.set_item("mean", to_numpy(py, t.mean.clone(), shape)?)?;
        dict.set_item("std", to_numpy(py, t.std.clone(), shape)?)?;
        dict.set_item("se", to_numpy(py, t.se.clone(), shape)?)?;
        dict.set_item("channel_ids", t.channel_ids.clone())?;
        dict.set_item("num_channels", t.num_channels)?;
        dict.set_item("num_samples", t.num_samples)?;
        dict.set_item("count", t.count)?;
        Ok(Some(dict))
    }

    /// Returns quality metrics dictionary for `unit_id`.
    pub fn unit_metrics<'py>(&self, py: Python<'py>, unit_id: usize) -> PyResult<Bound<'py, PyDict>> {
        let unit = self
            .inner
            .unit(unit_id)
            .ok_or_else(|| PyValueError::new_err(format!("Unit ID {unit_id} not found")))?;

        let dict = PyDict::new(py);
        dict.set_item("unit_id", unit.unit_id)?;
        dict.set_item("primary_channel", unit.primary_channel)?;
        dict.set_item("quality_label", unit.quality_label.as_str())?;
        dict.set_item("snr", unit.snr)?;
        dict.set_item("firing_rate_hz", unit.firing_rate_hz)?;
        dict.set_item("isi_violation_ratio", unit.isi_violation_ratio)?;
        dict.set_item("presence_ratio", unit.presence_ratio)?;
        dict.set_item("amplitude_cutoff", unit.amplitude_cutoff)?;
        dict.set_item("num_spikes", unit.spike_samples.len())?;
        Ok(dict)
    }

    /// Returns a list of metric dictionaries for all units in this sorting.
    pub fn summary_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for u in &self.inner.units {
            list.append(self.unit_metrics(py, u.unit_id)?)?;
        }
        Ok(list)
    }

    /// Saves the sorting output to `path` in format `format` (or auto-detected from path).
    #[pyo3(signature = (path, format=None))]
    pub fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        let fmt = parse_format(format)?;
        rust_save_sorting(&self.inner, Path::new(path), fmt)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Exports sorting output as a flat Phy / Kilosort compatible folder.
    pub fn export_to_phy(&self, folder: &str) -> PyResult<()> {
        rust_save_phy(&self.inner, Path::new(folder))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads a sorting output from `path` (Phy folder, `.sorting.zarr`, or `.nwb.zarr`).
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = rust_load_sorting(Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Constructs a `SortingOutput` from clustered spike timestamps and cluster labels.
    #[staticmethod]
    #[pyo3(signature = (sorter_name, spike_samples, labels, sample_rate_hz, total_samples=None, amplitudes=None, locations=None, probe=None, snippets=None))]
    pub fn from_clusters(
        sorter_name: &str,
        spike_samples: Vec<u64>,
        labels: Vec<i32>,
        sample_rate_hz: f64,
        total_samples: Option<u64>,
        amplitudes: Option<Vec<f32>>,
        locations: Option<Vec<[f32; 3]>>,
        probe: Option<PyRef<'_, PyProbeLayout>>,
        snippets: Option<Vec<PyRef<'_, super::extraction::PyWaveformSnippet>>>,
    ) -> PyResult<Self> {
        let tot = total_samples.unwrap_or_else(|| {
            spike_samples.iter().copied().max().unwrap_or(0) + 1
        });
        let probe_inner = probe.map(|p| p.inner.clone());

        let rust_snippets: Option<Vec<dsp_synapse::core::WaveformSnippet>> = snippets.map(|snips| {
            snips.iter().map(|s| s.inner.clone()).collect()
        });

        let dedup: Vec<dsp_synapse::core::DeduplicatedSpike> = if let Some(snips) = &rust_snippets {
            snips
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let amp = amplitudes.as_ref().and_then(|a| a.get(i).copied()).unwrap_or_else(|| {
                        s.waveform
                            .iter()
                            .copied()
                            .map(f32::abs)
                            .max_by(|a, b| a.total_cmp(b))
                            .unwrap_or(50.0)
                    });
                    dsp_synapse::core::DeduplicatedSpike {
                        sample_index: s.center_sample,
                        primary_channel: s.primary_channel,
                        peak_amplitude_uv: amp,
                        participating_channels: s.channel_ids.clone(),
                    }
                })
                .collect()
        } else {
            spike_samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let amp = amplitudes.as_ref().and_then(|a| a.get(i).copied()).unwrap_or(50.0);
                    dsp_synapse::core::DeduplicatedSpike {
                        sample_index: s,
                        primary_channel: 0,
                        peak_amplitude_uv: amp,
                        participating_channels: vec![0],
                    }
                })
                .collect()
        };

        let inner = SortingOutput::from_clustered_spikes(
            sorter_name,
            sample_rate_hz,
            tot,
            probe_inner,
            &dedup,
            &labels,
            rust_snippets.as_deref(),
            locations.as_deref(),
            &[],
            None,
        );
        Ok(Self { inner })
    }

    /// Constructs a `SortingOutput` from Convolutive BSS motor unit pulse trains.
    #[staticmethod]
    #[pyo3(signature = (sorter_name, cbss_units, sample_rate_hz, total_samples=None, probe=None))]
    pub fn from_cbss(
        sorter_name: &str,
        cbss_units: Bound<'_, PyList>,
        sample_rate_hz: f64,
        total_samples: Option<u64>,
        probe: Option<PyRef<'_, PyProbeLayout>>,
    ) -> PyResult<Self> {
        let mut rust_units = Vec::with_capacity(cbss_units.len());
        for item in cbss_units.iter() {
            let d = item.extract::<Bound<'_, PyDict>>()?;
            let unit_id: usize = d.get_item("unit_id")?.ok_or_else(|| {
                PyValueError::new_err("Missing unit_id in cbss_units item")
            })?.extract()?;
            let spike_samples: Vec<u64> = d.get_item("spike_samples")?.ok_or_else(|| {
                PyValueError::new_err("Missing spike_samples in cbss_units item")
            })?.extract()?;
            let pnr_db: f32 = d.get_item("pnr_db")?.ok_or_else(|| {
                PyValueError::new_err("Missing pnr_db in cbss_units item")
            })?.extract()?;
            let cov_isi: f32 = d.get_item("cov_isi")?.ok_or_else(|| {
                PyValueError::new_err("Missing cov_isi in cbss_units item")
            })?.extract()?;
            let ipt: Vec<f32> = if let Some(ipt_obj) = d.get_item("ipt")? {
                if let Ok(arr) = F32Array::new(&ipt_obj) {
                    arr.slice().to_vec()
                } else if let Ok(v) = ipt_obj.extract::<Vec<f32>>() {
                    v
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            rust_units.push(MotorUnitPulseTrain {
                unit_id,
                spike_samples,
                pnr_db,
                cov_isi,
                ipt,
            });
        }

        let tot = total_samples.unwrap_or_else(|| {
            rust_units
                .iter()
                .flat_map(|u| u.spike_samples.iter().copied())
                .max()
                .unwrap_or(0)
                + 1
        });
        let probe_inner = probe.map(|p| p.inner.clone());
        let inner = SortingOutput::from_motor_units(
            sorter_name,
            sample_rate_hz,
            tot,
            probe_inner,
            &rust_units,
        );
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SortingOutput(sorter='{}', units={}, total_spikes={}, sample_rate={:.1}Hz, total_samples={})",
            self.inner.sorter_name,
            self.inner.num_units(),
            self.inner.total_spikes(),
            self.inner.sample_rate_hz,
            self.inner.total_samples
        )
    }
}

/// Saves a `SortingOutput` to `path` with automatic or explicit format selection.
#[pyfunction]
#[pyo3(signature = (sorting, path, format=None))]
pub fn save_sorting(
    sorting: &PySortingOutput,
    path: &str,
    format: Option<&str>,
) -> PyResult<()> {
    sorting.save(path, format)
}

/// Loads a `SortingOutput` from `path` (Phy folder, `.sorting.zarr`, or `.nwb.zarr`).
#[pyfunction]
pub fn load_sorting(path: &str) -> PyResult<PySortingOutput> {
    PySortingOutput::load(path)
}

/// Exports a `SortingOutput` to a Phy / Kilosort directory.
#[pyfunction]
pub fn export_to_phy(sorting: &PySortingOutput, folder: &str) -> PyResult<()> {
    sorting.export_to_phy(folder)
}

/// Reads a Kilosort / Phy directory into a `SortingOutput`.
#[pyfunction]
#[pyo3(signature = (folder, sample_rate_hz=None))]
pub fn read_kilosort(folder: &str, sample_rate_hz: Option<f64>) -> PyResult<PySortingOutput> {
    let mut out = rust_load_phy(Path::new(folder))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if let Some(sr) = sample_rate_hz {
        out.sample_rate_hz = sr;
    }
    Ok(PySortingOutput { inner: out })
}

/// Saves a `SortingOutput` to the `/units` DynamicTable of an NWB Zarr store.
#[pyfunction]
pub fn save_nwb_units(sorting: &PySortingOutput, nwb_zarr_path: &str) -> PyResult<()> {
    rust_save_nwb(&sorting.inner, Path::new(nwb_zarr_path))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Loads a `SortingOutput` from the `/units` DynamicTable of an NWB Zarr store.
#[pyfunction]
#[pyo3(signature = (nwb_zarr_path, sample_rate_hz=None))]
pub fn load_nwb_units(nwb_zarr_path: &str, sample_rate_hz: Option<f64>) -> PyResult<PySortingOutput> {
    let inner = rust_load_nwb(Path::new(nwb_zarr_path), sample_rate_hz)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PySortingOutput { inner })
}
