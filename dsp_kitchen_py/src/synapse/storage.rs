//! PyO3 bindings for sorting output storage (`SortingOutput`, Phy, Zarr, NWB `/units`).

use std::path::Path;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{to_numpy, to_numpy_u64, F32Array};
use dsp_io::neuro::SortingFormat;
use dsp_synapse::core::sorting_output::MISSING_AMPLITUDE;
use dsp_synapse::core::SortingOutput;
use dsp_synapse::sorting::MotorUnitPulseTrain;
use dsp_synapse::storage::{load_nwb_units as rust_load_nwb, load_sorting as rust_load_sorting, save_sorting as rust_save_sorting};

/// Coordinates per location (x, y, z).
const COORDS: usize = 3;

use super::probe::PyProbeLayout;

/// `format=`: `"phy"` (Phy / Kilosort folder), `"sorting-zarr"` (dsp-kitchen `.sorting.zarr`) or
/// `"nwb-units"` (the `/units` table of a `.nwb.zarr` store); `None` = from the path's name.
fn parse_format(fmt: Option<&str>) -> PyResult<Option<SortingFormat>> {
    fmt.map(|f| match f {
        "phy" => Ok(SortingFormat::Phy),
        "sorting-zarr" => Ok(SortingFormat::SortingZarr),
        "nwb-units" => Ok(SortingFormat::NwbUnits),
        other => Err(PyValueError::new_err(format!("format must be 'phy', 'sorting-zarr' or 'nwb-units', got '{other}'"))),
    })
    .transpose()
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

    /// Spike amplitudes of `unit_id` (NaN where unknown).
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
        to_numpy(py, flat, &[n, COORDS])
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

    /// Loads a sorting output from `path` (Phy folder, `.sorting.zarr`, or `.nwb.zarr`).
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = rust_load_sorting(Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// A sorting from clustered spikes: `spike_samples` and `labels` (one per spike; `-1` =
    /// unassigned) over a recording of `total_samples` at `fs` Hz. Each spike's primary channel
    /// comes from `snippets` (which also give templates) or `primary_channels`; amplitudes are
    /// NaN unless given.
    #[staticmethod]
    #[pyo3(signature = (sorter_name, spike_samples, labels, *, fs, total_samples, primary_channels=None, amplitudes=None, locations=None, probe=None, snippets=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_clusters(
        sorter_name: &str,
        spike_samples: Vec<u64>,
        labels: Vec<i32>,
        fs: f64,
        total_samples: u64,
        primary_channels: Option<Vec<usize>>,
        amplitudes: Option<Vec<f32>>,
        locations: Option<Vec<[f32; 3]>>,
        probe: Option<PyRef<'_, PyProbeLayout>>,
        snippets: Option<Vec<PyRef<'_, super::extraction::PyWaveformSnippet>>>,
    ) -> PyResult<Self> {
        let n = spike_samples.len();
        let lengths_ok = labels.len() == n
            && amplitudes.as_ref().is_none_or(|a| a.len() == n)
            && locations.as_ref().is_none_or(|l| l.len() == n)
            && primary_channels.as_ref().is_none_or(|p| p.len() == n)
            && snippets.as_ref().is_none_or(|s| s.len() == n);
        if !lengths_ok {
            return Err(PyValueError::new_err("labels, amplitudes, locations, primary_channels and snippets need one entry per spike"));
        }
        let snippets: Option<Vec<dsp_synapse::core::WaveformSnippet>> = snippets.map(|s| s.iter().map(|s| s.inner.clone()).collect());
        let channels: Vec<(usize, Vec<usize>)> = match (&snippets, primary_channels) {
            (Some(s), _) => s.iter().map(|s| (s.primary_channel, s.channel_ids.clone())).collect(),
            (None, Some(p)) => p.into_iter().map(|c| (c, vec![c])).collect(),
            (None, None) => return Err(PyValueError::new_err("give snippets or primary_channels (each spike's channel)")),
        };
        let spikes: Vec<dsp_synapse::core::DeduplicatedSpike> = spike_samples
            .iter()
            .zip(channels)
            .enumerate()
            .map(|(i, (&sample, (primary, participating)))| dsp_synapse::core::DeduplicatedSpike {
                sample_index: sample,
                primary_channel: primary,
                peak_amplitude_uv: amplitudes.as_ref().map_or(MISSING_AMPLITUDE, |a| a[i]),
                participating_channels: participating,
            })
            .collect();
        let inner = SortingOutput::from_clustered_spikes(sorter_name, fs, total_samples, probe.map(|p| p.inner.clone()), &spikes, &labels, snippets.as_deref(), locations.as_deref(), &[], None);
        Ok(Self { inner })
    }

    /// Constructs a `SortingOutput` from Convolutive BSS motor unit pulse trains.
    #[staticmethod]
    #[pyo3(signature = (sorter_name, cbss_units, *, fs, total_samples, probe=None))]
    pub fn from_cbss(
        sorter_name: &str,
        cbss_units: Bound<'_, PyList>,
        fs: f64,
        total_samples: u64,
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
            // The innovation pulse train is optional; when given it must be numeric
            let ipt: Vec<f32> = match d.get_item("ipt")? {
                Some(obj) => F32Array::new(&obj)?.slice().to_vec(),
                None => Vec::new(),
            };

            rust_units.push(MotorUnitPulseTrain {
                unit_id,
                spike_samples,
                pnr_db,
                cov_isi,
                ipt,
            });
        }

        let probe_inner = probe.map(|p| p.inner.clone());
        let inner = SortingOutput::from_motor_units(
            sorter_name,
            fs,
            total_samples,
            probe_inner,
            &rust_units,
        );
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SortingOutput(sorter='{}', units={}, total_spikes={}, sample_rate={:.1} Hz, total_samples={})",
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

/// Loads the `/units` table of an NWB Zarr store; `fs` sets the sample rate its spike times
/// (seconds) are converted with, when the store does not give it.
#[pyfunction]
#[pyo3(signature = (nwb_zarr_path, *, fs=None))]
pub fn load_nwb_units(nwb_zarr_path: &str, fs: Option<f64>) -> PyResult<PySortingOutput> {
    let inner = rust_load_nwb(Path::new(nwb_zarr_path), fs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PySortingOutput { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySortingOutput>()?;
    m.add_function(wrap_pyfunction!(save_sorting, m)?)?;
    m.add_function(wrap_pyfunction!(load_sorting, m)?)?;
    m.add_function(wrap_pyfunction!(load_nwb_units, m)?)?;
    Ok(())
}
