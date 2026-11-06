//! `dspk.synapse.io`: probes (geometry, presets, probe files) and sortings (Phy, `.sorting.zarr`,
//! NWB units).

use std::path::PathBuf;

use dspk_core::{ChannelMap, ChannelMapFile};
use dspk_py_core::error::to_py;
use dspk_synapse::io::probe::{self, Position3D, SensorLayout, SensorSite};
use dspk_synapse::io::SortingFormat;
use dspk_synapse::{SortingOutput};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn map_json(map: &ChannelMap) -> String {
    serde_json::to_string(map).expect("a channel map serialises")
}

fn parse_map(text: &str) -> PyResult<ChannelMap> {
    // A bare map or a map file's {"remap": …}
    serde_json::from_str::<ChannelMapFile>(text).map(|f| f.remap).or_else(|_| serde_json::from_str::<ChannelMap>(text)).map_err(invalid)
}

/// A probe's geometry: one site per recording channel (positions in µm, shank per site).
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.synapse.io", frozen)]
#[derive(Clone)]
pub struct Probe {
    pub(crate) inner: SensorLayout,
}

#[gen_stub_pymethods]
#[pymethods]
impl Probe {
    /// A nominal layout: `neuropixels1`, `neuropixels2`, `tetrode`, `utah`, `hdemg4x8`,
    /// `hdemg8x8` (`pitch_um` for the HD-EMG grids).
    #[staticmethod]
    #[pyo3(signature = (name, *, pitch_um))]
    fn preset(name: &str, pitch_um: f32) -> PyResult<Self> {
        let inner = match name {
            "neuropixels1" => probe::neuropixels_1_0(),
            "neuropixels2" => probe::neuropixels_2_0(),
            "tetrode" => probe::tetrode(),
            "utah" => probe::utah_array(),
            "hdemg4x8" => probe::hdemg_4x8(pitch_um),
            "hdemg8x8" => probe::hdemg_8x8(pitch_um),
            other => return Err(invalid(format!("unknown preset {other:?} (neuropixels1, neuropixels2, tetrode, utah, hdemg4x8, hdemg8x8)"))),
        };
        Ok(Self { inner })
    }

    /// A `rows × cols` grid with `pitch_um` spacing; channel `row · cols + col`.
    #[staticmethod]
    fn grid(rows: usize, cols: usize, pitch_um: f32) -> Self {
        Self { inner: probe::hdemg_grid(&format!("{rows}×{cols} grid"), rows, cols, pitch_um) }
    }

    /// Sites at `positions` (`(channels, 2)` or `(channels, 3)` µm), channel `i` at row `i`;
    /// `shanks` per site (default: all 0).
    #[staticmethod]
    #[pyo3(signature = (positions, *, shanks, name))]
    fn from_positions(positions: PyReadonlyArray2<'_, f32>, shanks: Option<Vec<usize>>, name: &str) -> PyResult<Self> {
        let p = positions.as_array();
        if !(p.ncols() == 2 || p.ncols() == 3) {
            return Err(invalid(format!("positions are (channels, 2) or (channels, 3), got {:?}", p.shape())));
        }
        if let Some(s) = &shanks
            && s.len() != p.nrows()
        {
            return Err(invalid(format!("{} shanks for {} sites", s.len(), p.nrows())));
        }
        let sites = (0..p.nrows())
            .map(|i| {
                let z = if p.ncols() == 3 { p[[i, 2]] } else { 0.0 };
                SensorSite::new(i, Position3D::new(p[[i, 0]], p[[i, 1]], z), shanks.as_ref().map_or(0, |s| s[i]))
            })
            .collect();
        Ok(Self { inner: SensorLayout::new(name, sites) })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Number of sites.
    #[getter]
    fn channel_count(&self) -> usize {
        self.inner.total_channels()
    }

    /// The recording channel of every site, in site order.
    #[getter]
    fn channels(&self) -> Vec<usize> {
        self.inner.sites().iter().map(|s| s.channel_id).collect()
    }

    /// Site positions `(sites, 3)` in µm (x, y = depth, z).
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let sites = self.inner.sites();
        let flat: Vec<f32> = sites.iter().flat_map(|s| [s.position.x_um, s.position.y_um, s.position.z_um]).collect();
        Array2::from_shape_vec((sites.len(), 3), flat).expect("three per site").into_pyarray(py)
    }

    /// Shank of every site.
    #[getter]
    fn shanks(&self) -> Vec<usize> {
        self.inner.sites().iter().map(|s| s.shank_id).collect()
    }

    /// The geometry after a channel map (JSON of `{"base", "pairs"}`): site `i` where the map's
    /// input `order[i]` was.
    fn remap(&self, map: &str) -> PyResult<Self> {
        Ok(Self { inner: self.inner.remap(&parse_map(map)?).map_err(to_py)? })
    }

    /// A channel map (JSON) sorting the channels by shank, then depth, then x.
    fn depth_order(&self) -> String {
        map_json(&self.inner.depth_order())
    }

    /// The sites of `channels` (recording channels, in that order), renumbered `0..`.
    fn select(&self, channels: Vec<usize>) -> PyResult<Self> {
        if let Some(&c) = channels.iter().find(|&&c| self.inner.get_site(c).is_err()) {
            return Err(invalid(format!("channel {c} has no site on this probe")));
        }
        Ok(Self { inner: self.inner.select_channels(&channels) })
    }

    /// The `k` nearest sites of every channel (itself first), `(channels, k)`; padded with the
    /// channel itself where fewer exist.
    fn neighbours<'py>(&self, py: Python<'py>, k: usize) -> Bound<'py, PyArray2<u32>> {
        let n = self.inner.total_channels();
        let k = k.max(1);
        Array2::from_shape_vec((n, k), probe::precompute_knn_table(&self.inner, n, k)).expect("k per channel").into_pyarray(py)
    }

    /// The probe as JSON (round-trips through [`Probe::from_json`]).
    fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).expect("a layout serialises")
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self { inner: serde_json::from_str(text).map_err(invalid)? })
    }

    fn __len__(&self) -> usize {
        self.inner.total_channels()
    }

    fn __repr__(&self) -> String {
        let shanks = self.inner.sites().iter().map(|s| s.shank_id).max().map_or(0, |m| m + 1);
        format!("Probe(name={:?}, sites={}, shanks={shanks})", self.inner.name, self.inner.total_channels())
    }
}

/// A probe file (`.json`: probeinterface or Kilosort 4): the probe in contact order and the channel
/// map (JSON) that reads the recording in that order. `.mat` files are refused.
#[gen_stub_pyfunction(module = "dspk._native.synapse.io")]
#[pyfunction]
pub fn read_probe_file(path: PathBuf) -> PyResult<(Probe, String)> {
    let w = probe::read_probe_wiring(&path).map_err(to_py)?;
    Ok((Probe { inner: w.layout }, map_json(&w.map)))
}

/// The probe a recording file carries (SpikeGLX), for source `source` (default: the file's
/// default source); `None` when its format has none.
#[gen_stub_pyfunction(module = "dspk._native.synapse.io")]
#[pyfunction]
#[pyo3(signature = (path, source=None))]
pub fn probe_of(path: PathBuf, source: Option<String>) -> PyResult<Option<Probe>> {
    let id = match source {
        Some(s) => s,
        None => {
            let list = dspk_synapse::io::sources(&path).map_err(to_py)?;
            dspk_io::default_source(&list).ok_or_else(|| invalid(format!("{} has no sources", path.display())))?.id.clone()
        }
    };
    Ok(probe::probe_of(&path, &id).map_err(to_py)?.map(|inner| Probe { inner }))
}

/// A sorting: units with their spike trains, templates and quality metrics.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.synapse.io", frozen)]
pub struct Sorting {
    pub(crate) inner: SortingOutput,
}

fn format_of(name: Option<&str>) -> PyResult<Option<SortingFormat>> {
    match name {
        None => Ok(None),
        Some("phy") => Ok(Some(SortingFormat::Phy)),
        Some("sorting_zarr") => Ok(Some(SortingFormat::SortingZarr)),
        Some("nwb_units") => Ok(Some(SortingFormat::NwbUnits)),
        Some(other) => Err(invalid(format!("unknown sorting format {other:?} (phy, sorting_zarr, nwb_units)"))),
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Sorting {
    /// Loads a sorting: a Phy / Kilosort folder, a `.sorting.zarr` store or an NWB `.nwb.zarr`
    /// (its `/units`), detected from the contents.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| dspk_synapse::storage::load_sorting(&path)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// A sorting from flat arrays: `spike_samples` (sample index of every spike) and `spike_units`
    /// (its unit); metrics are computed from the trains.
    #[staticmethod]
    #[pyo3(signature = (spike_samples, spike_units, sample_rate, total_samples, *, sorter_name, probe=None))]
    fn from_spikes(spike_samples: Vec<u64>, spike_units: Vec<usize>, sample_rate: f64, total_samples: u64, sorter_name: &str, probe: Option<&Probe>) -> PyResult<Self> {
        if spike_samples.len() != spike_units.len() {
            return Err(invalid(format!("{} spike samples but {} units", spike_samples.len(), spike_units.len())));
        }
        let mut by_unit: std::collections::BTreeMap<usize, Vec<u64>> = Default::default();
        for (s, u) in spike_samples.into_iter().zip(spike_units) {
            by_unit.entry(u).or_default().push(s);
        }
        let units = by_unit
            .into_iter()
            .map(|(u, mut samples)| {
                samples.sort_unstable();
                dspk_synapse::SortedUnit::from_spikes_with(u, None, samples, Vec::new(), Vec::new(), None, sample_rate, total_samples, None, Default::default())
            })
            .collect();
        Ok(Self { inner: SortingOutput::new(sorter_name, sample_rate, total_samples, probe.map(|p| p.inner.clone()), units, None) })
    }

    /// Saves as `format` (`phy`, `sorting_zarr`, `nwb_units`; default: from the path's name).
    #[pyo3(signature = (path, format=None))]
    fn save(&self, py: Python<'_>, path: PathBuf, format: Option<&str>) -> PyResult<()> {
        let format = format_of(format)?;
        py.detach(|| dspk_synapse::storage::save_sorting(&self.inner, &path, format)).map_err(to_py)
    }

    #[getter]
    fn sorter_name(&self) -> &str {
        &self.inner.sorter_name
    }

    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    #[getter]
    fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }

    /// Unit ids, in order.
    #[getter]
    fn unit_ids(&self) -> Vec<usize> {
        self.inner.units.iter().map(|u| u.unit_id).collect()
    }

    /// The probe the sorting carries, if any.
    #[getter]
    fn probe(&self) -> Option<Probe> {
        self.inner.probe.clone().map(|inner| Probe { inner })
    }

    /// Every spike, sorted by time: `samples` (u64), `units` (i32), `amplitudes` (µV, NaN where
    /// unknown), `locations` (`(spikes, 3)` µm, NaN where unknown).
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (samples, units, amplitudes, locations) = self.inner.flattened_spikes();
        let out = PyDict::new(py);
        let n = samples.len();
        out.set_item("samples", samples.into_pyarray(py))?;
        out.set_item("units", units.into_pyarray(py))?;
        out.set_item("amplitudes", amplitudes.into_pyarray(py))?;
        let flat: Vec<f32> = locations.into_iter().flatten().collect();
        out.set_item("locations", Array2::from_shape_vec((n, 3), flat).expect("three per spike").into_pyarray(py))?;
        Ok(out)
    }

    /// The spike train of `unit_id` (sample indices, sorted).
    fn spike_train<'py>(&self, py: Python<'py>, unit_id: usize) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let unit = self.inner.unit(unit_id).ok_or_else(|| invalid(format!("no unit {unit_id}")))?;
        Ok(unit.spike_samples.clone().into_pyarray(py))
    }

    /// The template of `unit_id`: `mean`, `std`, `se` `(channels, samples)` µV on `channels`, the
    /// spike `count`, the `trough_index`; `None` without a template.
    fn template<'py>(&self, py: Python<'py>, unit_id: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        let unit = self.inner.unit(unit_id).ok_or_else(|| invalid(format!("no unit {unit_id}")))?;
        let Some(t) = &unit.template else { return Ok(None) };
        let out = PyDict::new(py);
        let shape = (t.num_channels, t.num_samples);
        for (key, values) in [("mean", &t.mean), ("std", &t.std), ("se", &t.se)] {
            if values.len() == shape.0 * shape.1 {
                out.set_item(key, Array2::from_shape_vec(shape, values.clone()).expect("shape checked").into_pyarray(py))?;
            }
        }
        out.set_item("channels", t.channel_ids.clone())?;
        out.set_item("count", t.count)?;
        out.set_item("trough_index", t.trough_index)?;
        Ok(Some(out))
    }

    /// One row per unit: `unit_id`, `primary_channel`, `spikes`, `firing_rate_hz`, `snr`,
    /// `isi_violation_ratio`, `presence_ratio`, `amplitude_cutoff`, `composite_score`, `label`
    /// (columns as arrays / lists).
    fn units<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let u = &self.inner.units;
        let out = PyDict::new(py);
        out.set_item("unit_id", u.iter().map(|x| x.unit_id as u64).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("primary_channel", u.iter().map(|x| x.primary_channel as u64).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("spikes", u.iter().map(|x| x.spike_samples.len() as u64).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("firing_rate_hz", u.iter().map(|x| x.firing_rate_hz).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("snr", u.iter().map(|x| x.snr).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("isi_violation_ratio", u.iter().map(|x| x.isi_violation_ratio).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("presence_ratio", u.iter().map(|x| x.presence_ratio).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("amplitude_cutoff", u.iter().map(|x| x.amplitude_cutoff).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("composite_score", u.iter().map(|x| x.composite_score).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item("label", u.iter().map(|x| format!("{:?}", x.quality_label)).collect::<Vec<_>>())?;
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.inner.num_units()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sorting(sorter={:?}, units={}, spikes={}, sample_rate={} Hz, duration={:.1} s)",
            self.inner.sorter_name,
            self.inner.num_units(),
            self.inner.total_spikes(),
            self.inner.sample_rate_hz,
            self.inner.total_samples as f64 / self.inner.sample_rate_hz
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Probe>()?;
    m.add_class::<Sorting>()?;
    m.add_function(wrap_pyfunction!(read_probe_file, m)?)?;
    m.add_function(wrap_pyfunction!(probe_of, m)?)?;
    Ok(())
}
