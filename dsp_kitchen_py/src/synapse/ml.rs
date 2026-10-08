//! `dsp_kitchen.synapse.ml`: sorters reimplemented from their papers (`dsp_synapse_ml::sorters`:
//! Kilosort4 and its fork EMUsort), each with its provenance, and the catalog of published
//! artifacts (`ModelHub`, feature `hub`). Settings default to the Rust defaults (the upstream
//! ones); stages that run on the device take `runtime=`.

use std::path::Path;

use cubecl::prelude::Client;
use cubecl::CubeElement;
use dsp_core::compute::ComputeTask;
use dsp_synapse_ml::sorters::emusort::delays::{ChannelAligner, ChannelDelayEstimator};
use dsp_synapse_ml::sorters::emusort::{emusort_provenance as emusort_record, EmusortConfig};
use dsp_synapse_ml::sorters::kilosort4::{
    detect_universal as detect, extract_clips as clips_of, fit_kilosort4_preprocessing,
    kilosort4_provenance as kilosort4_record, learn_universal_templates as learn,
    run_plan, CentreOptions, Kilosort4Config, Kilosort4Result, LearnOptions, RunPlan,
    TemplateCentres, UniversalSpike, UniversalTemplates,
};
use dsp_synapse_ml::Provenance;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3::types::{PyDict, PyList};

use super::probe::PyProbeLayout;
use super::storage::PySortingOutput;
use crate::array::{runtime_error, to_numpy, value_error, F32Array};
use crate::pipeline::engine::PyPipeline;
use crate::runtime::target;

// ------------------------------------------------------------------------------------------------
// Configuration
// ------------------------------------------------------------------------------------------------

/// Kilosort4 settings, under their upstream names (`kilosort/parameters.py`) where there is one.
///
/// Every argument defaults to Kilosort4's published default; thresholds are in whitened σ, lengths
/// of time in **samples** (they depend on the sampling rate), distances in µm. Change any setting
/// by keyword, or later as an attribute.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.synapse.ml import kilosort4
/// >>> config = kilosort4.Config(nt=121, th_learned=7.0)
/// >>> config.highpass_cutoff_hz = 250.0
#[gen_stub_pyclass]
#[pyclass(name = "Kilosort4Config", get_all, set_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyKilosort4Config {
    /// Samples per waveform window (odd; `nt` upstream). In samples, so it depends on the sampling rate: 61 is 2 ms at 30 kHz.
    nt: usize,
    /// Sample of the window the waveform trough is aligned to; `None`: `int(20 · nt / 61)` (about a third of `nt`).
    nt0min: Option<usize>,
    /// Universal-template detection threshold, in whitened σ (`Th_universal`).
    th_universal: f32,
    /// Learned-template matching threshold, in whitened σ (`Th_learned`).
    th_learned: f32,
    /// Single-channel thresholds of the clips the universal templates are learned from, in whitened σ (`Th_single_ch`; EMUsort pools several).
    th_single_ch: Vec<f32>,
    /// Learn `wPCA` / `wTEMP` from the recording; `False`: Kilosort4's predefined `wTEMP.npz`.
    templates_from_data: bool,
    /// Universal templates (`wTEMP` rows) learned.
    n_templates: usize,
    /// Temporal principal components (`wPCA` rows) learned; also the features per channel.
    n_pcs: usize,
    /// Every `nskip`-th batch fits the whitening and learns the templates.
    nskip: usize,
    /// Vertical spacing of the template positions, µm; `None`: the median vertical contact spacing.
    dmin: Option<f32>,
    /// Horizontal spacing of the template positions, µm.
    dminx: f32,
    /// Template positions farther than this from every contact are dropped, µm.
    max_channel_distance: f32,
    /// Width of the smallest spatial template (Gaussian σ), µm; the others are its multiples.
    min_template_size: f32,
    /// Spatial template widths tried: `min_template_size · (1 … template_sizes)`.
    template_sizes: usize,
    /// Channels per template position (and per spike's features).
    nearest_chans: usize,
    /// Neighbouring template positions in the local-maximum test of detection.
    nearest_templates: usize,
    /// Subtract the common average across channels before filtering (`do_CAR`).
    do_car: bool,
    /// High-pass cutoff, Hz.
    highpass_cutoff_hz: f64,
    /// Channels in each local whitening neighbourhood.
    whitening_range: usize,
    /// Samples per batch (60 000 is 2 s at 30 kHz).
    batch_size: usize,
    /// Every `cluster_downsampling`-th spike is a right node of the clustering graph (1: all; larger: faster, coarser).
    cluster_downsampling: usize,
    /// Neighbours of every spike in the clustering graph.
    cluster_neighbors: usize,
    /// At most this many right nodes per probe section, so long recordings keep the same neighbourhood scale.
    max_cluster_subset: usize,
    /// Pin the autotuned choices that change the numbers: the same input on the same device gives the same sort.
    reproducible: bool,
}

/// Kilosort4's defaults in their Python form: the source of every default of `Kilosort4Config(...)`.
fn defaults() -> PyKilosort4Config {
    PyKilosort4Config::from(&Kilosort4Config::default())
}

impl From<&Kilosort4Config> for PyKilosort4Config {
    fn from(c: &Kilosort4Config) -> Self {
        let o = &c.centres;
        Self {
            nt: c.nt,
            nt0min: c.nt0min,
            th_universal: c.th_universal,
            th_learned: c.th_learned,
            th_single_ch: c.th_single_ch.clone(),
            templates_from_data: c.templates_from_data,
            n_templates: c.n_templates,
            n_pcs: c.n_pcs,
            nskip: c.nskip,
            dmin: o.dmin,
            dminx: o.dminx,
            max_channel_distance: o.max_channel_distance,
            min_template_size: o.min_template_size,
            template_sizes: o.template_sizes,
            nearest_chans: o.nearest_chans,
            nearest_templates: o.nearest_templates,
            do_car: c.do_car,
            highpass_cutoff_hz: c.highpass_cutoff_hz,
            whitening_range: c.whitening_range,
            cluster_downsampling: c.clustering.graph.subset_stride,
            cluster_neighbors: c.clustering.graph.neighbours,
            max_cluster_subset: c.clustering.graph.max_subset,
            reproducible: c.reproducible,
            batch_size: c.batch_size,
        }
    }
}

impl PyKilosort4Config {
    fn to_rust(&self) -> Kilosort4Config {
        Kilosort4Config {
            nt: self.nt,
            nt0min: self.nt0min,
            th_universal: self.th_universal,
            th_learned: self.th_learned,
            th_single_ch: self.th_single_ch.clone(),
            templates_from_data: self.templates_from_data,
            n_templates: self.n_templates,
            n_pcs: self.n_pcs,
            nskip: self.nskip,
            centres: CentreOptions {
                dmin: self.dmin,
                dminx: self.dminx,
                max_channel_distance: self.max_channel_distance,
                min_template_size: self.min_template_size,
                template_sizes: self.template_sizes,
                nearest_chans: self.nearest_chans,
                nearest_templates: self.nearest_templates,
            },
            do_car: self.do_car,
            highpass_cutoff_hz: self.highpass_cutoff_hz,
            whitening_range: self.whitening_range,
            batch_size: self.batch_size,
            clustering: {
                let mut c = dsp_synapse_ml::sorters::kilosort4::ClusteringOptions::default();
                c.graph.subset_stride = self.cluster_downsampling;
                c.graph.neighbours = self.cluster_neighbors;
                c.graph.max_subset = self.max_cluster_subset;
                c
            },
            template_merge: Default::default(),
            max_peels: dsp_synapse_ml::sorters::kilosort4::MAX_PEELS,
            reproducible: self.reproducible,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyKilosort4Config {
    /// Kilosort4's defaults, changed by keyword (see the class and attribute docs).
    #[new]
    #[pyo3(signature = (*, nt = defaults().nt, nt0min = defaults().nt0min, th_universal = defaults().th_universal, th_learned = defaults().th_learned, th_single_ch = defaults().th_single_ch, templates_from_data = defaults().templates_from_data, n_templates = defaults().n_templates, n_pcs = defaults().n_pcs, nskip = defaults().nskip, dmin = defaults().dmin, dminx = defaults().dminx, max_channel_distance = defaults().max_channel_distance, min_template_size = defaults().min_template_size, template_sizes = defaults().template_sizes, nearest_chans = defaults().nearest_chans, nearest_templates = defaults().nearest_templates, do_car = defaults().do_car, highpass_cutoff_hz = defaults().highpass_cutoff_hz, whitening_range = defaults().whitening_range, batch_size = defaults().batch_size, cluster_downsampling = defaults().cluster_downsampling, cluster_neighbors = defaults().cluster_neighbors, max_cluster_subset = defaults().max_cluster_subset, reproducible = defaults().reproducible))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nt: usize,
        nt0min: Option<usize>,
        th_universal: f32,
        th_learned: f32,
        th_single_ch: Vec<f32>,
        templates_from_data: bool,
        n_templates: usize,
        n_pcs: usize,
        nskip: usize,
        dmin: Option<f32>,
        dminx: f32,
        max_channel_distance: f32,
        min_template_size: f32,
        template_sizes: usize,
        nearest_chans: usize,
        nearest_templates: usize,
        do_car: bool,
        highpass_cutoff_hz: f64,
        whitening_range: usize,
        batch_size: usize,
        cluster_downsampling: usize,
        cluster_neighbors: usize,
        max_cluster_subset: usize,
        reproducible: bool,
    ) -> Self {
        Self {
            nt,
            nt0min,
            th_universal,
            th_learned,
            th_single_ch,
            templates_from_data,
            n_templates,
            n_pcs,
            nskip,
            dmin,
            dminx,
            max_channel_distance,
            min_template_size,
            template_sizes,
            nearest_chans,
            nearest_templates,
            do_car,
            highpass_cutoff_hz,
            whitening_range,
            batch_size,
            cluster_downsampling,
            cluster_neighbors,
            max_cluster_subset,
            reproducible,
        }
    }

    /// Sample of a waveform its peak is aligned to (`nt0min`, or upstream's rule from `nt`).
    fn resolved_nt0min(&self) -> usize {
        self.to_rust().nt0min()
    }

    fn __repr__(&self) -> String {
        format!("Kilosort4Config(nt={}, th_universal={}, n_templates={}, n_pcs={}, ...)", self.nt, self.th_universal, self.n_templates, self.n_pcs)
    }
}

/// EMUsort settings: its Kilosort4 settings (`kilosort4`, with EMUsort's defaults) and its own.
///
/// EMUsort's Kilosort4 defaults differ from Kilosort4's: 9 universal templates and 9 PCs, clip
/// thresholds `[6, 9, 12, 15]`, `nskip = 2`, no common average reference. Edit them in place
/// (`config.kilosort4.nt = 121`): `kilosort4` is shared, not a copy.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.synapse.ml import emusort
/// >>> config = emusort.Config(hdbscan_min_cluster_size=30)
/// >>> config.kilosort4.nt = 121          # 5 ms at 24.4 kHz: match the MUAP width
#[gen_stub_pyclass]
#[pyclass(name = "EmusortConfig", get_all, set_all, skip_from_py_object)]
pub struct PyEmusortConfig {
    /// Kilosort4 settings of the run (EMUsort's defaults; see the class docs).
    kilosort4: Py<PyKilosort4Config>,
    /// Estimate the delay of every channel against a reference channel (±2 ms) and remove it
    /// before detection (`remove_chan_delays`).
    remove_channel_delays: bool,
    /// Remove HDBSCAN outliers from the clips before k-means learns the universal templates
    /// (`remove_spike_outliers`).
    remove_spike_outliers: bool,
    /// HDBSCAN `min_cluster_size` of the outlier removal.
    hdbscan_min_cluster_size: usize,
}

impl PyEmusortConfig {
    fn to_rust(&self, py: Python<'_>) -> EmusortConfig {
        EmusortConfig {
            kilosort4: self.kilosort4.borrow(py).to_rust(),
            remove_channel_delays: self.remove_channel_delays,
            remove_spike_outliers: self.remove_spike_outliers,
            hdbscan_min_cluster_size: self.hdbscan_min_cluster_size,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmusortConfig {
    /// EMUsort's defaults (paper and upstream), changed by keyword. `kilosort4=None`: EMUsort's
    /// Kilosort4 defaults (start from `EmusortConfig().kilosort4` to change a few).
    #[new]
    #[pyo3(signature = (*, kilosort4 = None, remove_channel_delays = EmusortConfig::default().remove_channel_delays, remove_spike_outliers = EmusortConfig::default().remove_spike_outliers, hdbscan_min_cluster_size = EmusortConfig::default().hdbscan_min_cluster_size))]
    fn new(
        py: Python<'_>,
        kilosort4: Option<Py<PyKilosort4Config>>,
        remove_channel_delays: bool,
        remove_spike_outliers: bool,
        hdbscan_min_cluster_size: usize,
    ) -> PyResult<Self> {
        let kilosort4 = match kilosort4 {
            Some(k) => k,
            None => Py::new(py, PyKilosort4Config::from(&EmusortConfig::default().kilosort4))?,
        };
        Ok(Self { kilosort4, remove_channel_delays, remove_spike_outliers, hdbscan_min_cluster_size })
    }

    /// Largest channel delay searched (±2 ms, EMUsort's `fs / 500`), in samples.
    ///
    /// Parameters
    /// ----------
    /// fs : float
    ///     Sampling rate, Hz.
    fn max_delay_samples(&self, py: Python<'_>, fs: f64) -> usize {
        self.to_rust(py).max_delay_samples(fs)
    }

    fn __repr__(&self) -> String {
        format!("EmusortConfig(remove_channel_delays={}, remove_spike_outliers={}, hdbscan_min_cluster_size={}, kilosort4=...)", self.remove_channel_delays, self.remove_spike_outliers, self.hdbscan_min_cluster_size)
    }
}

/// Kilosort4 settings of a `Kilosort4Config` or an `EmusortConfig`, and the template-learning
/// options of that sorter.
fn sorter_settings(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<(Kilosort4Config, LearnOptions)> {
    if let Ok(c) = config.cast::<PyKilosort4Config>() {
        let c = c.borrow().to_rust();
        let learn = c.learn_options();
        return Ok((c, learn));
    }
    if let Ok(c) = config.cast::<PyEmusortConfig>() {
        let c = c.borrow().to_rust(py);
        return Ok((c.kilosort4.clone(), c.learn_options()));
    }
    Err(PyValueError::new_err("config must be a Kilosort4Config or an EmusortConfig"))
}

// ------------------------------------------------------------------------------------------------
// Universal templates
// ------------------------------------------------------------------------------------------------

/// Kilosort4's universal templates: `wpca` (`[n_pcs, nt]`) and `wtemp` (`[n_templates, nt]`).
#[gen_stub_pyclass]
#[pyclass(name = "UniversalTemplates", skip_from_py_object)]
pub struct PyUniversalTemplates {
    inner: UniversalTemplates,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUniversalTemplates {
    /// Reads universal templates from an `.npz` (Kilosort4's `wTEMP.npz` layout: arrays `wPCA`, `wTEMP`).
    ///
    /// Parameters
    /// ----------
    /// path : str
    #[staticmethod]
    fn from_npz(path: &str) -> PyResult<Self> {
        Ok(Self { inner: UniversalTemplates::from_npz(Path::new(path)).map_err(value_error)? })
    }

    /// Kilosort4's predefined `wTEMP.npz`, downloaded and verified through the hub (feature `hub`).
    #[cfg(feature = "hub")]
    #[staticmethod]
    fn from_hub() -> PyResult<Self> {
        let hub = dsp_synapse_ml::ModelHub::new().map_err(runtime_error)?;
        let (_, path) = hub.pull(dsp_synapse_ml::sorters::kilosort4::KILOSORT4_WTEMP_MODEL_ID, false).map_err(runtime_error)?;
        Ok(Self { inner: UniversalTemplates::from_npz(&path).map_err(value_error)? })
    }

    /// Temporal basis, `[n_pcs, nt]` float32: features are projections onto its rows.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn wpca<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(py, self.inner.wpca.clone(), &[self.inner.n_pcs, self.inner.nt])
    }

    /// Universal templates, `[n_templates, nt]` float32 (unit-norm rows).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn wtemp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(py, self.inner.wtemp.clone(), &[self.inner.n_templates, self.inner.nt])
    }

    fn __repr__(&self) -> String {
        format!("UniversalTemplates(nt={}, n_pcs={}, n_templates={})", self.inner.nt, self.inner.n_pcs, self.inner.n_templates)
    }
}

/// Isolated single-channel peaks of a preprocessed batch, cut into clips: the first step of learning
/// universal templates.
///
/// Parameters
/// ----------
/// batch : numpy.ndarray
///     `[channels, samples]`, preprocessed (whitened), padded by at least `nt` samples on each side.
/// config : Kilosort4Config or EmusortConfig
///     The sorter's settings (`nt`, `nt0min`, the thresholds, …).
///     Peaks above each of `th_single_ch` count (EMUsort pools several thresholds).
///
/// Returns
/// -------
/// numpy.ndarray
///     `[clips, nt]` float32, each clip's peak at sample `nt0min`.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn extract_clips<'py>(py: Python<'py>, batch: Bound<'py, PyAny>, #[gen_stub(override_type(type_repr = "Kilosort4Config | EmusortConfig"))] config: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let options = if let Ok(c) = config.cast::<PyEmusortConfig>() { c.borrow().to_rust(py).clip_options() } else { sorter_settings(py, &config)?.0.clip_options() };
    let input = F32Array::new(&batch)?;
    let (channels, samples) = input.channels_samples(None)?;
    let x = input.slice();
    let mut out = Vec::new();
    let n = py.detach(|| clips_of(x, channels, samples, &options, &mut out));
    to_numpy(py, out, &[n, options.nt])
}

/// Learns universal templates from clips, on the device: `wPCA` from their SVD, `wTEMP` from k-means
/// (EMUsort: HDBSCAN outliers removed first).
///
/// Parameters
/// ----------
/// clips : numpy.ndarray
///     `[clips, nt]` (e.g. from `extract_clips` over several batches).
/// config : Kilosort4Config or EmusortConfig
///     The sorter's settings (`nt`, `nt0min`, the thresholds, …).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// UniversalTemplates
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (clips, config, *, runtime=None))]
fn learn_universal_templates(py: Python<'_>, clips: Bound<'_, PyAny>, #[gen_stub(override_type(type_repr = "Kilosort4Config | EmusortConfig"))] config: Bound<'_, PyAny>, runtime: Option<&str>) -> PyResult<PyUniversalTemplates> {
    struct Task<'a>(&'a [f32], usize, LearnOptions);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<UniversalTemplates>;
        fn run(self, client: Client) -> Self::Output {
            learn(&client, self.0, self.1, &self.2)
        }
    }
    let (ks, options) = sorter_settings(py, &config)?;
    let input = F32Array::new(&clips)?;
    let [_, nt] = *input.shape() else { return Err(PyValueError::new_err("clips must be [clips, nt]")) };
    if nt != ks.nt {
        return Err(PyValueError::new_err(format!("clips of {nt} samples, config nt = {}", ks.nt)));
    }
    let (x, target) = (input.slice(), target(runtime)?);
    let inner = py.detach(|| target.run(Task(x, nt, options))).map_err(runtime_error)?.map_err(value_error)?;
    Ok(PyUniversalTemplates { inner })
}

// ------------------------------------------------------------------------------------------------
// Universal-template detection
// ------------------------------------------------------------------------------------------------

/// Kilosort4's virtual template positions on a probe, with their spatial weights.
#[gen_stub_pyclass]
#[pyclass(name = "TemplateCentres", skip_from_py_object)]
pub struct PyTemplateCentres {
    inner: TemplateCentres,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateCentres {
    /// Template positions of a probe.
    ///
    /// Parameters
    /// ----------
    /// probe : ProbeLayout
    /// config : Kilosort4Config or EmusortConfig
    ///     Gives the spacing and sizes: `dmin`, `dminx`, `min_template_size`, `template_sizes`,
    ///     `nearest_chans`, `nearest_templates`.
    #[new]
    fn new(py: Python<'_>, probe: PyRef<'_, PyProbeLayout>, #[gen_stub(override_type(type_repr = "Kilosort4Config | EmusortConfig"))] config: Bound<'_, PyAny>) -> PyResult<Self> {
        let (ks, _) = sorter_settings(py, &config)?;
        Ok(Self { inner: TemplateCentres::new(&probe.inner, &ks.centres).map_err(value_error)? })
    }

    /// Number of template positions.
    #[getter]
    fn count(&self) -> usize {
        self.inner.n_centres()
    }

    /// `[centres, 2]` positions (x, y µm): Kilosort4's `xcup`, `ycup`.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let flat: Vec<f32> = self.inner.positions.iter().flatten().copied().collect();
        to_numpy(py, flat, &[self.inner.n_centres(), 2])
    }

    /// `[nearest_chans, centres]` nearest channels of each centre: Kilosort4's `iC`.
    #[getter]
    fn channels(&self) -> Vec<Vec<u32>> {
        self.inner.ic.chunks_exact(self.inner.n_centres().max(1)).map(<[u32]>::to_vec).collect()
    }
}

/// Universal-template detection on one preprocessed batch (the stage `run` applies to every batch).
///
/// Parameters
/// ----------
/// batch : numpy.ndarray
///     `[channels, samples]`, preprocessed and whitened.
/// centres : TemplateCentres
/// templates : UniversalTemplates
/// config : Kilosort4Config or EmusortConfig
///     The sorter's settings (`nt`, `nt0min`, the thresholds, …).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// list of dict
///     One per spike: `sample` (trough, batch sample), `centre`, `amplitude` (whitened σ), `template`,
///     `size`, `x_um`, `y_um`, `features` (`[nearest_chans, n_pcs]`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (batch, centres, templates, config, *, runtime=None))]
fn detect_universal<'py>(py: Python<'py>, batch: Bound<'py, PyAny>, centres: PyRef<'py, PyTemplateCentres>, templates: PyRef<'py, PyUniversalTemplates>, config: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyList>> {
    struct Task<'a> {
        x: &'a [f32],
        channels: usize,
        samples: usize,
        centres: &'a TemplateCentres,
        templates: &'a UniversalTemplates,
        th: f32,
        nt0min: usize,
    }
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<Vec<UniversalSpike>>;
        fn run(self, client: Client) -> Self::Output {
            let handle = client.create_from_slice(f32::as_bytes(self.x));
            detect(&client, &handle, self.channels, self.samples, self.centres, self.templates, self.th, self.nt0min)
        }
    }
    let (ks, _) = sorter_settings(py, &config)?;
    let input = F32Array::new(&batch)?;
    let (channels, samples) = input.channels_samples(None)?;
    let target = target(runtime)?;
    let task = Task { x: input.slice(), channels, samples, centres: &centres.inner, templates: &templates.inner, th: ks.th_universal, nt0min: ks.nt0min() };
    let spikes = py.detach(|| target.run(task)).map_err(runtime_error)?.map_err(value_error)?;
    let per_spike = ks.centres.nearest_chans * templates.inner.n_pcs;
    let out = PyList::empty(py);
    for s in spikes {
        let d = PyDict::new(py);
        d.set_item("sample", s.sample)?;
        d.set_item("centre", s.centre)?;
        d.set_item("amplitude", s.amplitude)?;
        d.set_item("template", s.template)?;
        d.set_item("size", s.size)?;
        d.set_item("x_um", s.x_um)?;
        d.set_item("y_um", s.y_um)?;
        d.set_item("features", to_numpy(py, s.features, &[per_spike / templates.inner.n_pcs.max(1), templates.inner.n_pcs])?)?;
        out.append(d)?;
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------
// EMUsort Runner
// ------------------------------------------------------------------------------------------------

/// EMUsort over a whole recording, without a progress bar. Prefer `dsp_kitchen.synapse.ml.emusort.run`,
/// which documents every argument and shows progress.
#[gen_stub_pyfunction]
#[pyfunction(name = "run_emusort")]
#[pyo3(signature = (recording, probe, config, *, preprocessing_from=None, progress=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn run_emusort_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    #[gen_stub(override_type(type_repr = "EmusortConfig"))]
    config: Bound<'_, PyAny>,
    preprocessing_from: Option<PyRef<'_, PyKilosort4Result>>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PyKilosort4Result> {
    let Ok(config) = config.cast::<PyEmusortConfig>() else {
        return Err(PyValueError::new_err("config must be an EmusortConfig"));
    };
    let plan = RunPlan::emusort(&config.borrow().to_rust(py), recording.inner.info().sample_rate_hz());
    run_plan_py(py, &recording, &probe, plan, preprocessing_from, progress, runtime)
}

/// A Python callable receiving `(stage, step, steps, done, total, unit)` for each progress report
/// of a run (called with the GIL re-taken; its errors are printed, not raised, so a bar cannot
/// stop a run).
struct PyProgress(Py<PyAny>);

impl dsp_core::ProgressSink for PyProgress {
    fn report(&self, e: &dsp_core::ProgressEvent<'_>) {
        Python::attach(|py| {
            if let Err(err) = self.0.call1(py, (e.stage, e.step, e.steps, e.done, e.total, e.unit)) {
                err.print(py);
            }
        });
    }
}

/// Runs `plan` over `recording` on `runtime` (shared by `kilosort4.run` and `emusort.run`),
/// reporting progress to the callable `progress` when given.
fn run_plan_py(
    py: Python<'_>,
    recording: &crate::buffer::PyRecording,
    probe: &PyProbeLayout,
    mut plan: RunPlan,
    preprocessing_from: Option<PyRef<'_, PyKilosort4Result>>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PyKilosort4Result> {
    struct Task<'a>(&'a dyn dsp_core::RecordingSource, &'a dsp_io::neuro::probe::SensorLayout, &'a RunPlan, &'a dyn dsp_core::ProgressSink);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<Kilosort4Result>;
        fn run(self, client: Client) -> Self::Output {
            run_plan(&client, self.0, self.1, self.2, self.3)
        }
    }
    let sink: Box<dyn dsp_core::ProgressSink> = match progress {
        Some(callable) => Box::new(PyProgress(callable)),
        None => Box::new(dsp_core::NoProgress),
    };
    plan.fitted = preprocessing_from.map(|r| r.inner.fitted.clone());
    let source = recording.inner.clone();
    let (layout, target) = (probe.inner.clone(), target(runtime)?);
    let inner = py.detach(|| target.run(Task(source.as_ref(), &layout, &plan, sink.as_ref()))).map_err(runtime_error)?.map_err(value_error)?;
    Ok(PyKilosort4Result { inner })
}

/// Result of a Kilosort4 or EMUsort run (`kilosort4.run`, `emusort.run`).
///
/// Holds the fitted preprocessing, the universal and learned templates, the spikes (learned-template
/// matches, with their unit) and the units. Spike times are recording samples at the waveform's
/// trough; for EMUsort they are in the reference channel's frame (see `channel_delays`). Amplitudes
/// are in whitened σ.
///
/// Examples
/// --------
/// >>> result = kilosort4.run(recording, probe, kilosort4.Config())
/// >>> result.n_units
/// >>> spikes = result.spikes()                  # dict of arrays, one entry per spike
/// >>> sorting = result.to_sorting_output(probe)  # units for export (Phy, zarr) and inspection
#[gen_stub_pyclass]
#[pyclass(name = "Kilosort4Result", skip_from_py_object)]
pub struct PyKilosort4Result {
    inner: Kilosort4Result,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyKilosort4Result {
    /// `"kilosort4"` or `"emusort"`.
    #[getter]
    fn sorter(&self) -> &'static str {
        self.inner.sorter
    }
    /// `(delays, reference_channel)`, or `None` without delay removal.
    #[getter]
    fn channel_delays(&self) -> Option<(Vec<isize>, usize)> {
        self.inner.fitted.channel_delays.as_ref().map(|d| (d.delays.clone(), d.reference))
    }
    /// Sampling rate of the recording, Hz.
    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.inner.sample_rate_hz
    }
    /// Units found by clustering the detected spikes.
    #[getter]
    fn n_units(&self) -> usize {
        self.inner.clusters.n_units
    }
    /// Learned templates: the units' templates aligned, near-duplicates merged.
    #[getter]
    fn n_learned_templates(&self) -> usize {
        self.inner.learned.n
    }
    /// Whether the run pinned its result-changing tuned choices (`config.reproducible`).
    #[getter]
    fn reproducible(&self) -> bool {
        self.inner.reproducible
    }
    /// The device the run used: reproducible runs give the same spikes on the same device.
    #[getter]
    fn device(&self) -> String {
        self.inner.device.clone()
    }
    /// Samples in the recording.
    #[getter]
    fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }
    /// Whitening matrix, `[channels, channels]` float32 (applied after filtering).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn whitening<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let whitening = &self.inner.fitted.whitening;
        to_numpy(py, whitening.matrix.clone(), &[whitening.num_channels, whitening.num_channels])
    }
    /// The universal templates the run detected with (learned or predefined).
    #[getter]
    fn templates(&self) -> PyUniversalTemplates {
        PyUniversalTemplates { inner: self.inner.templates.clone() }
    }
    /// Samples read before and after each batch (filter settling, waveform window, channel delays).
    #[getter]
    fn halos(&self) -> (u64, u64) {
        self.inner.fitted.halos
    }
    /// Batches the recording was processed in.
    #[getter]
    fn windows(&self) -> usize {
        self.inner.fitted.schedule.len()
    }
    /// The fitted preprocessing (filters, whitening) as a reusable `Pipeline`: run it on any
    /// stretch of the recording to see the signal the sorter saw.
    #[getter]
    fn preprocessing(&self) -> PyPipeline {
        PyPipeline { stages: self.inner.fitted.pipeline.stages().to_vec() }
    }
    /// The spikes, one entry per spike in every array.
    ///
    /// Returns
    /// -------
    /// dict
    ///     `sample` (recording sample of the trough; EMUsort: reference channel's frame), `unit`
    ///     (cluster), `template` (learned template matched), `amplitude` (whitened σ), `x_um`,
    ///     `y_um` (position, µm), `centre` (detection position index), `size` (spatial size index).
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.inner.spikes;
        let d = PyDict::new(py);
        d.set_item("sample", s.iter().map(|x| x.sample as u64).collect::<Vec<_>>())?;
        d.set_item("centre", s.iter().map(|x| x.centre).collect::<Vec<_>>())?;
        d.set_item("amplitude", to_numpy(py, s.iter().map(|x| x.amplitude).collect(), &[s.len()])?)?;
        d.set_item("template", s.iter().map(|x| x.template).collect::<Vec<_>>())?;
        d.set_item("size", s.iter().map(|x| x.size).collect::<Vec<_>>())?;
        d.set_item("x_um", to_numpy(py, s.iter().map(|x| x.x_um).collect(), &[s.len()])?)?;
        d.set_item("y_um", to_numpy(py, s.iter().map(|x| x.y_um).collect(), &[s.len()])?)?;
        d.set_item("unit", self.inner.clusters.labels.clone())?;
        Ok(d)
    }
    /// The units in `SortingOutput` form: one unit per cluster, with spike times, amplitudes,
    /// positions and a waveform template; for export (`save_sorting`: Phy, zarr) and inspection.
    ///
    /// Parameters
    /// ----------
    /// probe : ProbeLayout, optional
    ///     Geometry stored with the units (channel positions in exports).
    #[pyo3(signature = (probe=None))]
    fn to_sorting_output(&self, probe: Option<PyRef<'_, PyProbeLayout>>) -> PySortingOutput {
        PySortingOutput::new(self.inner.to_sorting_output(probe.map(|p| p.inner.clone())))
    }
    fn __repr__(&self) -> String {
        format!(
            "Kilosort4Result(sorter={}, spikes={}, windows={}, halos={:?}, delays={})",
            self.inner.sorter,
            self.inner.spikes.len(),
            self.inner.fitted.schedule.len(),
            self.inner.fitted.halos,
            self.inner.fitted.channel_delays.is_some()
        )
    }
}

/// Fits a sorter's preprocessing on a recording (high-pass, common average reference, local
/// whitening) and returns it as a reusable `Pipeline`.
///
/// Parameters
/// ----------
/// recording : Recording
/// probe : ProbeLayout
/// config : Kilosort4Config or EmusortConfig
///     The sorter's settings (`nt`, `nt0min`, the thresholds, …).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// Pipeline
#[gen_stub_pyfunction]
#[pyfunction(name = "create_preprocessing")]
#[pyo3(signature = (recording, probe, config, *, runtime=None))]
fn create_kilosort4_preprocessing_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    config: Bound<'_, PyAny>,
    runtime: Option<&str>,
) -> PyResult<PyPipeline> {
    struct Task<'a>(&'a dyn dsp_core::RecordingSource, &'a dsp_io::neuro::probe::SensorLayout, &'a Kilosort4Config);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<dsp_base::pipeline::Pipeline>;
        fn run(self, client: Client) -> Self::Output {
            fit_kilosort4_preprocessing(&client, self.0, self.1, self.2).map(|(pipe, _)| pipe)
        }
    }
    let source = recording.inner.clone();
    let ks_config = sorter_settings(py, &config)?.0;
    let (layout, target) = (probe.inner.clone(), target(runtime)?);
    let inner = py.detach(|| target.run(Task(source.as_ref(), &layout, &ks_config)))
        .map_err(runtime_error)?
        .map_err(value_error)?;
    Ok(PyPipeline { stages: inner.stages().to_vec() })
}

/// Kilosort4 over a whole recording, without a progress bar. Prefer
/// `dsp_kitchen.synapse.ml.kilosort4.run`, which documents every argument and shows progress.
#[gen_stub_pyfunction]
#[pyfunction(name = "run")]
#[pyo3(signature = (recording, probe, config, *, templates=None, preprocessing_from=None, progress=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn run_kilosort4_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    #[gen_stub(override_type(type_repr = "Kilosort4Config | EmusortConfig"))]
    config: Bound<'_, PyAny>,
    templates: Option<PyRef<'_, PyUniversalTemplates>>,
    preprocessing_from: Option<PyRef<'_, PyKilosort4Result>>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PyKilosort4Result> {
    let mut plan = RunPlan::kilosort4(&sorter_settings(py, &config)?.0);
    plan.templates = templates.map(|t| t.inner.clone());
    run_plan_py(py, &recording, &probe, plan, preprocessing_from, progress, runtime)
}

// ------------------------------------------------------------------------------------------------
// EMUsort channel delays
// ------------------------------------------------------------------------------------------------

/// EMUsort's channel delays from preprocessed batches, on the device: the rectified, normalised
/// channels are cross-correlated within ±`max_lag`, the reference channel is the one best correlated
/// with all others, and each channel's delay is its best lag against it. Every batch is uploaded once
/// and the correlations read back once.
///
/// Parameters
/// ----------
/// batches : list of numpy.ndarray
///     Each `[channels, samples]`, padded by `pad` samples on each side.
/// pad : int
///     Padding of each batch, samples (the correlation uses `[pad, samples − pad)`).
/// max_lag : int
///     Largest delay searched, samples (EMUsort: 2 ms).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// tuple
///     `(delays, reference_channel)`: `delays` (list of int, samples, one per channel).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (batches, *, pad, max_lag, runtime=None))]
fn estimate_channel_delays(py: Python<'_>, batches: Vec<Bound<'_, PyAny>>, pad: usize, max_lag: usize, runtime: Option<&str>) -> PyResult<(Vec<isize>, usize)> {
    struct Task<'a>(Vec<(&'a [f32], usize)>, usize, usize, usize);
    impl ComputeTask for Task<'_> {
        type Output = (Vec<isize>, usize);
        fn run(self, client: Client) -> Self::Output {
            let Task(batches, channels, pad, max_lag) = self;
            let mut est = ChannelDelayEstimator::new(&client, channels, max_lag);
            for (x, samples) in batches {
                est.add(&dsp_base::core::buffer::upload(&client, x), samples, pad..samples - pad);
            }
            est.delays()
        }
    }
    let arrays = batches.iter().map(F32Array::new).collect::<PyResult<Vec<_>>>()?;
    let mut channels = None;
    let mut shapes = Vec::with_capacity(arrays.len());
    for a in &arrays {
        let (c, samples) = a.channels_samples(None)?;
        if *channels.get_or_insert(c) != c {
            return Err(PyValueError::new_err(format!("batches have {} and {c} channels", channels.unwrap_or(c))));
        }
        if 2 * pad >= samples {
            return Err(PyValueError::new_err(format!("padding {pad} leaves no interior in {samples} samples")));
        }
        shapes.push(samples);
    }
    let Some(channels) = channels else {
        return Err(PyValueError::new_err("no batches"));
    };
    let target = target(runtime)?;
    let views: Vec<(&[f32], usize)> = arrays.iter().map(F32Array::slice).zip(shapes).collect();
    py.detach(|| target.run(Task(views, channels, pad, max_lag))).map_err(runtime_error)
}

/// Removes channel delays from a batch, on the device: `x[i, t] ← x[i, (t + delay_i) mod samples]`
/// (a circular shift: only the padding wraps around).
///
/// Parameters
/// ----------
/// batch : numpy.ndarray
///     `[channels, samples]`, converted to float32.
/// delays : list of int
///     Delay of each channel, samples (from `estimate_channel_delays`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     `[channels, samples]` float32.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (batch, delays, *, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn apply_channel_delays<'py>(py: Python<'py>, batch: Bound<'py, PyAny>, delays: Vec<isize>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    struct Task<'a>(&'a [f32], Vec<isize>, usize);
    impl ComputeTask for Task<'_> {
        type Output = Vec<f32>;
        fn run(self, client: Client) -> Self::Output {
            let Task(x, delays, samples) = self;
            let total = x.len();
            let mut aligner = ChannelAligner::new(&client, delays, samples);
            let out = aligner.align(&dsp_base::core::buffer::upload(&client, x), samples);
            dsp_base::core::buffer::download::<f32>(&client, out)[..total].to_vec()
        }
    }
    let input = F32Array::new(&batch)?;
    let (channels, samples) = input.channels_samples(None)?;
    if delays.len() != channels {
        return Err(PyValueError::new_err(format!("{} delays for {channels} channels", delays.len())));
    }
    let (x, target) = (input.slice(), target(runtime)?);
    let out = py.detach(|| target.run(Task(x, delays, samples))).map_err(runtime_error)?;
    to_numpy(py, out, input.shape())
}

// ------------------------------------------------------------------------------------------------
// Provenance
// ------------------------------------------------------------------------------------------------

/// Where a sorter or model comes from: paper (with DOI), code, license, downloaded artifacts.
#[gen_stub_pyclass]
#[pyclass(name = "Provenance", skip_from_py_object)]
pub struct PyProvenance {
    inner: Provenance,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProvenance {
    /// One-line citation: authors, year, title, venue, DOI link, code and license.
    fn citation(&self) -> String {
        self.inner.citation()
    }

    /// Every field, as nested dicts.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.inner;
        let d = PyDict::new(py);
        d.set_item("name", &p.name)?;
        d.set_item("kind", format!("{:?}", p.kind))?;
        if let Some(paper) = &p.paper {
            let pd = PyDict::new(py);
            pd.set_item("title", &paper.title)?;
            pd.set_item("authors", &paper.authors)?;
            pd.set_item("venue", &paper.venue)?;
            pd.set_item("year", paper.year)?;
            pd.set_item("doi", &paper.doi)?;
            pd.set_item("doi_url", paper.doi_url())?;
            d.set_item("paper", pd)?;
        }
        let code = PyDict::new(py);
        code.set_item("url", &p.code.url)?;
        code.set_item("license", &p.code.license)?;
        code.set_item("version", &p.code.version)?;
        d.set_item("code", code)?;
        let artifacts = PyList::empty(py);
        for a in &p.artifacts {
            let ad = PyDict::new(py);
            ad.set_item("name", &a.name)?;
            ad.set_item("url", &a.url)?;
            ad.set_item("sha256", &a.sha256)?;
            ad.set_item("size_bytes", a.size_bytes)?;
            artifacts.append(ad)?;
        }
        d.set_item("artifacts", artifacts)?;
        d.set_item("notes", &p.notes)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("Provenance({})", self.inner.citation())
    }
}

/// Provenance of the Kilosort4 reimplementation.
#[gen_stub_pyfunction]
#[pyfunction]
fn kilosort4_provenance() -> PyProvenance {
    PyProvenance { inner: kilosort4_record() }
}

/// Provenance of the EMUsort reimplementation.
#[gen_stub_pyfunction]
#[pyfunction]
fn emusort_provenance() -> PyProvenance {
    PyProvenance { inner: emusort_record() }
}

// ------------------------------------------------------------------------------------------------
// Hub
// ------------------------------------------------------------------------------------------------

/// The catalog of published artifacts (sorter arrays, model weights), downloaded and verified
/// into a local cache.
#[cfg(feature = "hub")]
#[gen_stub_pyclass]
#[pyclass(name = "ModelHub", skip_from_py_object)]
pub struct PyModelHub {
    inner: dsp_synapse_ml::ModelHub,
}

#[cfg(feature = "hub")]
fn entry_dict<'py>(py: Python<'py>, e: &dsp_synapse_ml::ModelHubEntry) -> PyResult<Bound<'py, PyDict>> {
    let m = &e.manifest;
    let d = PyDict::new(py);
    d.set_item("id", &m.id)?;
    d.set_item("name", &m.name)?;
    d.set_item("family", &m.family)?;
    d.set_item("task", &m.task)?;
    d.set_item("format", m.format.to_string())?;
    d.set_item("status", e.status.to_string())?;
    d.set_item("url", &e.resolved_download_url)?;
    d.set_item("sha256", &m.sha256)?;
    d.set_item("size_bytes", m.size_bytes)?;
    d.set_item("local_path", e.local_path.to_string_lossy().into_owned())?;
    d.set_item("citation", m.provenance.citation())?;
    Ok(d)
}

#[cfg(feature = "hub")]
#[gen_stub_pymethods]
#[pymethods]
impl PyModelHub {
    /// The catalog with its local cache (`cache_dir`): downloads land there and are verified.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { inner: dsp_synapse_ml::ModelHub::new().map_err(runtime_error)? })
    }

    /// Every catalog entry with its local status.
    fn list<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner.list().iter().map(|e| entry_dict(py, e)).collect()
    }

    /// Catalog entry `id` with its local status (as one item of `list`).
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Catalog id (see `list()`), e.g. `"kilosort4/wtemp-v1"`.
    fn info<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyDict>> {
        entry_dict(py, &self.inner.info(id).map_err(value_error)?)
    }

    /// Provenance of entry `id`: paper, code and license, file source.
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Catalog id (see `list()`), e.g. `"kilosort4/wtemp-v1"`.
    fn provenance(&self, id: &str) -> PyResult<PyProvenance> {
        Ok(PyProvenance { inner: self.inner.info(id).map_err(value_error)?.manifest.provenance })
    }

    /// Downloads entry `id` (size and SHA-256 verified) unless it is installed.
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Catalog id (see `list()`), e.g. `"kilosort4/wtemp-v1"`.
    /// force : bool, default False
    ///     Download again even when installed.
    ///
    /// Returns
    /// -------
    /// str
    ///     Path of the local file.
    #[pyo3(signature = (id, *, force=false))]
    fn pull(&self, py: Python<'_>, id: &str, force: bool) -> PyResult<String> {
        let hub = &self.inner;
        let (_, path) = py.detach(|| hub.pull(id, force)).map_err(runtime_error)?;
        Ok(path.to_string_lossy().into_owned())
    }

    /// Re-hashes entry `id`'s local file.
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Catalog id (see `list()`), e.g. `"kilosort4/wtemp-v1"`.
    ///
    /// Returns
    /// -------
    /// str
    ///     Its status: `"[INSTALLED]"`, `"[AVAILABLE]"` (not downloaded) or `"[CORRUPTED]"` (a file whose
    ///     size or hash does not match).
    fn verify(&self, py: Python<'_>, id: &str) -> PyResult<String> {
        let hub = &self.inner;
        Ok(py.detach(|| hub.verify(id, false)).map_err(runtime_error)?.status.to_string())
    }

    /// Deletes entry `id`'s local file.
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Catalog id (see `list()`), e.g. `"kilosort4/wtemp-v1"`.
    ///
    /// Returns
    /// -------
    /// bool
    ///     Whether there was one.
    fn remove(&self, id: &str) -> PyResult<bool> {
        self.inner.remove(id).map_err(runtime_error)
    }

    /// Deletes every downloaded file; bytes freed.
    fn clean(&self) -> PyResult<u64> {
        self.inner.clean().map_err(runtime_error)
    }

    #[getter]
    /// Folder downloaded artifacts are stored in.
    fn cache_dir(&self) -> String {
        self.inner.hub().cache().root_dir().to_string_lossy().into_owned()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKilosort4Config>()?;
    m.add_class::<PyEmusortConfig>()?;
    m.add_class::<PyUniversalTemplates>()?;
    m.add_class::<PyTemplateCentres>()?;
    m.add_class::<PyProvenance>()?;
    m.add_class::<PyKilosort4Result>()?;
    m.add_function(wrap_pyfunction!(run_emusort_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_kilosort4_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_kilosort4_preprocessing_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_clips, m)?)?;
    m.add_function(wrap_pyfunction!(learn_universal_templates, m)?)?;
    m.add_function(wrap_pyfunction!(detect_universal, m)?)?;
    m.add_function(wrap_pyfunction!(apply_channel_delays, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_channel_delays, m)?)?;
    m.add_function(wrap_pyfunction!(kilosort4_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(emusort_provenance, m)?)?;
    #[cfg(feature = "hub")]
    m.add_class::<PyModelHub>()?;
    Ok(())
}
