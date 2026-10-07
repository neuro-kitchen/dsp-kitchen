//! Clustering, template matching and HD-EMG decomposition. Defaults come from dsp-synapse
//! (`GmmClusterer::default()`, k-means and matching-pursuit `DEFAULT_*`, which follow
//! scikit-learn where it applies); parameters without a library default are required.

use cubecl::prelude::Client;
use dsp_core::compute::ComputeTask;
use dsp_synapse::core::WaveformTemplate;
use dsp_synapse::sorting::kmeans::{DEFAULT_MAX_ITER, DEFAULT_N_INIT, DEFAULT_TOL};
use dsp_synapse::sorting::matching_pursuit::{DEFAULT_MAX_AMPLITUDE_SCALE, DEFAULT_MAX_PASSES, DEFAULT_MIN_AMPLITUDE_SCALE, DEFAULT_MIN_EXPLAINED_ENERGY_UV2};
use dsp_synapse::sorting::{
    cluster_density_peaks as density_peaks, cluster_kde_merge as kde_merge, hdbscan as hdbscan_labels, kmeans as kmeans_fit, match_spikes_matching_pursuit,
    ConvolutiveBssDecomposer, GmmClusterer, GmmCovarianceKind, GmmResult, KMeansOptions, MotorUnitPulseTrain,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{runtime_error, to_numpy, F32Array};
use crate::runtime::target;

/// Template standard deviation when only the mean is given (matching does not use it).
const STD_NOT_GIVEN: f32 = f32::NAN;

/// `(values, spikes, features)` of a `[spikes, features]` matrix.
fn matrix<'a>(a: &'a F32Array<'_>, what: &str) -> PyResult<(&'a [f32], usize, usize)> {
    match *a.shape() {
        [n, d] => Ok((a.slice(), n, d)),
        _ => Err(PyValueError::new_err(format!("{what} must be a 2-D [spikes, features] array"))),
    }
}

fn parse_covariance(kind: &str) -> PyResult<GmmCovarianceKind> {
    match kind {
        "full" => Ok(GmmCovarianceKind::Full),
        "diagonal" => Ok(GmmCovarianceKind::Diagonal),
        "masked" => Ok(GmmCovarianceKind::Masked),
        other => Err(PyValueError::new_err(format!("covariance must be 'full', 'diagonal' or 'masked', got '{other}'"))),
    }
}

struct GmmTask<'a> {
    clusterer: GmmClusterer,
    x: &'a [f32],
    n: usize,
    d: usize,
    masks: Option<&'a [f32]>,
}

impl ComputeTask for GmmTask<'_> {
    type Output = GmmResult;
    fn run(self, client: Client) -> GmmResult {
        self.clusterer.fit(&client, self.x, self.n, self.d, self.masks)
    }
}

/// Gaussian mixture of `features` (`[spikes, features]`) by EM on the device, the number of
/// components chosen by BIC in `min_clusters..=max_clusters`. `masks` (same shape, in `[0, 1]`)
/// selects masked EM (KlustaKwik). Unset options take `GmmClusterer::default()`.
#[pyfunction]
#[pyo3(signature = (features, *, min_clusters=None, max_clusters=None, covariance=None, max_iterations=None, tolerance=None, regularization=None, masks=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn cluster_gmm<'py>(
    py: Python<'py>,
    features: Bound<'py, PyAny>,
    min_clusters: Option<usize>,
    max_clusters: Option<usize>,
    covariance: Option<&str>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    regularization: Option<f32>,
    masks: Option<Bound<'py, PyAny>>,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = GmmClusterer::default();
    let kind = match (&masks, covariance) {
        (Some(_), _) => GmmCovarianceKind::Masked,
        (None, Some(c)) => parse_covariance(c)?,
        (None, None) => defaults.covariance_kind,
    };
    let mut clusterer = GmmClusterer::new(min_clusters.unwrap_or(defaults.k_min), max_clusters.unwrap_or(defaults.k_max), kind);
    clusterer.max_iterations = max_iterations.unwrap_or(defaults.max_iterations);
    clusterer.tolerance = tolerance.unwrap_or(defaults.tolerance);
    clusterer.regularization = regularization.unwrap_or(defaults.regularization);

    let feats = F32Array::new(&features)?;
    let (x, n, d) = matrix(&feats, "features")?;
    let mask_array = masks.as_ref().map(F32Array::new).transpose()?;
    if let Some(m) = &mask_array
        && m.shape() != feats.shape()
    {
        return Err(PyValueError::new_err("masks must have the shape of features"));
    }
    let target = target(runtime)?;
    let task = GmmTask { clusterer, x, n, d, masks: mask_array.as_ref().map(F32Array::slice) };
    let res = py.detach(|| target.run(task)).map_err(runtime_error)?;
    let k = res.num_clusters;
    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("num_clusters", k)?;
    dict.set_item("weights", to_numpy(py, res.weights, &[k])?)?;
    dict.set_item("means", to_numpy(py, res.means, &[k, d])?)?;
    dict.set_item("responsibilities", to_numpy(py, res.responsibilities, &[n, k])?)?;
    dict.set_item("log_likelihood", res.log_likelihood)?;
    dict.set_item("bic", res.bic)?;
    Ok(dict)
}

/// k-means with k-means++ seeding (scikit-learn `KMeans` defaults: `n_init`, `max_iter`, `tol`), on
/// the device (`features` uploaded once).
#[pyfunction]
#[pyo3(signature = (features, k, *, n_init=DEFAULT_N_INIT, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, seed=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn kmeans<'py>(py: Python<'py>, features: Bound<'py, PyAny>, k: usize, n_init: usize, max_iter: usize, tol: f64, seed: Option<u64>, runtime: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
    struct Task<'a>(&'a [f32], usize, usize, usize, KMeansOptions);
    impl ComputeTask for Task<'_> {
        type Output = dsp_synapse::sorting::KMeansResult;
        fn run(self, client: Client) -> Self::Output {
            kmeans_fit(&client, self.0, self.1, self.2, self.3, &self.4)
        }
    }
    let feats = F32Array::new(&features)?;
    let (x, n, d) = matrix(&feats, "features")?;
    if k == 0 || k > n {
        return Err(PyValueError::new_err(format!("k = {k} must be in 1..={n}")));
    }
    let options = KMeansOptions { n_init, max_iter, tol, seed: seed.unwrap_or(KMeansOptions::default().seed) };
    let target = target(runtime)?;
    let res = py.detach(|| target.run(Task(x, n, d, k, options))).map_err(runtime_error)?;
    let centers = res.centers.len() / d.max(1);
    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("centers", to_numpy(py, res.centers, &[centers, d])?)?;
    dict.set_item("inertia", res.inertia)?;
    Ok(dict)
}

/// HDBSCAN labels of `features` (`-1` = noise), excess-of-mass selection (scikit-learn semantics),
/// on the device (`features` uploaded once).
#[pyfunction]
#[pyo3(signature = (features, min_cluster_size, *, runtime=None))]
pub fn hdbscan(py: Python<'_>, features: Bound<'_, PyAny>, min_cluster_size: usize, runtime: Option<&str>) -> PyResult<Vec<i32>> {
    struct Task<'a>(&'a [f32], usize, usize, usize);
    impl ComputeTask for Task<'_> {
        type Output = Vec<i32>;
        fn run(self, client: Client) -> Self::Output {
            hdbscan_labels(&client, self.0, self.1, self.2, self.3)
        }
    }
    let feats = F32Array::new(&features)?;
    let (x, n, d) = matrix(&feats, "features")?;
    let target = target(runtime)?;
    py.detach(|| target.run(Task(x, n, d, min_cluster_size))).map_err(runtime_error)
}

/// Density-peaks clustering (Rodriguez & Laio 2014) with `cutoff_distance` and `num_clusters`.
#[pyfunction]
#[pyo3(signature = (features, *, cutoff_distance, num_clusters))]
pub fn cluster_density_peaks<'py>(py: Python<'py>, features: Bound<'py, PyAny>, cutoff_distance: f32, num_clusters: usize) -> PyResult<Bound<'py, PyDict>> {
    let feats = F32Array::new(&features)?;
    let (x, n, d) = matrix(&feats, "features")?;
    let res = py.detach(|| density_peaks(x, n, d, cutoff_distance, num_clusters));
    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("cluster_centers", res.cluster_centers)?;
    dict.set_item("densities", to_numpy(py, res.densities, &[n])?)?;
    dict.set_item("deltas", to_numpy(py, res.deltas, &[n])?)?;
    Ok(dict)
}

/// Over-clusters `features` into `initial_k` groups, then merges neighbours whose density valley
/// is shallower than `dip_threshold` (a heuristic in the spirit of IsoSplit; no significance test).
#[pyfunction]
#[pyo3(signature = (features, *, initial_k, dip_threshold, min_cluster_size))]
pub fn cluster_kde_merge<'py>(py: Python<'py>, features: Bound<'py, PyAny>, initial_k: usize, dip_threshold: f32, min_cluster_size: usize) -> PyResult<Bound<'py, PyDict>> {
    let feats = F32Array::new(&features)?;
    let (x, n, d) = matrix(&feats, "features")?;
    let res = py.detach(|| kde_merge(x, n, d, initial_k, dip_threshold, min_cluster_size));
    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("num_clusters", res.num_clusters)?;
    dict.set_item("centroids", to_numpy(py, res.centroids, &[res.num_clusters, d])?)?;
    Ok(dict)
}

/// A template from a dict (`mean` `[channels, samples]`, optional `channel_ids`) or a
/// `[channels, samples]` array (channels `0..channels`).
fn template(item: &Bound<'_, PyAny>) -> PyResult<WaveformTemplate> {
    let (mean, ids) = match item.cast::<PyDict>() {
        Ok(dict) => {
            let mean = dict.get_item("mean")?.ok_or_else(|| PyValueError::new_err("template dict without 'mean'"))?;
            let ids: Option<Vec<usize>> = dict.get_item("channel_ids")?.map(|c| c.extract()).transpose()?;
            (mean, ids)
        }
        Err(_) => (item.clone(), None),
    };
    let mean = F32Array::new(&mean)?;
    let [channels, samples] = *mean.shape() else { return Err(PyValueError::new_err("a template mean must be [channels, samples]")) };
    let ids = ids.unwrap_or_else(|| (0..channels).collect());
    Ok(WaveformTemplate::new(ids, samples, mean.slice().to_vec(), vec![STD_NOT_GIVEN; channels * samples]))
}

/// Greedy matching pursuit of `templates` in `data` (`[channels, samples]`) on the device, with
/// amplitudes bounded to `[min_amplitude_scale, max_amplitude_scale]`; returns dicts
/// (`unit_id`, `sample`, `subsample_lag`, `amplitude_scale`, `score`).
#[pyfunction(name = "match_spikes_matching_pursuit")]
#[pyo3(signature = (data, templates, *, min_amplitude_scale=DEFAULT_MIN_AMPLITUDE_SCALE, max_amplitude_scale=DEFAULT_MAX_AMPLITUDE_SCALE, min_explained_energy=DEFAULT_MIN_EXPLAINED_ENERGY_UV2, max_passes=DEFAULT_MAX_PASSES, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn match_spikes_matching_pursuit_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    templates: Bound<'py, PyList>,
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    struct Task<'a> {
        x: &'a [f32],
        channels: usize,
        samples: usize,
        templates: &'a [WaveformTemplate],
        bounds: (f32, f32, f32, usize),
    }
    impl ComputeTask for Task<'_> {
        type Output = Vec<dsp_synapse::core::MatchedSpike>;
        fn run(self, client: Client) -> Self::Output {
            let (lo, hi, energy, passes) = self.bounds;
            match_spikes_matching_pursuit(&client, self.x, self.channels, self.samples, self.templates, lo, hi, energy, passes)
        }
    }
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let templates: Vec<WaveformTemplate> = templates.iter().map(|t| template(&t)).collect::<PyResult<_>>()?;
    let target = target(runtime)?;
    let task = Task { x: input.slice(), channels, samples, templates: &templates, bounds: (min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes) };
    let matched = py.detach(|| target.run(task)).map_err(runtime_error)?;
    let out = PyList::empty(py);
    for m in matched {
        let d = PyDict::new(py);
        d.set_item("unit_id", m.unit_id)?;
        d.set_item("sample", m.sample_index)?;
        d.set_item("subsample_lag", m.subsample_lag)?;
        d.set_item("amplitude_scale", m.amplitude_scale)?;
        d.set_item("score", m.score)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Motor units of HD-EMG `data` (`[channels, samples]` at `fs` Hz) by convolutive blind source
/// separation (extension, whitening, FastICA on the device, peak picking). `min_pnr_db` drops
/// units below that pulse-to-noise ratio when given.
#[pyfunction]
#[pyo3(signature = (data, *, fs, extension_factor, num_sources, refractory_ms, min_pnr_db=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn decompose_hdemg_cbss<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    fs: f64,
    extension_factor: usize,
    num_sources: usize,
    refractory_ms: f64,
    min_pnr_db: Option<f32>,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    struct Task<'a>(ConvolutiveBssDecomposer, &'a [f32], usize, usize, f64);
    impl ComputeTask for Task<'_> {
        type Output = Vec<MotorUnitPulseTrain>;
        fn run(self, client: Client) -> Self::Output {
            self.0.decompose(&client, self.1, self.2, self.3, self.4)
        }
    }
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let decomposer = ConvolutiveBssDecomposer::new(extension_factor, num_sources, refractory_ms);
    let (x, target) = (input.slice(), target(runtime)?);
    let units = py.detach(|| target.run(Task(decomposer, x, channels, samples, fs))).map_err(runtime_error)?;
    let out = PyList::empty(py);
    for u in units.into_iter().filter(|u| min_pnr_db.is_none_or(|min| u.pnr_db >= min)) {
        let d = PyDict::new(py);
        d.set_item("unit_id", u.unit_id)?;
        d.set_item("spike_samples", u.spike_samples)?;
        d.set_item("pnr_db", u.pnr_db)?;
        d.set_item("cov_isi", u.cov_isi)?;
        let len = u.ipt.len();
        d.set_item("ipt", to_numpy(py, u.ipt, &[len])?)?;
        out.append(d)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cluster_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(hdbscan, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_density_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_kde_merge, m)?)?;
    m.add_function(wrap_pyfunction!(match_spikes_matching_pursuit_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_hdemg_cbss, m)?)?;
    Ok(())
}
