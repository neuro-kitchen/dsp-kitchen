//! `dspk.synapse.clustering`, `dspk.synapse.matching` and `dspk.synapse.spatial`: the building
//! blocks of a sorter on in-memory arrays — clustering of feature points (k-means, HDBSCAN, GMM,
//! density peaks, KDE valley merging, peak subsampling), templates (similarity, merge
//! suggestions, matching pursuit) and space (spike localisation, probe drift).

use dspk_synapse::sorting::{self as srt, GmmClusterer, GmmCovarianceKind, HdbscanOptions, KMeansOptions, KdeMergeOptions, DensityPeaksOptions, MatchingPursuitMatcher, SubsampleOptions};
use dspk_synapse::spatial::drift::DriftOptions;
use dspk_synapse::{self as syn, GridConvolutionLocalizer, WaveformTemplate};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::io::Probe;

type Dict<'py> = Bound<'py, PyDict>;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn client(runtime: Option<&str>) -> PyResult<cubecl::prelude::Client> {
    dspk_py_core::runtime::open_client(dspk_py_core::runtime::target(runtime)?).map_err(PyRuntimeError::new_err)
}

/// `[n, d]` points, row-major, and their shape.
fn points(x: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let a = x.as_array();
    (a.iter().copied().collect(), a.nrows(), a.ncols())
}

fn matrix<'py, T: numpy::Element>(py: Python<'py>, rows: usize, cols: usize, data: Vec<T>) -> PyResult<Bound<'py, numpy::PyArray2<T>>> {
    Ok(Array2::from_shape_vec((rows, cols), data).map_err(invalid)?.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// Clustering
// ----------------------------------------------------------------------------

/// k-means (k-means++ starts, `n_init` restarts, the lowest inertia kept) of `points` `(n, d)` on
/// the runtime's device: `labels`, `centers` `(k, d)`, `inertia`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (points, k, *, n_init, max_iter, tol, seed, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn kmeans<'py>(py: Python<'py>, points: PyReadonlyArray2<'_, f32>, k: usize, n_init: usize, max_iter: usize, tol: f64, seed: u64, runtime: Option<&str>) -> PyResult<Dict<'py>> {
    let (x, n, d) = self::points(&points);
    if k == 0 || k > n {
        return Err(invalid(format!("k must be in 1..={n} (the number of points), got {k}")));
    }
    let client = client(runtime)?;
    let options = KMeansOptions { n_init, max_iter, tol, seed };
    let r = py.detach(|| srt::cluster_kmeans(&client, &x, n, d, k, &options));
    let out = PyDict::new(py);
    out.set_item("labels", r.labels.iter().map(|&l| l as i64).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("centers", matrix(py, k, d, r.centers)?)?;
    out.set_item("inertia", r.inertia)?;
    Ok(out)
}

/// HDBSCAN labels of `points` `(n, d)` on the runtime's device: cluster per point (`0..`), `-1`
/// for noise.
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (points, *, min_cluster_size, allow_single_cluster, runtime=None))]
pub fn hdbscan<'py>(py: Python<'py>, points: PyReadonlyArray2<'_, f32>, min_cluster_size: usize, allow_single_cluster: bool, runtime: Option<&str>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    let (x, n, d) = self::points(&points);
    if min_cluster_size < 2 {
        return Err(invalid("min_cluster_size must be at least 2"));
    }
    let client = client(runtime)?;
    let options = HdbscanOptions { min_cluster_size, allow_single_cluster };
    let labels = py.detach(|| srt::cluster_hdbscan(&client, &x, n, d, &options));
    Ok(labels.into_iter().map(i64::from).collect::<Vec<_>>().into_pyarray(py))
}

fn covariance(kind: &str) -> PyResult<GmmCovarianceKind> {
    match kind {
        "full" => Ok(GmmCovarianceKind::Full),
        "diagonal" => Ok(GmmCovarianceKind::Diagonal),
        "masked" => Ok(GmmCovarianceKind::Masked),
        other => Err(invalid(format!("covariance is full, diagonal or masked, got {other:?}"))),
    }
}

/// A Gaussian mixture of `points` `(n, d)` with the number of components chosen by BIC in
/// `k_min..=k_max` (EM on the runtime's device). `labels` (`-1`: beyond `outlier_mahalanobis_sq`),
/// `weights`, `means` `(k, d)`, `covariances` `(k, d, d)`, `responsibilities` `(n, k)`,
/// `mahalanobis_sq`, `log_likelihood`, `bic`. `mask` `(n, d)` weights the features of each
/// point (masked covariance only).
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (points, *, k_min, k_max, covariance, max_iterations, tolerance, regularization, outlier_mahalanobis_sq=None, mask=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn gmm<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f32>,
    k_min: usize,
    k_max: usize,
    covariance: &str,
    max_iterations: usize,
    tolerance: f64,
    regularization: f32,
    outlier_mahalanobis_sq: Option<f32>,
    mask: Option<PyReadonlyArray2<'_, f32>>,
    runtime: Option<&str>,
) -> PyResult<Dict<'py>> {
    let (x, n, d) = self::points(&points);
    let mask: Option<Vec<f32>> = match mask {
        Some(m) => {
            let (m, mn, md) = self::points(&m);
            if (mn, md) != (n, d) {
                return Err(invalid(format!("mask must be {n} × {d}, got {mn} × {md}")));
            }
            Some(m)
        }
        None => None,
    };
    let clusterer = GmmClusterer { max_iterations, tolerance, regularization, outlier_mahalanobis_sq, ..GmmClusterer::new(k_min, k_max, self::covariance(covariance)?) };
    let client = client(runtime)?;
    let r = py.detach(|| clusterer.fit(&client, &x, n, d, mask.as_deref()));
    let k = r.num_clusters;
    let out = PyDict::new(py);
    out.set_item("labels", r.labels.iter().map(|&l| i64::from(l)).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("weights", r.weights.into_pyarray(py))?;
    out.set_item("means", matrix(py, k, d, r.means)?)?;
    out.set_item("covariances", numpy::ndarray::Array3::from_shape_vec((k, d, d), r.covariances).map_err(invalid)?.into_pyarray(py))?;
    out.set_item("responsibilities", matrix(py, n, k, r.responsibilities)?)?;
    out.set_item("mahalanobis_sq", r.mahalanobis_sq.into_pyarray(py))?;
    out.set_item("log_likelihood", r.log_likelihood)?;
    out.set_item("bic", r.bic)?;
    Ok(out)
}

/// Rodriguez–Laio density peaks of `points` `(n, d)` (host): `labels`, `centers` (indices of the
/// exemplar points), `densities`, `deltas`. `max_points` clusters an evenly strided subsample
/// (the others take their nearest subsampled point's label).
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (points, *, cutoff_distance, num_clusters, max_points=None))]
pub fn density_peaks<'py>(py: Python<'py>, points: PyReadonlyArray2<'_, f32>, cutoff_distance: f32, num_clusters: usize, max_points: Option<usize>) -> PyResult<Dict<'py>> {
    let (x, n, d) = self::points(&points);
    let options = DensityPeaksOptions { cutoff_distance, num_clusters, max_points };
    let r = py.detach(|| srt::cluster_density_peaks(&x, n, d, &options));
    let out = PyDict::new(py);
    out.set_item("labels", r.labels.iter().map(|&l| l as i64).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("centers", r.cluster_centers.iter().map(|&c| c as u64).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("densities", r.densities.into_pyarray(py))?;
    out.set_item("deltas", r.deltas.into_pyarray(py))?;
    Ok(out)
}

/// Over-clusters `points` `(n, d)` with k-means (`initial_k`), then merges pairs with no density
/// valley between them (dip score below `dip_threshold`) and clusters smaller than
/// `min_cluster_size` (host): `labels`, `centroids` `(clusters, d)`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (points, *, initial_k, dip_threshold, min_cluster_size))]
pub fn kde_merge<'py>(py: Python<'py>, points: PyReadonlyArray2<'_, f32>, initial_k: usize, dip_threshold: f32, min_cluster_size: usize) -> PyResult<Dict<'py>> {
    let (x, n, d) = self::points(&points);
    let options = KdeMergeOptions { initial_k, dip_threshold, min_cluster_size };
    let r = py.detach(|| srt::cluster_kde_merge(&x, n, d, &options));
    let out = PyDict::new(py);
    out.set_item("labels", r.labels.iter().map(|&l| i64::from(l)).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("centroids", matrix(py, r.num_clusters, d, r.centroids)?)?;
    Ok(out)
}

/// Indices (sorted by sample) of `n_peaks` peaks chosen at random from `samples` / `channels`
/// (in total, or of each channel with `per_channel`).
#[gen_stub_pyfunction(module = "dspk._native.synapse.clustering")]
#[pyfunction]
#[pyo3(signature = (samples, channels, *, n_peaks, per_channel, seed))]
pub fn subsample_peaks<'py>(py: Python<'py>, samples: PyReadonlyArray1<'_, u64>, channels: PyReadonlyArray1<'_, u32>, n_peaks: usize, per_channel: bool, seed: u64) -> PyResult<Bound<'py, numpy::PyArray1<u64>>> {
    let (s, c) = (samples.as_slice()?, channels.as_slice()?);
    if s.len() != c.len() {
        return Err(invalid("samples and channels differ in length"));
    }
    let channels: Vec<usize> = c.iter().map(|&c| c as usize).collect();
    let kept = srt::subsample_peaks(s, &channels, &SubsampleOptions { n_peaks, per_channel, seed });
    Ok(kept.into_iter().map(|i| i as u64).collect::<Vec<_>>().into_pyarray(py))
}

// ----------------------------------------------------------------------------
// Templates and matching
// ----------------------------------------------------------------------------

/// Templates from `mean` `(units, rows, samples)`; row `r` of unit `u` is recording channel
/// `channels[u, r]` (default: rows are channels `0..rows`).
fn templates(mean: &PyReadonlyArray3<'_, f32>, channels: Option<&PyReadonlyArray2<'_, i64>>) -> PyResult<Vec<WaveformTemplate>> {
    let a = mean.as_array();
    let (u, rows, len) = a.dim();
    if let Some(c) = channels {
        if c.as_array().dim() != (u, rows) {
            return Err(invalid(format!("channels must be {u} × {rows} (units × rows of the templates), got {:?}", c.as_array().dim())));
        }
    }
    (0..u)
        .map(|i| {
            let ids: Vec<usize> = match channels {
                Some(c) => c.as_array().row(i).iter().map(|&v| usize::try_from(v).map_err(|_| invalid(format!("unit {i}: a negative channel")))).collect::<PyResult<_>>()?,
                None => (0..rows).collect(),
            };
            let m: Vec<f32> = a.index_axis(numpy::ndarray::Axis(0), i).iter().copied().collect();
            Ok(WaveformTemplate::new(ids, len, m, vec![0.0; rows * len]))
        })
        .collect()
}

/// Lag-maximised cosine similarity of every pair of `templates` `(units, rows, samples)` (rows
/// paired by recording channel, `channels` `(units, rows)`; default: rows are channels), lags up
/// to `max_lag_samples`: a `(units, units)` matrix.
#[gen_stub_pyfunction(module = "dspk._native.synapse.matching")]
#[pyfunction]
#[pyo3(signature = (templates, *, max_lag_samples, channels=None))]
pub fn template_similarity<'py>(py: Python<'py>, templates: PyReadonlyArray3<'_, f32>, max_lag_samples: usize, channels: Option<PyReadonlyArray2<'_, i64>>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let t = self::templates(&templates, channels.as_ref())?;
    let u = t.len();
    let m = py.detach(|| srt::compute_template_similarity_matrix(&t, max_lag_samples));
    matrix(py, u, u, m)
}

/// Pairs of units whose templates' similarity is at least `threshold`: `(i, j, similarity)`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.matching")]
#[pyfunction]
#[pyo3(signature = (templates, *, max_lag_samples, threshold, channels=None))]
pub fn suggest_merges(py: Python<'_>, templates: PyReadonlyArray3<'_, f32>, max_lag_samples: usize, threshold: f32, channels: Option<PyReadonlyArray2<'_, i64>>) -> PyResult<Vec<(usize, usize, f32)>> {
    let t = self::templates(&templates, channels.as_ref())?;
    Ok(py.detach(|| srt::suggest_template_merges(&t, max_lag_samples, threshold)))
}

/// Matching pursuit of `data` `(channels, samples)` against `templates` `(units, rows, samples)`
/// on the runtime's device: each pass fits the template (amplitude in `[min_amplitude_scale,
/// max_amplitude_scale]`) that explains the most energy (at least `min_explained_energy`, µV²)
/// and subtracts it; `max_passes` passes resolve overlaps. `samples` (each match's trough),
/// `units`, `amplitudes` (scales), `scores`, by time.
#[gen_stub_pyfunction(module = "dspk._native.synapse.matching")]
#[pyfunction]
#[pyo3(signature = (data, templates, *, min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes, channels=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn matching_pursuit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    templates: PyReadonlyArray3<'_, f32>,
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
    channels: Option<PyReadonlyArray2<'_, i64>>,
    runtime: Option<&str>,
) -> PyResult<Dict<'py>> {
    let (x, c, n) = self::points(&data);
    let t = self::templates(&templates, channels.as_ref())?;
    let settings = MatchingPursuitMatcher { min_amplitude_scale, max_amplitude_scale, min_explained_energy, max_passes };
    let client = client(runtime)?;
    let m = py.detach(|| srt::match_spikes_matching_pursuit(&client, &x, c, n, &t, &settings));
    let out = PyDict::new(py);
    out.set_item("samples", m.iter().map(|s| s.sample_index).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("units", m.iter().map(|s| s.unit_id as i64).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("amplitudes", m.iter().map(|s| s.amplitude_scale).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("scores", m.iter().map(|s| s.score).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

// ----------------------------------------------------------------------------
// Spatial
// ----------------------------------------------------------------------------

/// Position `(spikes, 3)` (x, y, z µm on `probe`) of each spike from its snippets `waveforms`
/// `(spikes, k, samples)` on `channels` `(spikes, k)` (as `dspk.synapse.extraction.snippets`
/// gives them). `method`: `center_of_mass` (peak-to-peak weights to `power`), `monopolar`
/// (Levenberg–Marquardt, `max_iterations`), `grid_convolution` (`xy_radius_um`, `xy_step_um`,
/// `z_values_um`, `softmax_temperature`) or `dipole` (signed peaks, `max_iterations`).
#[gen_stub_pyfunction(module = "dspk._native.synapse.spatial")]
#[pyfunction]
#[pyo3(signature = (waveforms, channels, probe, *, method, power, max_iterations, xy_radius_um, xy_step_um, z_values_um, softmax_temperature))]
#[allow(clippy::too_many_arguments)]
pub fn localize<'py>(
    py: Python<'py>,
    waveforms: PyReadonlyArray3<'_, f32>,
    channels: PyReadonlyArray2<'_, i64>,
    probe: &Probe,
    method: &str,
    power: f32,
    max_iterations: usize,
    xy_radius_um: f32,
    xy_step_um: f32,
    z_values_um: Vec<f32>,
    softmax_temperature: f32,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let w = waveforms.as_array();
    let c = channels.as_array();
    let (n, k, _) = w.dim();
    if c.dim() != (n, k) {
        return Err(invalid(format!("channels must be {n} × {k} (spikes × snippet channels), got {:?}", c.dim())));
    }
    if !matches!(method, "center_of_mass" | "monopolar" | "grid_convolution" | "dipole") {
        return Err(invalid(format!("method is center_of_mass, monopolar, grid_convolution or dipole, got {method:?}")));
    }
    let grid = GridConvolutionLocalizer { xy_radius_um, xy_step_um, z_values_um, softmax_temperature };
    let layout = &probe.inner;
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let ids: Vec<usize> = c.row(i).iter().map(|&v| v.max(0) as usize).collect();
        let rows = w.index_axis(numpy::ndarray::Axis(0), i);
        let ptp: Vec<f32> = rows.rows().into_iter().map(|r| r.iter().fold(f32::MIN, |a, &b| a.max(b)) - r.iter().fold(f32::MAX, |a, &b| a.min(b))).collect();
        let p = match method {
            "center_of_mass" => syn::localize_spike_center_of_mass(&ids, &ptp, layout, power),
            "monopolar" => syn::localize_spike_monopolar(&ids, &ptp, layout, max_iterations).0,
            "grid_convolution" => syn::localize_spike_grid_convolution(&ids, &ptp, layout, &grid),
            _ => {
                // Signed peak of each row: the value of largest magnitude
                let peaks: Vec<f32> = rows.rows().into_iter().map(|r| r.iter().copied().fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a })).collect();
                syn::localize_spike_dipole(&ids, &peaks, layout, max_iterations).position_um
            }
        };
        out.extend_from_slice(&p);
    }
    matrix(py, n, 3, out)
}

/// Probe drift from spikes at `samples` (of a recording of `total_samples` at `sample_rate` Hz)
/// with their `depths_um` and `amplitudes_uv`, between `depth_min_um` and `depth_max_um`. Rigid:
/// `times_s`, `drift_um` (per time bin), `activity` `(time bins, depth bins)`. Non-rigid
/// (`depth_blocks` > 1 and `rigid=False`): `times_s`, `block_centers_um`, `drift_um`
/// `(blocks, time bins)`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.spatial")]
#[pyfunction]
#[pyo3(signature = (samples, depths_um, amplitudes_uv, total_samples, sample_rate, *, depth_min_um, depth_max_um, time_bin_s, depth_bin_um, max_drift_um, depth_blocks, rigid))]
#[allow(clippy::too_many_arguments)]
pub fn drift<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, u64>,
    depths_um: PyReadonlyArray1<'_, f32>,
    amplitudes_uv: PyReadonlyArray1<'_, f32>,
    total_samples: u64,
    sample_rate: f64,
    depth_min_um: f32,
    depth_max_um: f32,
    time_bin_s: f64,
    depth_bin_um: f32,
    max_drift_um: f32,
    depth_blocks: usize,
    rigid: bool,
) -> PyResult<Dict<'py>> {
    let (s, d, a) = (samples.as_slice()?, depths_um.as_slice()?, amplitudes_uv.as_slice()?);
    if s.len() != d.len() || s.len() != a.len() {
        return Err(invalid("samples, depths_um and amplitudes_uv differ in length"));
    }
    if !(depth_max_um > depth_min_um) {
        return Err(invalid("depth_max_um must be above depth_min_um"));
    }
    let options = DriftOptions { time_bin_s, depth_min_um, depth_max_um, depth_bin_um, max_drift_um, depth_blocks };
    let out = PyDict::new(py);
    if rigid {
        let e = py.detach(|| syn::estimate_rigid_drift(s, d, a, total_samples, sample_rate, &options));
        out.set_item("times_s", e.time_bin_centers_sec.into_pyarray(py))?;
        out.set_item("drift_um", e.drift_um.into_pyarray(py))?;
        out.set_item("activity", matrix(py, e.num_time_bins, e.num_depth_bins, e.activity_map)?)?;
    } else {
        let e = py.detach(|| syn::estimate_nonrigid_drift(s, d, a, total_samples, sample_rate, &options));
        out.set_item("times_s", e.time_bin_centers_sec.into_pyarray(py))?;
        out.set_item("block_centers_um", e.block_centers_um.into_pyarray(py))?;
        out.set_item("drift_um", matrix(py, e.num_depth_blocks, e.num_time_bins, e.block_drift_um)?)?;
    }
    Ok(out)
}

pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_N_INIT", usize, KMeansOptions::default().n_init);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_MAX_ITER", usize, KMeansOptions::default().max_iter);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_TOL", f64, KMeansOptions::default().tol);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_MIN_CLUSTER_SIZE", usize, HdbscanOptions::default().min_cluster_size);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_GMM_K_MIN", usize, GmmClusterer::default().k_min);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_GMM_K_MAX", usize, GmmClusterer::default().k_max);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_GMM_MAX_ITERATIONS", usize, GmmClusterer::default().max_iterations);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_GMM_TOLERANCE", f64, GmmClusterer::default().tolerance);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_GMM_REGULARIZATION", f32, GmmClusterer::default().regularization);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_DIP_THRESHOLD", f32, srt::DEFAULT_DIP_THRESHOLD);
pyo3_stub_gen::module_variable!("dspk._native.synapse.clustering", "DEFAULT_KDE_MIN_CLUSTER_SIZE", usize, srt::DEFAULT_KDE_MIN_CLUSTER_SIZE);
pyo3_stub_gen::module_variable!("dspk._native.synapse.matching", "DEFAULT_MIN_AMPLITUDE_SCALE", f32, MatchingPursuitMatcher::default().min_amplitude_scale);
pyo3_stub_gen::module_variable!("dspk._native.synapse.matching", "DEFAULT_MAX_AMPLITUDE_SCALE", f32, MatchingPursuitMatcher::default().max_amplitude_scale);
pyo3_stub_gen::module_variable!("dspk._native.synapse.matching", "DEFAULT_MIN_EXPLAINED_ENERGY", f32, MatchingPursuitMatcher::default().min_explained_energy);
pyo3_stub_gen::module_variable!("dspk._native.synapse.matching", "DEFAULT_MAX_PASSES", usize, MatchingPursuitMatcher::default().max_passes);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_COM_POWER", f32, syn::CenterOfMassLocalizer::default().amplitude_power);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_MONOPOLAR_ITERATIONS", usize, syn::MonopolarTriangulator::default().max_iterations);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_DIPOLE_ITERATIONS", usize, syn::DipoleLocalizer::default().max_iterations);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_GRID_XY_RADIUS_UM", f32, GridConvolutionLocalizer::default().xy_radius_um);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_GRID_XY_STEP_UM", f32, GridConvolutionLocalizer::default().xy_step_um);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_GRID_Z_VALUES_UM", Vec<f32>, GridConvolutionLocalizer::default().z_values_um.clone());
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_GRID_SOFTMAX_TEMPERATURE", f32, GridConvolutionLocalizer::default().softmax_temperature);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_DRIFT_TIME_BIN_S", f64, DriftOptions::new(0.0, 1.0).time_bin_s);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_DRIFT_DEPTH_BIN_UM", f32, DriftOptions::new(0.0, 1.0).depth_bin_um);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_MAX_DRIFT_UM", f32, DriftOptions::new(0.0, 1.0).max_drift_um);
pyo3_stub_gen::module_variable!("dspk._native.synapse.spatial", "DEFAULT_DRIFT_DEPTH_BLOCKS", usize, DriftOptions::new(0.0, 1.0).depth_blocks);

pub(crate) fn register_clustering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let k = KMeansOptions::default();
    m.add("DEFAULT_N_INIT", k.n_init)?;
    m.add("DEFAULT_MAX_ITER", k.max_iter)?;
    m.add("DEFAULT_TOL", k.tol)?;
    m.add("DEFAULT_MIN_CLUSTER_SIZE", HdbscanOptions::default().min_cluster_size)?;
    let g = GmmClusterer::default();
    m.add("DEFAULT_GMM_K_MIN", g.k_min)?;
    m.add("DEFAULT_GMM_K_MAX", g.k_max)?;
    m.add("DEFAULT_GMM_MAX_ITERATIONS", g.max_iterations)?;
    m.add("DEFAULT_GMM_TOLERANCE", g.tolerance)?;
    m.add("DEFAULT_GMM_REGULARIZATION", g.regularization)?;
    m.add("DEFAULT_DIP_THRESHOLD", srt::DEFAULT_DIP_THRESHOLD)?;
    m.add("DEFAULT_KDE_MIN_CLUSTER_SIZE", srt::DEFAULT_KDE_MIN_CLUSTER_SIZE)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(hdbscan, m)?)?;
    m.add_function(wrap_pyfunction!(gmm, m)?)?;
    m.add_function(wrap_pyfunction!(density_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(kde_merge, m)?)?;
    m.add_function(wrap_pyfunction!(subsample_peaks, m)?)?;
    Ok(())
}

pub(crate) fn register_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mp = MatchingPursuitMatcher::default();
    m.add("DEFAULT_MIN_AMPLITUDE_SCALE", mp.min_amplitude_scale)?;
    m.add("DEFAULT_MAX_AMPLITUDE_SCALE", mp.max_amplitude_scale)?;
    m.add("DEFAULT_MIN_EXPLAINED_ENERGY", mp.min_explained_energy)?;
    m.add("DEFAULT_MAX_PASSES", mp.max_passes)?;
    m.add_function(wrap_pyfunction!(template_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_merges, m)?)?;
    m.add_function(wrap_pyfunction!(matching_pursuit, m)?)?;
    Ok(())
}

pub(crate) fn register_spatial(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let g = GridConvolutionLocalizer::default();
    m.add("DEFAULT_COM_POWER", syn::CenterOfMassLocalizer::default().amplitude_power)?;
    m.add("DEFAULT_MONOPOLAR_ITERATIONS", syn::MonopolarTriangulator::default().max_iterations)?;
    m.add("DEFAULT_DIPOLE_ITERATIONS", syn::DipoleLocalizer::default().max_iterations)?;
    m.add("DEFAULT_GRID_XY_RADIUS_UM", g.xy_radius_um)?;
    m.add("DEFAULT_GRID_XY_STEP_UM", g.xy_step_um)?;
    m.add("DEFAULT_GRID_Z_VALUES_UM", g.z_values_um.clone())?;
    m.add("DEFAULT_GRID_SOFTMAX_TEMPERATURE", g.softmax_temperature)?;
    let d = DriftOptions::new(0.0, 1.0);
    m.add("DEFAULT_DRIFT_TIME_BIN_S", d.time_bin_s)?;
    m.add("DEFAULT_DRIFT_DEPTH_BIN_UM", d.depth_bin_um)?;
    m.add("DEFAULT_MAX_DRIFT_UM", d.max_drift_um)?;
    m.add("DEFAULT_DRIFT_DEPTH_BLOCKS", d.depth_blocks)?;
    m.add_function(wrap_pyfunction!(localize, m)?)?;
    m.add_function(wrap_pyfunction!(drift, m)?)?;
    Ok(())
}
