use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{to_numpy, F32Array};
use dsp_synapse::core::WaveformTemplate;
use dsp_synapse::sorting::{
    cluster_density_peaks as rust_cluster_density_peaks,
    cluster_isosplit as rust_cluster_isosplit, match_spikes_omp as rust_match_spikes_omp,
    ConvolutiveBssDecomposer, GmmClusterer, GmmCovarianceKind,
};

/// Cluster a 2D feature matrix `[n_spikes, n_features]` using Gaussian Mixture Models (GMM) with
/// Expectation-Maximization, K-means++ initialization, and automatic BIC cluster count selection.
///
/// `covariance_type` may be `"diagonal"`, `"full"`, or `"masked"` (KlustaKwik Masked EM).
/// If `channel_masks` (`[n_spikes, n_features]` in `[0, 1]`) is passed, Masked EM is used automatically.
#[pyfunction]
#[pyo3(signature = (features, min_clusters=1, max_clusters=8, covariance_type="diagonal", max_iters=100, reg_covar=1e-3, channel_masks=None))]
pub fn cluster_gmm<'py>(
    py: Python<'py>,
    features: Bound<'py, PyAny>,
    min_clusters: usize,
    max_clusters: usize,
    covariance_type: &str,
    max_iters: usize,
    reg_covar: f32,
    channel_masks: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let feat_arr = F32Array::new(&features)?;
    if feat_arr.shape().len() != 2 {
        return Err(PyValueError::new_err(
            "cluster_gmm requires a 2D feature matrix [n_spikes, n_features]",
        ));
    }
    let (n, d) = (feat_arr.shape()[0], feat_arr.shape()[1]);

    let mut cov_kind = match covariance_type.to_ascii_lowercase().as_str() {
        "diagonal" | "diag" => GmmCovarianceKind::Diagonal,
        "full" => GmmCovarianceKind::Full,
        "masked" => GmmCovarianceKind::Masked,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown covariance_type '{other}'. Expected 'diagonal', 'full', or 'masked'."
            )))
        }
    };
    if channel_masks.is_some() {
        cov_kind = GmmCovarianceKind::Masked;
    }

    let mut clusterer = GmmClusterer::new(min_clusters, max_clusters, cov_kind);
    clusterer.max_iterations = max_iters;
    clusterer.regularization = reg_covar;

    let res = if let Some(mask_obj) = channel_masks {
        let mask_arr = F32Array::new(&mask_obj)?;
        if mask_arr.shape() != feat_arr.shape() {
            return Err(PyValueError::new_err(
                "channel_masks must have the same [n_spikes, n_features] shape as features",
            ));
        }
        clusterer.fit(feat_arr.slice(), n, d, Some(mask_arr.slice()))
    } else {
        clusterer.fit(feat_arr.slice(), n, d, None)
    };

    let k = res.num_clusters;
    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("num_clusters", k)?;
    dict.set_item("weights", to_numpy(py, res.weights, &[k])?)?;
    dict.set_item("means", to_numpy(py, res.means, &[k, d])?)?;
    dict.set_item(
        "responsibilities",
        to_numpy(py, res.responsibilities, &[n, k])?,
    )?;
    dict.set_item("log_likelihood", res.log_likelihood)?;
    dict.set_item("bic", res.bic)?;
    Ok(dict)
}

/// Cluster a 2D feature matrix `[n_spikes, n_features]` using Rodriguez & Laio (2014) Density Peaks.
#[pyfunction]
#[pyo3(signature = (features, dc=1.5, num_clusters=4))]
pub fn cluster_density_peaks<'py>(
    py: Python<'py>,
    features: Bound<'py, PyAny>,
    dc: f32,
    num_clusters: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let feat_arr = F32Array::new(&features)?;
    if feat_arr.shape().len() != 2 {
        return Err(PyValueError::new_err(
            "cluster_density_peaks requires a 2D feature matrix [n_spikes, n_features]",
        ));
    }
    let (n, d) = (feat_arr.shape()[0], feat_arr.shape()[1]);
    let res = rust_cluster_density_peaks(feat_arr.slice(), n, d, dc, num_clusters);

    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("num_clusters", res.cluster_centers.len())?;
    dict.set_item("cluster_centers", res.cluster_centers)?;
    dict.set_item("rho", to_numpy(py, res.densities, &[n])?)?;
    dict.set_item("delta", to_numpy(py, res.deltas, &[n])?)?;
    Ok(dict)
}

/// Non-parametric IsoSplit clustering (Magland & Barnett 2015 / MountainSort) via 1D dip/unimodality tests.
#[pyfunction]
#[pyo3(signature = (features, initial_k=10, dip_threshold=2.0, min_cluster_size=10))]
pub fn cluster_isosplit<'py>(
    py: Python<'py>,
    features: Bound<'py, PyAny>,
    initial_k: usize,
    dip_threshold: f32,
    min_cluster_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let feat_arr = F32Array::new(&features)?;
    if feat_arr.shape().len() != 2 {
        return Err(PyValueError::new_err(
            "cluster_isosplit requires a 2D feature matrix [n_spikes, n_features]",
        ));
    }
    let (n, d) = (feat_arr.shape()[0], feat_arr.shape()[1]);
    let res = rust_cluster_isosplit(
        feat_arr.slice(),
        n,
        d,
        initial_k,
        dip_threshold,
        min_cluster_size,
    );

    let dict = PyDict::new(py);
    dict.set_item("labels", res.labels)?;
    dict.set_item("num_clusters", res.num_clusters)?;
    dict.set_item(
        "centroids",
        to_numpy(py, res.centroids, &[res.num_clusters, d])?,
    )?;
    Ok(dict)
}

/// Deconvolve overlapping spikes against unit templates using Orthogonal Matching Pursuit (OMP).
///
/// `templates` is a list of template dicts or `(channel_ids, mean_waveform)` pairs where `mean_waveform`
/// has shape `[num_channels, num_samples]`.
#[pyfunction]
#[pyo3(signature = (data, templates, min_amplitude_scale=0.65, max_amplitude_scale=1.45, min_explained_energy=500.0, max_passes=4, channels=None))]
pub fn match_spikes_omp<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    templates: Bound<'py, PyList>,
    min_amplitude_scale: f32,
    max_amplitude_scale: f32,
    min_explained_energy: f32,
    max_passes: usize,
    channels: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let input = F32Array::new(&data)?;
    let (ch, samples) = input.channels_samples(channels)?;

    let mut rust_templates = Vec::with_capacity(templates.len());
    for item in templates.iter() {
        if let Ok(dict) = item.cast::<PyDict>() {
            let mean_obj = dict
                .get_item("mean")?
                .ok_or_else(|| PyValueError::new_err("template dict missing 'mean' array"))?;
            let mean_arr = F32Array::new(&mean_obj)?;
            if mean_arr.shape().len() != 2 {
                return Err(PyValueError::new_err(
                    "template 'mean' must be 2D [num_channels, num_samples]",
                ));
            }
            let (t_ch, t_samples) = (mean_arr.shape()[0], mean_arr.shape()[1]);
            let channel_ids: Vec<usize> = match dict.get_item("channel_ids")? {
                Some(c_obj) => c_obj.extract()?,
                None => (0..t_ch).collect(),
            };
            let std_vec = vec![1.0f32; t_ch * t_samples];
            rust_templates.push(WaveformTemplate::new(
                channel_ids,
                t_samples,
                mean_arr.slice().to_vec(),
                std_vec,
            ));
        } else {
            let mean_arr = F32Array::new(&item)?;
            if mean_arr.shape().len() != 2 {
                return Err(PyValueError::new_err(
                    "Each template must be a 2D array [num_channels, num_samples] or template dict",
                ));
            }
            let (t_ch, t_samples) = (mean_arr.shape()[0], mean_arr.shape()[1]);
            rust_templates.push(WaveformTemplate::new(
                (0..t_ch).collect(),
                t_samples,
                mean_arr.slice().to_vec(),
                vec![1.0f32; t_ch * t_samples],
            ));
        }
    }

    let matched = rust_match_spikes_omp(
        input.slice(),
        ch,
        samples,
        &rust_templates,
        min_amplitude_scale,
        max_amplitude_scale,
        min_explained_energy,
        max_passes,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out_list = PyList::empty(py);
    for m in matched {
        let d = PyDict::new(py);
        d.set_item("unit_id", m.unit_id)?;
        d.set_item("sample_index", m.sample_index)?;
        d.set_item("subsample_lag", m.subsample_lag)?;
        d.set_item("amplitude_scale", m.amplitude_scale)?;
        d.set_item("score", m.score)?;
        out_list.append(d)?;
    }
    Ok(out_list)
}

/// Decompose multi-channel surface HD-EMG `[channels, samples]` into individual motor unit
/// spike trains via Convolutive Blind Source Separation (Holobar & Zazula / Negro et al. 2016).
#[pyfunction]
#[pyo3(signature = (data, sample_rate_hz=2048.0, extension_factor=6, num_units=8, min_pnr_db=15.0, min_distance_ms=20.0, channels=None))]
pub fn decompose_hdemg_cbss<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    sample_rate_hz: f64,
    extension_factor: usize,
    num_units: usize,
    min_pnr_db: f32,
    min_distance_ms: f64,
    channels: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let input = F32Array::new(&data)?;
    let (ch, samples) = input.channels_samples(channels)?;

    let decomposer = ConvolutiveBssDecomposer::new(extension_factor, num_units, min_distance_ms);
    let units = decomposer.decompose(input.slice(), ch, samples, sample_rate_hz);

    let out_list = PyList::empty(py);
    for u in units {
        if u.pnr_db < min_pnr_db {
            continue;
        }
        let d = PyDict::new(py);
        d.set_item("unit_id", u.unit_id)?;
        d.set_item("spike_samples", u.spike_samples)?;
        d.set_item("pnr_db", u.pnr_db)?;
        d.set_item("cov_isi", u.cov_isi)?;
        let len = u.ipt.len();
        d.set_item("ipt", to_numpy(py, u.ipt, &[len])?)?;
        out_list.append(d)?;
    }
    Ok(out_list)
}
