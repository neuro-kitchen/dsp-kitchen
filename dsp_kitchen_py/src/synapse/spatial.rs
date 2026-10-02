use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{to_numpy, F32Array};
use dsp_synapse::extraction::SnippetBatch;
use dsp_synapse::spatial::{
    correct_traces_drift_kriging, estimate_nonrigid_drift as rust_estimate_nonrigid_drift,
    estimate_rigid_drift as rust_estimate_rigid_drift, CenterOfMassLocalizer, DipoleLocalizer,
    DriftEstimate, GridConvolutionLocalizer, MonopolarTriangulator,
};
use dsp_synapse::traits::PeakLocalizer;

use super::extraction::PyWaveformSnippet;
use super::probe::PyProbeLayout;

/// Localize spike positions `[N, 3]` (`[x_um, y_um, z_um]`) from extracted waveform snippets.
///
/// Supported `method` values: `"center_of_mass"` (or `"com"`), `"monopolar"`, `"dipole"`, `"grid_convolution"`.
#[pyfunction]
#[pyo3(signature = (snippets, probe, method="center_of_mass", max_iterations=35))]
pub fn localize_spikes<'py>(
    py: Python<'py>,
    snippets: Vec<PyRef<PyWaveformSnippet>>,
    probe: PyRef<PyProbeLayout>,
    method: &str,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_snippets: Vec<_> = snippets.iter().map(|s| s.inner.clone()).collect();
    let Some(batch) = SnippetBatch::from_snippets(&rust_snippets) else {
        return to_numpy(py, Vec::new(), &[0, 3]);
    };
    let n = batch.num_spikes;
    let layout = &probe.inner;

    let coords = match method.to_ascii_lowercase().as_str() {
        "center_of_mass" | "com" => CenterOfMassLocalizer::default()
            .localize(&batch, layout)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        "monopolar" => MonopolarTriangulator { max_iterations }
            .localize(&batch, layout)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        "dipole" => DipoleLocalizer { max_iterations }
            .localize(&batch, layout)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        "grid_convolution" | "grid" => GridConvolutionLocalizer::default()
            .localize(&batch, layout)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown localization method '{other}'. Expected 'center_of_mass', 'monopolar', 'dipole', or 'grid_convolution'."
            )))
        }
    };

    let flat: Vec<f32> = coords.into_iter().flatten().collect();
    to_numpy(py, flat, &[n, 3])
}

/// Estimate 1D rigid vertical probe drift over time from spike timestamps and depths (`y_um`).
#[pyfunction]
#[pyo3(signature = (spike_samples, spike_depths_um, spike_amplitudes_uv=None, sample_rate_hz=30000.0, time_bin_sec=2.0, depth_bin_um=5.0, max_drift_um=100.0))]
pub fn estimate_rigid_drift<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    spike_depths_um: Vec<f32>,
    spike_amplitudes_uv: Option<Vec<f32>>,
    sample_rate_hz: f64,
    time_bin_sec: f64,
    depth_bin_um: f32,
    max_drift_um: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let fs = sample_rate_hz.max(1.0);
    let spike_times_sec: Vec<f64> = spike_samples.iter().map(|&s| (s as f64) / fs).collect();
    let total_duration_sec = spike_times_sec
        .iter()
        .copied()
        .fold(time_bin_sec.max(1.0), f64::max);
    let amps = spike_amplitudes_uv.unwrap_or_else(|| vec![100.0f32; spike_depths_um.len()]);
    let depth_min_um = spike_depths_um
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let depth_max_um = spike_depths_um
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let (d_min, d_max) = if depth_min_um.is_finite() && depth_max_um.is_finite() {
        (depth_min_um, (depth_max_um + depth_bin_um).max(depth_min_um + 20.0))
    } else {
        (0.0, 1000.0)
    };

    let est = rust_estimate_rigid_drift(
        &spike_times_sec,
        &spike_depths_um,
        &amps,
        total_duration_sec,
        time_bin_sec,
        d_min,
        d_max,
        depth_bin_um,
        max_drift_um,
    );
    let dict = PyDict::new(py);
    dict.set_item("time_bin_centers_sec", est.time_bin_centers_sec)?;
    dict.set_item(
        "drift_um",
        to_numpy(py, est.drift_um.clone(), &[est.drift_um.len()])?,
    )?;
    dict.set_item("num_time_bins", est.num_time_bins)?;
    dict.set_item("num_depth_bins", est.num_depth_bins)?;
    dict.set_item("depth_min_um", est.depth_min_um)?;
    dict.set_item("depth_bin_size_um", est.depth_bin_size_um)?;
    Ok(dict)
}

/// Estimate non-rigid depth-dependent vertical probe drift `[num_depth_blocks, num_time_bins]`.
#[pyfunction]
#[pyo3(signature = (spike_samples, spike_depths_um, spike_amplitudes_uv=None, sample_rate_hz=30000.0, num_depth_blocks=4, time_bin_sec=2.0, depth_bin_um=5.0, max_drift_um=100.0))]
pub fn estimate_nonrigid_drift<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    spike_depths_um: Vec<f32>,
    spike_amplitudes_uv: Option<Vec<f32>>,
    sample_rate_hz: f64,
    num_depth_blocks: usize,
    time_bin_sec: f64,
    depth_bin_um: f32,
    max_drift_um: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let fs = sample_rate_hz.max(1.0);
    let spike_times_sec: Vec<f64> = spike_samples.iter().map(|&s| (s as f64) / fs).collect();
    let total_duration_sec = spike_times_sec
        .iter()
        .copied()
        .fold(time_bin_sec.max(1.0), f64::max);
    let amps = spike_amplitudes_uv.unwrap_or_else(|| vec![100.0f32; spike_depths_um.len()]);
    let depth_min_um = spike_depths_um
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let depth_max_um = spike_depths_um
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let (d_min, d_max) = if depth_min_um.is_finite() && depth_max_um.is_finite() {
        (depth_min_um, (depth_max_um + depth_bin_um).max(depth_min_um + 40.0))
    } else {
        (0.0, 1000.0)
    };

    let est = rust_estimate_nonrigid_drift(
        &spike_times_sec,
        &spike_depths_um,
        &amps,
        total_duration_sec,
        time_bin_sec,
        d_min,
        d_max,
        depth_bin_um,
        max_drift_um,
        num_depth_blocks,
    );
    let dict = PyDict::new(py);
    dict.set_item("time_bin_centers_sec", est.time_bin_centers_sec)?;
    dict.set_item("block_centers_um", est.block_centers_um)?;
    dict.set_item(
        "drift_um",
        to_numpy(
            py,
            est.block_drift_um,
            &[est.num_depth_blocks, est.num_time_bins],
        )?,
    )?;
    dict.set_item("num_depth_blocks", est.num_depth_blocks)?;
    dict.set_item("num_time_bins", est.num_time_bins)?;
    Ok(dict)
}

/// Apply spatial Gaussian-Process Kriging interpolation to compensate for rigid vertical drift on traces `[C, S]`.
#[pyfunction]
#[pyo3(signature = (data, probe, time_bin_centers_sec, drift_um, sample_rate_hz=30000.0, start_sample=0, sigma_um=20.0, radius_um=60.0, channels=None))]
pub fn correct_drift_kriging<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    probe: PyRef<PyProbeLayout>,
    time_bin_centers_sec: Vec<f64>,
    drift_um: Vec<f32>,
    sample_rate_hz: f64,
    start_sample: u64,
    sigma_um: f32,
    radius_um: f32,
    channels: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = F32Array::new(&data)?;
    let (ch, samples) = input.channels_samples(channels)?;
    let layout = &probe.inner;
    let n_bins = time_bin_centers_sec.len().min(drift_um.len());
    let drift_est = DriftEstimate {
        time_bin_centers_sec: time_bin_centers_sec[..n_bins].to_vec(),
        drift_um: drift_um[..n_bins].to_vec(),
        activity_map: Vec::new(),
        num_time_bins: n_bins,
        num_depth_bins: 0,
        depth_min_um: 0.0,
        depth_bin_size_um: 1.0,
    };

    let corrected = correct_traces_drift_kriging(
        input.slice(),
        ch,
        samples,
        start_sample,
        layout,
        &drift_est,
        sample_rate_hz,
        sigma_um,
        radius_um,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    to_numpy(py, corrected, input.shape())
}
