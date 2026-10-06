//! Spike localization, probe drift estimation and kriging drift correction. Binning, drift
//! limits and kriging scales depend on the probe and recording: they have no defaults.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{to_numpy, value_error, F32Array};
use dsp_synapse::core::PeakLocalizer;
use dsp_synapse::extraction::SnippetBatch;
use dsp_synapse::spatial::{
    correct_traces_drift_kriging, estimate_nonrigid_drift as nonrigid, estimate_rigid_drift as rigid, CenterOfMassLocalizer, DipoleLocalizer, DriftEstimate,
    GridConvolutionLocalizer, MonopolarTriangulator,
};

use super::extraction::PyWaveformSnippet;
use super::probe::PyProbeLayout;

/// Coordinates per position (x, y, z).
const COORDS: usize = 3;
/// Weight of each spike when no amplitudes are given (every spike counts once).
const UNIT_WEIGHT: f32 = 1.0;

/// `[spikes, 3]` positions `(x, y, z)` in µm of the snippets' sources: `"center_of_mass"`,
/// `"monopolar"`, `"dipole"` or `"grid_convolution"`; `max_iterations` overrides the iterative
/// methods' default.
#[pyfunction]
#[pyo3(signature = (snippets, probe, method="center_of_mass", *, max_iterations=None))]
pub fn localize_spikes<'py>(py: Python<'py>, snippets: Vec<PyRef<'py, PyWaveformSnippet>>, probe: PyRef<'py, PyProbeLayout>, method: &str, max_iterations: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
    let Some(first) = snippets.first() else { return to_numpy(py, Vec::new(), &[0, COORDS]) };
    let peak_index = first.peak_index;
    let all: Vec<_> = snippets.iter().map(|s| s.inner.clone()).collect();
    let batch = SnippetBatch::from_snippets(&all, peak_index).ok_or_else(|| PyValueError::new_err("snippets of different shapes"))?;
    let layout = &probe.inner;
    let localizer: Box<dyn PeakLocalizer> = match method {
        "center_of_mass" => Box::new(CenterOfMassLocalizer::default()),
        "monopolar" => {
            let mut l = MonopolarTriangulator::default();
            l.max_iterations = max_iterations.unwrap_or(l.max_iterations);
            Box::new(l)
        }
        "dipole" => {
            let mut l = DipoleLocalizer::default();
            l.max_iterations = max_iterations.unwrap_or(l.max_iterations);
            Box::new(l)
        }
        "grid_convolution" => Box::new(GridConvolutionLocalizer::default()),
        other => return Err(PyValueError::new_err(format!("method must be 'center_of_mass', 'monopolar', 'dipole' or 'grid_convolution', got '{other}'"))),
    };
    let positions = py.detach(|| localizer.localize(&batch, layout)).map_err(value_error)?;
    let n = positions.len();
    to_numpy(py, positions.into_iter().flatten().collect(), &[n, COORDS])
}

/// Inputs shared by the drift estimators: spike times (s), amplitudes, depth range and duration.
struct DriftInputs {
    times_sec: Vec<f64>,
    amplitudes: Vec<f32>,
    depth_range: (f32, f32),
    duration_sec: f64,
}

fn drift_inputs(spike_samples: &[u64], depths_um: &[f32], amplitudes: Option<Vec<f32>>, fs: f64, depth_range_um: Option<(f32, f32)>, depth_bin_um: f32, duration_sec: Option<f64>) -> PyResult<DriftInputs> {
    if spike_samples.len() != depths_um.len() || amplitudes.as_ref().is_some_and(|a| a.len() != depths_um.len()) {
        return Err(PyValueError::new_err("spike_samples, spike_depths_um and spike_amplitudes must have the same length"));
    }
    if depths_um.is_empty() {
        return Err(PyValueError::new_err("no spikes to estimate drift from"));
    }
    let times_sec: Vec<f64> = spike_samples.iter().map(|&s| s as f64 / fs).collect();
    let depth_range = depth_range_um.unwrap_or_else(|| {
        let (lo, hi) = depths_um.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &d| (a.min(d), b.max(d)));
        (lo, hi + depth_bin_um)
    });
    let duration_sec = duration_sec.unwrap_or_else(|| times_sec.iter().copied().fold(0.0, f64::max));
    Ok(DriftInputs { amplitudes: amplitudes.unwrap_or_else(|| vec![UNIT_WEIGHT; depths_um.len()]), times_sec, depth_range, duration_sec })
}

/// Rigid vertical drift over time, from spike times and depths (`y`, µm): activity profiles per
/// time bin registered by cross-correlation. Amplitudes weight spikes (default: equally); the
/// depth range defaults to the spikes' span, the duration to the last spike.
#[pyfunction]
#[pyo3(signature = (spike_samples, spike_depths_um, *, fs, time_bin_sec, depth_bin_um, max_drift_um, spike_amplitudes=None, depth_range_um=None, duration_sec=None))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_rigid_drift<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    spike_depths_um: Vec<f32>,
    fs: f64,
    time_bin_sec: f64,
    depth_bin_um: f32,
    max_drift_um: f32,
    spike_amplitudes: Option<Vec<f32>>,
    depth_range_um: Option<(f32, f32)>,
    duration_sec: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = drift_inputs(&spike_samples, &spike_depths_um, spike_amplitudes, fs, depth_range_um, depth_bin_um, duration_sec)?;
    let est = py.detach(|| rigid(&d.times_sec, &spike_depths_um, &d.amplitudes, d.duration_sec, time_bin_sec, d.depth_range.0, d.depth_range.1, depth_bin_um, max_drift_um));
    let dict = PyDict::new(py);
    dict.set_item("time_bin_centers_sec", est.time_bin_centers_sec)?;
    dict.set_item("drift_um", to_numpy(py, est.drift_um.clone(), &[est.drift_um.len()])?)?;
    dict.set_item("depth_min_um", est.depth_min_um)?;
    dict.set_item("depth_bin_size_um", est.depth_bin_size_um)?;
    Ok(dict)
}

/// Depth-dependent drift: rigid drift in `num_depth_blocks` blocks (`drift_um` is
/// `[blocks, time bins]`).
#[pyfunction]
#[pyo3(signature = (spike_samples, spike_depths_um, *, fs, num_depth_blocks, time_bin_sec, depth_bin_um, max_drift_um, spike_amplitudes=None, depth_range_um=None, duration_sec=None))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_nonrigid_drift<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    spike_depths_um: Vec<f32>,
    fs: f64,
    num_depth_blocks: usize,
    time_bin_sec: f64,
    depth_bin_um: f32,
    max_drift_um: f32,
    spike_amplitudes: Option<Vec<f32>>,
    depth_range_um: Option<(f32, f32)>,
    duration_sec: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = drift_inputs(&spike_samples, &spike_depths_um, spike_amplitudes, fs, depth_range_um, depth_bin_um, duration_sec)?;
    let est = py.detach(|| nonrigid(&d.times_sec, &spike_depths_um, &d.amplitudes, d.duration_sec, time_bin_sec, d.depth_range.0, d.depth_range.1, depth_bin_um, max_drift_um, num_depth_blocks));
    let dict = PyDict::new(py);
    dict.set_item("time_bin_centers_sec", est.time_bin_centers_sec)?;
    dict.set_item("block_centers_um", est.block_centers_um)?;
    dict.set_item("drift_um", to_numpy(py, est.block_drift_um, &[est.num_depth_blocks, est.num_time_bins])?)?;
    Ok(dict)
}

/// `data` (`[channels, samples]`, first sample `start_sample`) resampled by Gaussian-process
/// kriging onto drift-corrected positions, for rigid drift `drift_um` at `time_bin_centers_sec`.
#[pyfunction]
#[pyo3(signature = (data, probe, time_bin_centers_sec, drift_um, *, fs, sigma_um, radius_um, start_sample=0))]
#[allow(clippy::too_many_arguments)]
pub fn correct_drift_kriging<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    probe: PyRef<'py, PyProbeLayout>,
    time_bin_centers_sec: Vec<f64>,
    drift_um: Vec<f32>,
    fs: f64,
    sigma_um: f32,
    radius_um: f32,
    start_sample: u64,
) -> PyResult<Bound<'py, PyAny>> {
    if time_bin_centers_sec.len() != drift_um.len() {
        return Err(PyValueError::new_err("time_bin_centers_sec and drift_um must have the same length"));
    }
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let n = drift_um.len();
    let drift = DriftEstimate { time_bin_centers_sec, drift_um, activity_map: Vec::new(), num_time_bins: n, num_depth_bins: 0, depth_min_um: 0.0, depth_bin_size_um: 0.0 };
    let (x, layout) = (input.slice(), &probe.inner);
    let corrected = py.detach(|| correct_traces_drift_kriging(x, channels, samples, start_sample, layout, &drift, fs, sigma_um, radius_um)).map_err(value_error)?;
    to_numpy(py, corrected, input.shape())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(localize_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rigid_drift, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_nonrigid_drift, m)?)?;
    m.add_function(wrap_pyfunction!(correct_drift_kriging, m)?)?;
    Ok(())
}
