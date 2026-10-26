//! Spike localization, probe drift estimation and kriging drift correction. Binning, drift
//! limits and kriging scales depend on the probe and recording: they have no defaults.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyfunction};
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

/// Position of each spike's source from its snippet.
///
/// Parameters
/// ----------
/// snippets : list of WaveformSnippet
/// probe : ProbeLayout
/// method : {"center_of_mass", "monopolar", "dipole", "grid_convolution"}, default "center_of_mass"
///     `"center_of_mass"`: amplitude-weighted mean of the channel positions (fast, biased towards the
///     probe); `"monopolar"` / `"dipole"`: least-squares fit of a point source / dipole (iterative);
///     `"grid_convolution"`: a soft-argmax over a 3-D grid of monopole footprints around the primary
///     channel, by cosine similarity with the spike's peak-to-peak amplitudes (inspired by
///     SpikeInterface's, results differ).
/// max_iterations : int, optional
///     Iterations of the fitted methods; default: each method's own.
///
/// Returns
/// -------
/// numpy.ndarray
///     `[spikes, 3]` float32 `(x, y, z)`, µm.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (snippets, probe, method="center_of_mass", *, max_iterations=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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

/// Vertical drift of the probe over time, the same at every depth: per time bin, the depth profile of
/// spike activity is registered against a reference by cross-correlation.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// spike_depths_um : list of float
///     Spike depths (`y`), µm (e.g. from `localize_spikes`).
/// fs : float
///     Sampling rate, Hz.
/// time_bin_sec : float
///     Time bin, s.
/// depth_bin_um : float
///     Depth bin of the activity profiles, µm.
/// max_drift_um : float
///     Largest shift searched, µm.
/// spike_amplitudes : list of float, optional
///     Weights of the spikes; default: every spike counts once.
/// depth_range_um : tuple of float, optional
///     `(min, max)` depth considered; default: the spikes' span.
/// duration_sec : float, optional
///     Length of the recording, s; default: up to the last spike.
///
/// Returns
/// -------
/// dict
///     `time_bin_centers_sec`, `drift_um` (`[time bins]` float32), `depth_min_um`, `depth_bin_size_um`.
#[gen_stub_pyfunction]
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

/// Depth-dependent drift: rigid drift estimated separately in `num_depth_blocks` depth blocks.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// spike_depths_um : list of float
///     Spike depths (`y`), µm.
/// fs : float
///     Sampling rate, Hz.
/// num_depth_blocks : int
///     Depth blocks, each with its own rigid drift.
/// time_bin_sec : float
///     Time bin, s.
/// depth_bin_um : float
///     Depth bin of the activity profiles, µm.
/// max_drift_um : float
///     Largest shift searched, µm.
/// spike_amplitudes : list of float, optional
///     Weights of the spikes; default: every spike counts once.
/// depth_range_um : tuple of float, optional
///     `(min, max)` depth considered; default: the spikes' span.
/// duration_sec : float, optional
///     Length of the recording, s; default: up to the last spike.
///
/// Returns
/// -------
/// dict
///     `time_bin_centers_sec`, `block_centers_um`, `drift_um` (`[blocks, time bins]` float32).
#[gen_stub_pyfunction]
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

/// Corrects a signal for drift: every channel is resampled at its drift-corrected position by
/// Gaussian-process (kriging) interpolation from its neighbours.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]`, converted to float32.
/// probe : ProbeLayout
/// time_bin_centers_sec : list of float
///     Times of the drift estimate, s (from `estimate_rigid_drift`).
/// drift_um : list of float
///     Rigid drift at those times, µm.
/// fs : float
///     Sampling rate, Hz.
/// sigma_um : float
///     Length scale of the Gaussian kernel, µm.
/// radius_um : float
///     Neighbours used for each channel, µm.
/// start_sample : int, default 0
///     Recording sample of `data`'s first column (to find each sample's drift).
///
/// Returns
/// -------
/// numpy.ndarray
///     `[channels, samples]` float32.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, probe, time_bin_centers_sec, drift_um, *, fs, sigma_um, radius_um, start_sample=0))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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
