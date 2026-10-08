//! Out-of-core streaming detection of a whole recording, in Rust and on the device: noise
//! calibration, halo windows through the pipeline, detection, deduplication, snippets and
//! per-channel templates. It is detection, not clustering: one unit per primary channel.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3::types::PyDict;

use dsp_synapse::{StreamingDetectionConfig, StreamingDetectionResult, StreamingDetector};

use super::detection::{parse_distance_rule, parse_polarity, PyDeduplicatedSpike};
use super::probe::PyProbeLayout;
use super::storage::PySortingOutput;
use crate::array::{runtime_error, to_numpy};
use crate::buffer::PyRecording;
use crate::pipeline::PyPipeline;
use crate::runtime::target;

/// Name recorded as the sorter of the `SortingOutput` this result converts to.
const SORTER_NAME: &str = "dsp-synapse streaming detection";

/// Result of `detect_recording`: the deduplicated spikes of a whole recording, the noise levels it
/// was calibrated with, and per-channel templates.
#[gen_stub_pyclass]
#[pyclass(name = "StreamingDetectionResult", skip_from_py_object)]
pub struct PyStreamingDetectionResult {
    inner: StreamingDetectionResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamingDetectionResult {
    /// Channels of the recording.
    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels
    }
    /// Samples of the recording.
    #[getter]
    fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }
    /// Sampling rate, Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate_hz
    }
    /// `(left, right)` halo samples each window was read with.
    #[getter]
    fn halos(&self) -> (u64, u64) {
        self.inner.halos
    }
    /// Noise σ of every channel (from calibration), in the recording's unit.
    #[getter]
    fn noise_sigmas(&self) -> Vec<f32> {
        self.inner.channel_sigmas_uv.clone()
    }
    /// Threshold crossings before deduplication.
    #[getter]
    fn total_crossings(&self) -> u64 {
        self.inner.total_raw_crossings
    }
    /// Spikes after deduplication.
    #[getter]
    fn total_spikes(&self) -> u64 {
        self.inner.total_dedup_spikes
    }
    /// Spikes per primary channel.
    #[getter]
    fn channel_spike_counts(&self) -> Vec<u64> {
        self.inner.channel_spike_counts.clone()
    }

    /// Every deduplicated spike, with recording sample times.
    fn spikes(&self) -> Vec<PyDeduplicatedSpike> {
        self.inner.spikes.iter().cloned().map(PyDeduplicatedSpike::from).collect()
    }

    /// Template of the spikes whose primary channel is `channel`.
    ///
    /// Parameters
    /// ----------
    /// channel : int
    ///
    /// Returns
    /// -------
    /// dict or None
    ///     `mean`, `std`, `se` (`[channels, samples]` float32), `channel_ids`, `count`; `None` when no spike
    ///     was detected there.
    fn template<'py>(&self, py: Python<'py>, channel: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(Some(t)) = self.inner.channel_templates.get(channel) else { return Ok(None) };
        let shape = [t.num_channels, t.num_samples];
        let d = PyDict::new(py);
        d.set_item("mean", to_numpy(py, t.mean.clone(), &shape)?)?;
        d.set_item("std", to_numpy(py, t.std.clone(), &shape)?)?;
        d.set_item("se", to_numpy(py, t.se.clone(), &shape)?)?;
        d.set_item("channel_ids", t.channel_ids.clone())?;
        d.set_item("count", t.count)?;
        Ok(Some(d))
    }

    /// The spikes as a `SortingOutput`, one unit per primary channel (detection, not sorting: a unit is
    /// every spike seen strongest on one channel).
    ///
    /// Parameters
    /// ----------
    /// probe : ProbeLayout, optional
    ///     Geometry stored with the units.
    #[pyo3(signature = (*, probe=None))]
    fn to_sorting_output(&self, probe: Option<PyRef<'_, PyProbeLayout>>) -> PySortingOutput {
        PySortingOutput::new(self.inner.to_sorting_output(SORTER_NAME, probe.map(|p| p.inner.clone())))
    }

    fn __repr__(&self) -> String {
        format!("StreamingDetectionResult(channels={}, spikes={}, halos={:?})", self.inner.channels, self.inner.total_dedup_spikes, self.inner.halos)
    }
}

/// Detects spikes in a whole recording: windows are read with halos, run through `pipeline` on the
/// device, thresholded, deduplicated across channels and cut into snippets, with bounded memory. Any
/// batch length gives the same spikes as processing the recording in one piece.
///
/// Parameters
/// ----------
/// recording : Recording
/// pipeline : Pipeline
///     Preprocessing (e.g. `[CommonAverageReference(), BandpassFilter(300.0, 5000.0)]`).
/// probe : ProbeLayout
/// threshold_factor : float, default 5.0
///     Threshold in multiples of each channel's noise σ (MAD, calibrated on the first chunks).
/// refractory_ms : float, default 1.0
///     Smallest interval between two crossings on a channel, ms.
/// polarity : {"negative", "positive", "both"}, default "negative"
/// distance_rule : {"locally-exclusive", "scipy"}, default "locally-exclusive"
///     See `detect_spikes`.
/// spatial_radius_um : float, default 150.0
///     Crossings within this radius (and the refractory window) are one spike.
/// k_neighbors : int, default 4
///     Channels of each snippet (the primary and its nearest).
/// pre_ms, post_ms : float, default 1.0, 2.0
///     Snippet window before and after each spike, ms.
/// batch_duration_sec : float, default 10.0
///     Length of each window, s (memory, not results).
/// calibration_duration_sec : float, default 5.0
///     Length of each noise-calibration chunk, s.
/// calibration_chunks : int, default 5
///     Chunks the noise σ is estimated from.
/// apply_sinc_shift : bool, default True
///     Realign snippets to the sub-sample trough (sinc interpolation).
/// start_sec, duration_sec : float, optional
///     Detect only in this window; default: the whole recording.
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// StreamingDetectionResult
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (recording, pipeline, probe, *, threshold_factor=None, refractory_ms=None, polarity=None, distance_rule=None, spatial_radius_um=None, k_neighbors=None, pre_ms=None, post_ms=None, batch_duration_sec=None, calibration_duration_sec=None, calibration_chunks=None, apply_sinc_shift=None, start_sec=None, duration_sec=None, runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn detect_recording(
    py: Python<'_>,
    recording: PyRef<'_, PyRecording>,
    pipeline: PyRef<'_, PyPipeline>,
    probe: PyRef<'_, PyProbeLayout>,
    threshold_factor: Option<f32>,
    refractory_ms: Option<f64>,
    polarity: Option<&str>,
    distance_rule: Option<&str>,
    spatial_radius_um: Option<f32>,
    k_neighbors: Option<usize>,
    pre_ms: Option<f64>,
    post_ms: Option<f64>,
    batch_duration_sec: Option<f64>,
    calibration_duration_sec: Option<f64>,
    calibration_chunks: Option<usize>,
    apply_sinc_shift: Option<bool>,
    start_sec: Option<f64>,
    duration_sec: Option<f64>,
    runtime: Option<&str>,
) -> PyResult<PyStreamingDetectionResult> {
    let d = StreamingDetectionConfig::default();
    let config = StreamingDetectionConfig {
        batch_duration_sec: batch_duration_sec.unwrap_or(d.batch_duration_sec),
        calibration_duration_sec: calibration_duration_sec.unwrap_or(d.calibration_duration_sec),
        calibration_chunks: calibration_chunks.unwrap_or(d.calibration_chunks),
        threshold_factor: threshold_factor.unwrap_or(d.threshold_factor),
        refractory_ms: refractory_ms.unwrap_or(d.refractory_ms),
        polarity: polarity.map(parse_polarity).transpose()?.unwrap_or(d.polarity),
        distance_rule: distance_rule.map(parse_distance_rule).transpose()?.unwrap_or(d.distance_rule),
        spatial_radius_um: spatial_radius_um.unwrap_or(d.spatial_radius_um),
        k_neighbors: k_neighbors.unwrap_or(d.k_neighbors),
        pre_ms: pre_ms.unwrap_or(d.pre_ms),
        post_ms: post_ms.unwrap_or(d.post_ms),
        apply_sinc_shift: apply_sinc_shift.unwrap_or(d.apply_sinc_shift),
    };
    let source = match (start_sec, duration_sec) {
        (None, None) => recording.inner.clone(),
        _ => recording.slice_time(start_sec.unwrap_or_default(), None, duration_sec, None)?.inner,
    };
    let (pipeline, layout) = (pipeline.pipeline(), probe.inner.clone());
    let target = target(runtime)?;
    let detector = StreamingDetector::new(config);
    let inner = py.detach(|| detector.run_with(target, source.as_ref(), &pipeline, &layout)).map_err(runtime_error)?;
    Ok(PyStreamingDetectionResult { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStreamingDetectionResult>()?;
    m.add_function(wrap_pyfunction!(detect_recording, m)?)?;
    Ok(())
}
