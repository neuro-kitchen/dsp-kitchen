//! PyO3 bindings for out-of-core streaming spike sorting (`StreamingSpikeRunner`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::to_numpy;

use dsp_synapse::streaming::{StreamingSortConfig, StreamingSortResult, StreamingSpikeRunner};

use crate::buffer::PyNwbZarrRecording;
use crate::pipeline::PyPipeline;
use super::detection::PyDeduplicatedSpike;
use super::probe::PyProbeLayout;

/// Result of out-of-core streaming spike sorting across a `NwbZarrRecording`.
#[pyclass(name = "StreamingSortResult", skip_from_py_object)]
pub struct PyStreamingSortResult {
    inner: StreamingSortResult,
}

#[pymethods]
impl PyStreamingSortResult {
    #[getter]
    pub fn channels(&self) -> usize {
        self.inner.channels
    }

    #[getter]
    pub fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    #[getter]
    pub fn halos(&self) -> (u64, u64) {
        self.inner.halos
    }

    #[getter]
    pub fn channel_sigmas_uv(&self) -> Vec<f32> {
        self.inner.channel_sigmas_uv.clone()
    }

    #[getter]
    pub fn total_raw_crossings(&self) -> u64 {
        self.inner.total_raw_crossings
    }

    #[getter]
    pub fn total_dedup_spikes(&self) -> u64 {
        self.inner.total_dedup_spikes
    }

    #[getter]
    pub fn channel_spike_counts(&self) -> Vec<u64> {
        self.inner.channel_spike_counts.clone()
    }

    /// Returns the list of all deduplicated spikes (with global `sample_index`).
    pub fn spikes(&self) -> Vec<PyDeduplicatedSpike> {
        self.inner
            .spikes
            .iter()
            .map(|d| PyDeduplicatedSpike {
                primary_channel: d.primary_channel,
                sample_index: d.sample_index,
                peak_amplitude_uv: d.peak_amplitude_uv,
                participating_channels: d.participating_channels.clone(),
            })
            .collect()
    }

    /// Returns the online Welford-accumulated template (`{"mean": ndarray, "std": ndarray, ...}`)
    /// for `channel`, or `None` if no spikes were detected on `channel`.
    pub fn template<'py>(
        &self,
        py: Python<'py>,
        channel: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(Some(t)) = self.inner.channel_templates.get(channel) else {
            return Ok(None);
        };

        let (k, s) = (t.num_channels, t.num_samples);
        let mean_arr = to_numpy(py, t.mean.clone(), &[k, s])?;
        let std_arr = to_numpy(py, t.std.clone(), &[k, s])?;

        let d = PyDict::new(py);
        d.set_item("mean", mean_arr)?;
        d.set_item("std", std_arr)?;
        d.set_item("num_channels", k)?;
        d.set_item("num_samples", s)?;
        d.set_item(
            "count",
            self.inner.channel_spike_counts.get(channel).copied().unwrap_or(0),
        )?;
        Ok(Some(d))
    }

    /// Returns a list of length `channels` containing the template dict (or `None`) for every channel.
    pub fn all_templates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Option<Bound<'py, PyDict>>>> {
        let mut out = Vec::with_capacity(self.inner.channels);
        for ch in 0..self.inner.channels {
            out.push(self.template(py, ch)?);
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        let active = self
            .inner
            .channel_templates
            .iter()
            .filter(|t| t.is_some())
            .count();
        format!(
            "StreamingSortResult(channels={}, active_channels={}, total_dedup_spikes={}, halos={:?})",
            self.inner.channels, active, self.inner.total_dedup_spikes, self.inner.halos
        )
    }
}

/// Runs out-of-core threshold spike sorting across the full `recording` (or a lazy `[start_sec, start_sec + duration_sec]` window)
/// in constant memory, automatically computing boundary halos from `sample_rate` and `pipeline` filter cutoffs,
/// prefetching Zarr chunks in a background thread, and accumulating Welford templates per channel.
#[pyfunction]
#[pyo3(signature = (
    recording,
    pipeline,
    probe,
    threshold_factor=5.0,
    refractory_ms=1.0,
    spatial_radius_um=150.0,
    k_neighbors=4,
    pre_ms=1.0,
    post_ms=2.0,
    batch_duration_sec=10.0,
    calibration_duration_sec=5.0,
    calibration_chunks=5,
    apply_sinc_shift=true,
    start_sec=None,
    duration_sec=None
))]
#[allow(clippy::too_many_arguments)]
pub fn sort_recording(
    py: Python<'_>,
    recording: PyRef<'_, PyNwbZarrRecording>,
    pipeline: PyRef<'_, PyPipeline>,
    probe: PyRef<'_, PyProbeLayout>,
    threshold_factor: f32,
    refractory_ms: f64,
    spatial_radius_um: f32,
    k_neighbors: usize,
    pre_ms: f64,
    post_ms: f64,
    batch_duration_sec: f64,
    calibration_duration_sec: f64,
    calibration_chunks: usize,
    apply_sinc_shift: bool,
    start_sec: Option<f64>,
    duration_sec: Option<f64>,
) -> PyResult<PyStreamingSortResult> {
    let source: std::sync::Arc<dyn dsp_core::RecordingSource> =
        if start_sec.is_some() || duration_sec.is_some() {
            let sliced = recording.slice_time(start_sec.unwrap_or(0.0), None, duration_sec, None)?;
            sliced.inner
        } else {
            recording.inner.clone()
        };
    let rust_pipeline = pipeline.to_rust_pipeline();
    let rust_probe = probe.inner.clone();

    let config = StreamingSortConfig {
        batch_duration_sec,
        calibration_duration_sec,
        calibration_chunks,
        threshold_factor,
        refractory_ms,
        spatial_radius_um,
        k_neighbors,
        pre_ms,
        post_ms,
        apply_sinc_shift,
    };

    let runner = StreamingSpikeRunner::new(config);
    let result = py
        .detach(|| runner.run(source.as_ref(), &rust_pipeline, &rust_probe))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Streaming sort error: {e}")))?;

    Ok(PyStreamingSortResult { inner: result })
}
