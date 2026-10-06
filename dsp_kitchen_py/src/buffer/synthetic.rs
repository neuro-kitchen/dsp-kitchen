//! `dsp_kitchen.io.SyntheticRecording`: dsp-io's procedural recording (noise, line hum, drifting
//! units) with its ground-truth spike times, for tests and examples.

use std::path::PathBuf;
use std::sync::Arc;

use dsp_core::RecordingSource;
use dsp_io::{SyntheticParams, SyntheticRecording};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::recording::PyRecording;
use crate::array::value_error;

#[pyclass(name = "SyntheticRecording", skip_from_py_object)]
pub struct PySyntheticRecording {
    inner: Arc<SyntheticRecording>,
}

#[pymethods]
impl PySyntheticRecording {
    /// `channels` × `duration_sec` at `fs` Hz; unset options take dsp-io's `SyntheticParams`
    /// defaults (noise and line hum in µV, `units` spiking units, drift, seed).
    #[new]
    #[pyo3(signature = (channels, fs, duration_sec, *, noise=None, line_noise=None, units=None, drift_channels=None, drift_period_sec=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        channels: usize,
        fs: f64,
        duration_sec: f64,
        noise: Option<f32>,
        line_noise: Option<f32>,
        units: Option<usize>,
        drift_channels: Option<f32>,
        drift_period_sec: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let d = SyntheticParams::default();
        let params = SyntheticParams {
            channels,
            sample_rate_hz: fs,
            duration_sec,
            noise_uv: noise.unwrap_or(d.noise_uv),
            line_noise_uv: line_noise.unwrap_or(d.line_noise_uv),
            units: units.unwrap_or(d.units),
            drift_channels: drift_channels.unwrap_or(d.drift_channels),
            drift_period_sec: drift_period_sec.unwrap_or(d.drift_period_sec),
            seed: seed.unwrap_or(d.seed),
        };
        Ok(Self { inner: Arc::new(SyntheticRecording::new(params).map_err(value_error)?) })
    }

    #[getter]
    fn unit_count(&self) -> usize {
        self.inner.unit_count()
    }

    /// Ground-truth trough samples of `unit` in `[start, end)` (default: the whole recording).
    #[pyo3(signature = (unit, start=0, end=None))]
    fn spike_times(&self, unit: usize, start: u64, end: Option<u64>) -> PyResult<Vec<u64>> {
        if unit >= self.inner.unit_count() {
            return Err(PyValueError::new_err(format!("unit {unit} of {}", self.inner.unit_count())));
        }
        Ok(self.inner.spike_times(unit, start..end.unwrap_or(self.inner.info().samples)))
    }

    /// The samples as a `Recording` (computed on demand, nothing stored).
    fn recording(&self) -> PyRecording {
        let source: Arc<dyn RecordingSource> = self.inner.clone();
        PyRecording::from_source(PathBuf::new(), source)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySyntheticRecording>()
}
