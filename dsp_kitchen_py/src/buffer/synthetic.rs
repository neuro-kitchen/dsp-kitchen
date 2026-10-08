//! `dsp_kitchen.io.SyntheticRecording`: dsp-io's procedural recording (noise, line hum, drifting
//! units) with its ground-truth spike times, for tests and examples.

use std::path::PathBuf;
use std::sync::Arc;

use dsp_core::RecordingSource;
use dsp_io::{SyntheticParams, SyntheticRecording};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::recording::PyRecording;
use crate::array::value_error;

/// A procedural recording with ground truth: Gaussian noise, 60 Hz hum and spiking units that drift
/// across channels, computed on demand (nothing is stored), with every unit's true spike times.
///
/// For tests and examples: sort `recording()` and compare with `spike_times(unit)`.
///
/// Parameters
/// ----------
/// channels : int
///     Channels, on a line (unit positions are in channel units).
/// fs : float
///     Sampling rate, Hz.
/// duration_sec : float
///     Length, s.
/// noise : float, default 10.0
///     Noise RMS, µV.
/// line_noise : float, default 20.0
///     60 Hz hum amplitude, µV.
/// units : int, default 8
///     Spiking units (0: no spikes).
/// drift_channels : float, default 2.0
///     Peak displacement of the units over time, in channels.
/// drift_period_sec : float, default 600.0
///     Period of the drift, s.
/// seed : int
///     Seed of everything random: the same seed gives the same recording.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.io import SyntheticRecording
/// >>> truth = SyntheticRecording(32, 30000.0, 60.0, units=5)
/// >>> rec = truth.recording()
/// >>> truth.spike_times(0)[:5]
#[gen_stub_pyclass]
#[pyclass(name = "SyntheticRecording", skip_from_py_object)]
pub struct PySyntheticRecording {
    inner: Arc<SyntheticRecording>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySyntheticRecording {
    /// See the class docs; defaults come from dsp-io's `SyntheticParams`.
    #[new]
    #[pyo3(signature = (channels, fs, duration_sec, *, noise = SyntheticParams::default().noise_uv, line_noise = SyntheticParams::default().line_noise_uv, units = SyntheticParams::default().units, drift_channels = SyntheticParams::default().drift_channels, drift_period_sec = SyntheticParams::default().drift_period_sec, seed = SyntheticParams::default().seed))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        channels: usize,
        fs: f64,
        duration_sec: f64,
        noise: f32,
        line_noise: f32,
        units: usize,
        drift_channels: f32,
        drift_period_sec: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let params = SyntheticParams {
            channels,
            sample_rate_hz: fs,
            duration_sec,
            noise_uv: noise,
            line_noise_uv: line_noise,
            units,
            drift_channels,
            drift_period_sec,
            seed,
        };
        Ok(Self { inner: Arc::new(SyntheticRecording::new(params).map_err(value_error)?) })
    }

    /// Number of spiking units.
    #[getter]
    fn unit_count(&self) -> usize {
        self.inner.unit_count()
    }

    /// True spike times of a unit: samples of the waveform troughs in `[start, end)`.
    ///
    /// Parameters
    /// ----------
    /// unit : int
    ///     Unit index, `0 ≤ unit < unit_count`.
    /// start, end : int
    ///     Sample range; default: the whole recording.
    #[pyo3(signature = (unit, start=0, end=None))]
    fn spike_times(&self, unit: usize, start: u64, end: Option<u64>) -> PyResult<Vec<u64>> {
        if unit >= self.inner.unit_count() {
            return Err(PyValueError::new_err(format!("unit {unit} of {}", self.inner.unit_count())));
        }
        Ok(self.inner.spike_times(unit, start..end.unwrap_or(self.inner.info().samples)))
    }

    /// The signal as a `Recording` (samples computed on demand, nothing stored), in µV.
    fn recording(&self) -> PyRecording {
        let source: Arc<dyn RecordingSource> = self.inner.clone();
        PyRecording::from_source(PathBuf::new(), source)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySyntheticRecording>()
}
