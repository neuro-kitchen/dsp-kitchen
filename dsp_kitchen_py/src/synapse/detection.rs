use pyo3::prelude::*;
use dsp_synapse::detection::{
    detect_spikes_multichannel, estimate_noise_std, deduplicate_spikes_spatial,
    SpikeEvent, DeduplicatedSpike,
};
use super::probe::PyProbeLayout;
use crate::array::F32Array;

/// Detected spike event.
#[pyclass(name = "SpikeEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpikeEvent {
    #[pyo3(get)]
    pub channel_id: usize,
    #[pyo3(get)]
    pub sample_index: u64,
    #[pyo3(get)]
    pub peak_amplitude_uv: f32,
}

#[pymethods]
impl PySpikeEvent {
    #[new]
    pub fn new(channel_id: usize, sample_index: u64, peak_amplitude_uv: f32) -> Self {
        Self {
            channel_id,
            sample_index,
            peak_amplitude_uv,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpikeEvent(channel={}, sample={}, peak={:.2}uV)",
            self.channel_id, self.sample_index, self.peak_amplitude_uv
        )
    }
}

/// Spatially deduplicated action potential event.
#[pyclass(name = "DeduplicatedSpike", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDeduplicatedSpike {
    #[pyo3(get)]
    pub primary_channel: usize,
    #[pyo3(get)]
    pub sample_index: u64,
    #[pyo3(get)]
    pub peak_amplitude_uv: f32,
    #[pyo3(get)]
    pub participating_channels: Vec<usize>,
}

#[pymethods]
impl PyDeduplicatedSpike {
    fn __repr__(&self) -> String {
        format!(
            "DeduplicatedSpike(primary={}, sample={}, peak={:.2}uV, neighbors={:?})",
            self.primary_channel, self.sample_index, self.peak_amplitude_uv, self.participating_channels
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, channels=None, threshold_factor=5.0, refractory_samples=30))]
pub fn detect_spikes<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    channels: Option<usize>,
    threshold_factor: f32,
    refractory_samples: usize,
) -> PyResult<Vec<PySpikeEvent>> {
    let input = F32Array::new(&data)?;
    let (ch, samples) = input.channels_samples(channels)?;
    let x = input.slice();
    let spikes = py.detach(|| detect_spikes_multichannel(x, ch, samples, threshold_factor, refractory_samples));
    let py_spikes = spikes
        .into_iter()
        .map(|s: SpikeEvent| PySpikeEvent {
            channel_id: s.channel_id,
            sample_index: s.sample_index,
            peak_amplitude_uv: s.peak_amplitude_uv,
        })
        .collect();

    Ok(py_spikes)
}

#[pyfunction]
#[pyo3(signature = (spikes, probe, radius_um=50.0, window_samples=15))]
pub fn deduplicate_spikes(
    py: Python<'_>,
    spikes: Vec<PyRef<PySpikeEvent>>,
    probe: PyRef<PyProbeLayout>,
    radius_um: f32,
    window_samples: u64,
) -> Vec<PyDeduplicatedSpike> {
    let rust_spikes: Vec<SpikeEvent> = spikes
        .iter()
        .map(|s| SpikeEvent {
            channel_id: s.channel_id,
            sample_index: s.sample_index,
            peak_amplitude_uv: s.peak_amplitude_uv,
        })
        .collect();

    let layout = &probe.inner;
    let deduped = py.detach(|| deduplicate_spikes_spatial(&rust_spikes, layout, radius_um, window_samples));

    deduped
        .into_iter()
        .map(|d: DeduplicatedSpike| PyDeduplicatedSpike {
            primary_channel: d.primary_channel,
            sample_index: d.sample_index,
            peak_amplitude_uv: d.peak_amplitude_uv,
            participating_channels: d.participating_channels,
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn estimate_noise<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<f32> {
    let input = F32Array::new(&data)?;
    let x = input.slice();
    Ok(py.detach(|| estimate_noise_std(x)))
}

