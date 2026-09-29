use pyo3::prelude::*;
use dsp_synapse::detection::{detect_spikes_multichannel, estimate_noise_std, SpikeEvent};

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
    fn __repr__(&self) -> String {
        format!(
            "SpikeEvent(channel={}, sample={}, peak={:.2}uV)",
            self.channel_id, self.sample_index, self.peak_amplitude_uv
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
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let (ch, samples) = match shape.len() {
        1 => {
            let c = channels.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("channels must be specified for 1D arrays")
            })?;
            (c, shape[0] / c)
        }
        2 => (shape[0], shape[1]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data must be a 1D or 2D float32 array [channels, samples]",
            ));
        }
    };

    let py_bytes = arr.call_method0("tobytes")?;
    let raw_bytes: &[u8] = py_bytes.extract()?;
    let float_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            raw_bytes.as_ptr() as *const f32,
            raw_bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    let spikes = detect_spikes_multichannel(float_slice, ch, samples, threshold_factor, refractory_samples);
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
#[pyo3(signature = (data))]
pub fn estimate_noise<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<f32> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
    let py_bytes = arr.call_method0("tobytes")?;
    let raw_bytes: &[u8] = py_bytes.extract()?;
    let float_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            raw_bytes.as_ptr() as *const f32,
            raw_bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    Ok(estimate_noise_std(float_slice))
}
