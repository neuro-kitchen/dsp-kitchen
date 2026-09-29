use pyo3::prelude::*;
use pyo3::types::PyDict;
use dsp_core::SensorLayout;
use dsp_synapse::probe::{neuropixels_1_0, neuropixels_2_0, tetrode, utah_array, find_k_nearest_neighbors};
use dsp_synapse::detection::{detect_spikes_multichannel, estimate_noise_std, SpikeEvent};

/// Neural Probe Layout representation for Python (SpikeInterface compatible).
#[pyclass(name = "ProbeLayout", skip_from_py_object)]
#[derive(Clone)]
pub struct PyProbeLayout {
    pub inner: SensorLayout,
}

#[pymethods]
impl PyProbeLayout {
    #[staticmethod]
    pub fn neuropixels_1_0() -> Self {
        Self {
            inner: neuropixels_1_0(),
        }
    }

    #[staticmethod]
    pub fn neuropixels_2_0() -> Self {
        Self {
            inner: neuropixels_2_0(),
        }
    }

    #[staticmethod]
    pub fn tetrode() -> Self {
        Self {
            inner: tetrode(),
        }
    }

    #[staticmethod]
    pub fn utah_array() -> Self {
        Self {
            inner: utah_array(),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn total_channels(&self) -> usize {
        self.inner.total_channels()
    }

    #[getter]
    pub fn active_channels(&self) -> usize {
        self.inner.active_channels()
    }

    pub fn contact_positions(&self) -> Vec<[f32; 3]> {
        self.inner
            .contacts
            .iter()
            .map(|c| [c.position.x_um, c.position.y_um, c.position.z_um])
            .collect()
    }

    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.channel_id).collect()
    }

    pub fn shank_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.shank_id).collect()
    }

    pub fn k_nearest_neighbors(&self, channel_id: usize, k: usize) -> Vec<usize> {
        find_k_nearest_neighbors(&self.inner, channel_id, k)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.inner.name)?;
        dict.set_item("ndim", 3)?;
        dict.set_item("total_channels", self.inner.total_channels())?;
        dict.set_item("contact_positions", self.contact_positions())?;
        dict.set_item("channel_ids", self.channel_ids())?;
        dict.set_item("shank_ids", self.shank_ids())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbeLayout(name='{}', channels={}, active={})",
            self.inner.name,
            self.inner.total_channels(),
            self.inner.active_channels()
        )
    }
}

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
