use pyo3::prelude::*;
use pyo3::types::PyBytes;
use dsp_base::linalg::PcaModel;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;

#[pyclass(name = "PCA")]
pub struct PyPca {
    inner: Option<PcaModel>,
    n_components: usize,
}

#[pymethods]
impl PyPca {
    #[new]
    #[pyo3(signature = (n_components=3))]
    pub fn new(n_components: usize) -> Self {
        Self {
            inner: None,
            n_components,
        }
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Fits the PCA model to the input array of shape [channels, samples].
    #[pyo3(signature = (data, channels=None))]
    pub fn fit<'py>(&mut self, py: Python<'py>, data: Bound<'py, PyAny>, channels: Option<usize>) -> PyResult<()> {
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

        self.inner = Some(PcaModel::fit(float_slice, ch, samples, self.n_components));
        Ok(())
    }

    /// Transforms data [channels, samples] into PCA space [n_components, samples].
    #[pyo3(signature = (data, channels=None, use_gpu=false))]
    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        channels: Option<usize>,
        use_gpu: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("PCA model must be fitted before calling transform")
        })?;

        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let (ch, samples) = match shape.len() {
            1 => {
                let c = channels.unwrap_or(model.num_channels);
                (c, shape[0] / c)
            }
            2 => (shape[0], shape[1]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Data must be a 1D or 2D float32 array [channels, samples]",
                ));
            }
        };

        if ch != model.num_channels {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Channel count mismatch: model expects {}, got {}",
                model.num_channels, ch
            )));
        }

        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;

        let out_bytes: Vec<u8> = if use_gpu {
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);
            let in_handle = client.create_from_slice(raw_bytes);
            let out_handle = client.empty(model.num_components * samples * std::mem::size_of::<f32>());
            model.project_gpu::<WgpuRuntime>(&client, &in_handle, &out_handle, ch, samples, false);
            client.read_one_unchecked(out_handle).to_vec()
        } else {
            let float_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    raw_bytes.as_ptr() as *const f32,
                    raw_bytes.len() / std::mem::size_of::<f32>(),
                )
            };
            let projected = model.project_cpu(float_slice, ch, samples);
            let projected_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    projected.as_ptr() as *const u8,
                    projected.len() * std::mem::size_of::<f32>(),
                )
            };
            projected_bytes.to_vec()
        };

        let out_py_bytes = PyBytes::new(py, &out_bytes);
        let flat_arr = np.call_method1("frombuffer", (out_py_bytes, "float32"))?;
        let reshaped = flat_arr.call_method1("reshape", ((model.num_components, samples),))?;
        Ok(reshaped)
    }

    #[getter]
    pub fn explained_variance_ratio<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Some(model) => {
                let np = py.import("numpy")?;
                let arr = np.call_method1("array", (model.explained_variance_ratio.clone(),))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    #[getter]
    pub fn explained_variance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Some(model) => {
                let np = py.import("numpy")?;
                let arr = np.call_method1("array", (model.explained_variance.clone(),))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("PCA(n_components={})", self.n_components)
    }
}
