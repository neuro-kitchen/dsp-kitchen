use pyo3::prelude::*;
use dsp_base::linalg::PcaModel;
use cubecl::prelude::ComputeClient;
use cubecl::{CubeElement, Runtime};
use dsp_core::compute::ComputeTask;

use crate::pipeline::compute_target;

use crate::array::{to_numpy, F32Array};

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
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        let (x, k) = (input.slice(), self.n_components);
        self.inner = Some(py.detach(|| PcaModel::fit(x, ch, samples, k)));
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
        let input = F32Array::new(&data)?;
        let flat = input.ndim() == 1;
        let (ch, samples) = input.channels_samples(channels.or(flat.then_some(model.num_channels)))?;
        if ch != model.num_channels {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Channel count mismatch: model expects {}, got {}",
                model.num_channels, ch
            )));
        }
        struct Project<'a> {
            model: &'a PcaModel,
            x: &'a [f32],
            channels: usize,
            samples: usize,
        }
        impl ComputeTask for Project<'_> {
            type Output = Vec<f32>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Vec<f32> {
                let in_handle = client.create_from_slice(f32::as_bytes(self.x));
                let out_handle = client.empty(self.model.num_components * self.samples * std::mem::size_of::<f32>());
                self.model.project_gpu::<R>(&client, &in_handle, &out_handle, self.channels, self.samples);
                f32::from_bytes(&client.read_one_unchecked(out_handle)).to_vec()
            }
        }
        let x = input.slice();
        let target = compute_target()?;
        let projected = py.detach(|| {
            if use_gpu {
                target.run(Project { model, x, channels: ch, samples })
            } else {
                Ok(model.project_cpu(x, ch, samples))
            }
        });
        let projected = projected.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        to_numpy(py, projected, &[model.num_components, samples])
    }

    #[getter]
    pub fn explained_variance_ratio<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Some(model) => {
                let v = model.explained_variance_ratio.clone();
                let n = v.len();
                Ok(Some(to_numpy(py, v, &[n])?))
            }
            None => Ok(None),
        }
    }

    #[getter]
    pub fn explained_variance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Some(model) => {
                let v = model.explained_variance.clone();
                let n = v.len();
                Ok(Some(to_numpy(py, v, &[n])?))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("PCA(n_components={})", self.n_components)
    }
}
