use pyo3::prelude::*;
use dsp_base::linalg::PpcaModel;
use crate::array::{to_numpy, F32Array};

#[pyclass(name = "PPCA")]
pub struct PyPpca {
    inner: Option<PpcaModel>,
    n_components: usize,
}

#[pymethods]
impl PyPpca {
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

    #[getter]
    pub fn noise_variance(&self) -> Option<f32> {
        self.inner.as_ref().map(|m| m.noise_variance)
    }

    #[pyo3(signature = (data, channels=None))]
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        channels: Option<usize>,
    ) -> PyResult<()> {
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        let (x, k) = (input.slice(), self.n_components);
        self.inner = Some(py.detach(|| PpcaModel::fit(x, ch, samples, k)));
        Ok(())
    }

    #[pyo3(signature = (data, channels=None))]
    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        channels: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("PPCA model must be fitted before transform")
        })?;
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        let x = input.slice();
        let projected = py.detach(|| model.project_cpu(x, ch, samples));
        to_numpy(py, projected, &[model.num_components, samples])
    }

    #[pyo3(signature = (latent, samples=None))]
    pub fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        latent: Bound<'py, PyAny>,
        samples: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("PPCA model must be fitted before reconstruct")
        })?;
        let input = F32Array::new(&latent)?;
        let (comp, s) = input.channels_samples(samples.map(|_| model.num_components))?;
        if comp != model.num_components {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} latent components, got {}",
                model.num_components, comp
            )));
        }
        let z = input.slice();
        let recon = py.detach(|| model.reconstruct_cpu(z, s));
        to_numpy(py, recon, &[model.num_channels, s])
    }

    fn __repr__(&self) -> String {
        format!("PPCA(n_components={})", self.n_components)
    }
}
