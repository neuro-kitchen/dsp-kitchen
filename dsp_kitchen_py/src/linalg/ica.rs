use pyo3::prelude::*;
use dsp_base::linalg::{FastIcaModel, IcaContrast};
use crate::array::{to_numpy, F32Array};

#[pyclass(name = "FastICA")]
pub struct PyFastIca {
    inner: Option<FastIcaModel>,
    n_components: usize,
    contrast: IcaContrast,
    max_iter: usize,
    tol: f32,
}

#[pymethods]
impl PyFastIca {
    #[new]
    #[pyo3(signature = (n_components=4, contrast="logcosh", max_iter=100, tol=1e-4))]
    pub fn new(n_components: usize, contrast: &str, max_iter: usize, tol: f32) -> PyResult<Self> {
        let c = match contrast.to_ascii_lowercase().as_str() {
            "logcosh" | "tanh" => IcaContrast::LogCosh,
            "cube" | "kurtosis" => IcaContrast::Cube,
            "skew" => IcaContrast::Skew,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown FastICA contrast '{other}'; expected 'logcosh', 'cube', or 'skew'"
                )))
            }
        };
        Ok(Self {
            inner: None,
            n_components,
            contrast: c,
            max_iter,
            tol,
        })
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
        let x = input.slice();
        let (k, contrast, max_iter, tol) = (self.n_components, self.contrast, self.max_iter, self.tol);
        self.inner = Some(py.detach(|| FastIcaModel::fit(x, ch, samples, k, contrast, max_iter, tol)));
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
            pyo3::exceptions::PyRuntimeError::new_err("FastICA model must be fitted before transform")
        })?;
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        let x = input.slice();
        let out = py.detach(|| model.transform_cpu(x, ch, samples));
        to_numpy(py, out, &[model.num_components, samples])
    }

    pub fn unmixing<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Some(m) => Ok(Some(to_numpy(
                py,
                m.unmixing.clone(),
                &[m.num_components, m.num_channels],
            )?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("FastICA(n_components={})", self.n_components)
    }
}
