use pyo3::prelude::*;
use dsp_base::spatial::SpatialWhitening;
use crate::array::{to_numpy, F32Array};

#[pyclass(name = "SpatialWhitening", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpatialWhitening {
    pub inner: SpatialWhitening,
}

#[pymethods]
impl PySpatialWhitening {
    #[staticmethod]
    #[pyo3(signature = (data, channels=None, epsilon=1e-5))]
    pub fn fit_zca<'py>(
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        channels: Option<usize>,
        epsilon: f32,
    ) -> PyResult<Self> {
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        let x = input.slice();
        let inner = py.detach(|| SpatialWhitening::fit_zca(x, ch, samples, epsilon));
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data, positions, k_neighbors=8, channels=None, epsilon=1e-5))]
    pub fn fit_local_knn<'py>(
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        positions: Vec<[f32; 2]>,
        k_neighbors: usize,
        channels: Option<usize>,
        epsilon: f32,
    ) -> PyResult<Self> {
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        if positions.len() != ch {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "positions length ({}) must equal channels ({})",
                positions.len(),
                ch
            )));
        }
        let x = input.slice();
        let inner = py.detach(|| {
            SpatialWhitening::fit_local_knn(x, ch, samples, &positions, k_neighbors, epsilon)
        });
        Ok(Self { inner })
    }

    #[getter]
    pub fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    pub fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.inner.num_channels;
        to_numpy(py, self.inner.matrix.clone(), &[c, c])
    }

    #[pyo3(signature = (data, channels=None))]
    pub fn run<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        channels: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = F32Array::new(&data)?;
        let (ch, samples) = input.channels_samples(channels)?;
        if ch != self.inner.num_channels {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} channels, got {}",
                self.inner.num_channels, ch
            )));
        }
        let x = input.slice();
        let out = py.detach(|| self.inner.apply_cpu(x, ch, samples));
        to_numpy(py, out, &[ch, samples])
    }

    fn __repr__(&self) -> String {
        format!("SpatialWhitening(channels={})", self.inner.num_channels)
    }
}
