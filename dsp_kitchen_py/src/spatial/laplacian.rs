use pyo3::prelude::*;
use dsp_base::spatial::SurfaceLaplacian;
use crate::array::{to_numpy, F32Array};

#[pyclass(name = "SurfaceLaplacian", skip_from_py_object)]
#[derive(Clone)]
pub struct PySurfaceLaplacian {
    pub inner: SurfaceLaplacian,
}

#[pymethods]
impl PySurfaceLaplacian {
    #[staticmethod]
    pub fn from_grid_2d(rows: usize, cols: usize) -> Self {
        Self {
            inner: SurfaceLaplacian::from_grid_2d(rows, cols),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (positions, k_neighbors=4))]
    pub fn from_coordinates_knn(positions: Vec<[f32; 2]>, k_neighbors: usize) -> Self {
        Self {
            inner: SurfaceLaplacian::from_coordinates_knn(&positions, k_neighbors),
        }
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
        format!("SurfaceLaplacian(channels={})", self.inner.num_channels)
    }
}
