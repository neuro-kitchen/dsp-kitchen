//! Surface Laplacian: each channel minus the mean of its neighbours (grid or k nearest).

use dsp_base::pipeline::PipelineStage;
use dsp_base::spatial::SurfaceLaplacian;
use pyo3::prelude::*;

use crate::array::to_numpy;
use crate::pipeline::run_stage;

#[pyclass(name = "SurfaceLaplacian", skip_from_py_object)]
#[derive(Clone)]
pub struct PySurfaceLaplacian {
    pub inner: SurfaceLaplacian,
}

#[pymethods]
impl PySurfaceLaplacian {
    /// 4-neighbour Laplacian of a `rows × cols` grid (channels row by row).
    #[staticmethod]
    fn from_grid_2d(rows: usize, cols: usize) -> Self {
        Self { inner: SurfaceLaplacian::from_grid_2d(rows, cols) }
    }

    /// Laplacian over each channel's `k_neighbors` nearest contacts (`positions` in µm).
    #[staticmethod]
    fn from_coordinates_knn(positions: Vec<[f32; 2]>, k_neighbors: usize) -> Self {
        Self { inner: SurfaceLaplacian::from_coordinates_knn(&positions, k_neighbors) }
    }

    #[getter]
    fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    /// The `[channels, channels]` operator.
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.inner.num_channels;
        to_numpy(py, self.inner.matrix.clone(), &[c, c])
    }

    /// Applies the Laplacian to `data` (`[channels, samples]`) on the device.
    #[pyo3(signature = (data, *, runtime=None))]
    fn run<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        run_stage(py, PipelineStage::SurfaceLaplacian(self.inner.clone()), &data, None, runtime)
    }

    fn __repr__(&self) -> String {
        format!("SurfaceLaplacian(channels={})", self.inner.num_channels)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySurfaceLaplacian>()
}
