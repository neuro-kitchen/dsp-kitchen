//! Surface Laplacian: each channel minus the mean of its neighbours (grid or k nearest).

use dsp_base::pipeline::PipelineStage;
use dsp_base::spatial::SurfaceLaplacian;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::array::to_numpy;
use crate::pipeline::run_stage;

/// Surface Laplacian: each channel minus the mean of its neighbours, a spatial high-pass that sharpens
/// local sources (HD-EMG, ECoG grids). A pipeline stage, or applied with `run`.
///
/// Examples
/// --------
/// >>> lap = SurfaceLaplacian.from_grid_2d(8, 8)
/// >>> y = lap.run(x)
#[gen_stub_pyclass]
#[pyclass(name = "SurfaceLaplacian", skip_from_py_object)]
#[derive(Clone)]
pub struct PySurfaceLaplacian {
    pub inner: SurfaceLaplacian,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySurfaceLaplacian {
    /// 4-neighbour Laplacian of a grid (channel `r · cols + c` at row `r`, column `c`).
    ///
    /// Parameters
    /// ----------
    /// rows, cols : int
    #[staticmethod]
    fn from_grid_2d(rows: usize, cols: usize) -> Self {
        Self { inner: SurfaceLaplacian::from_grid_2d(rows, cols) }
    }

    /// Laplacian over each channel's nearest contacts.
    ///
    /// Parameters
    /// ----------
    /// positions : list of (float, float)
    ///     `(x, y)` of each channel's contact, µm.
    /// k_neighbors : int
    ///     Neighbours averaged per channel.
    #[staticmethod]
    fn from_coordinates_knn(positions: Vec<[f32; 2]>, k_neighbors: usize) -> Self {
        Self { inner: SurfaceLaplacian::from_coordinates_knn(&positions, k_neighbors) }
    }

    #[getter]
    /// Number of channels.
    fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    /// The operator, `[channels, channels]` float32 (`y = L · x`).
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.inner.num_channels;
        to_numpy(py, self.inner.matrix.clone(), &[c, c])
    }

    /// Applies the Laplacian on the device.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, converted to float32.
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     `[channels, samples]` float32.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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
