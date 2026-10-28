//! `dspk.ops.spatial`: operations across channels on the device (`dspk_ops::spatial`): common
//! references, whitening and surface Laplacian.

use crate::ops::device::DeviceInput;
use dspk_ops::spatial::{execute_common_median, execute_direct_car, SpatialWhitening, SurfaceLaplacian, DEFAULT_WHITENING_EPSILON};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::device;
use super::linalg::check_channels;
use crate::runtime;

/// Common average reference: the mean over channels subtracted at every sample.
#[gen_stub_pyfunction(module = "dspk._native.ops.spatial")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn car<'py>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    device::run(py, data, runtime, |client, input, output, channels, samples| {
        execute_direct_car::<f32>(client, input, output, channels, samples);
        Ok(())
    })
}

/// Common median reference: the median over channels subtracted at every sample.
#[gen_stub_pyfunction(module = "dspk._native.ops.spatial")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn cmr<'py>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    device::run(py, data, runtime, |client, input, output, channels, samples| {
        execute_common_median::<f32>(client, input, output, channels, samples);
        Ok(())
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(car, m)?)?;
    m.add_function(wrap_pyfunction!(cmr, m)?)?;
    m.add_class::<Whitening>()?;
    m.add_class::<Laplacian>()?;
    m.add("DEFAULT_WHITENING_EPSILON", DEFAULT_WHITENING_EPSILON)?;
    Ok(())
}

/// Spatial whitening `(channels, channels)`, fitted on the device (global ZCA, or local ZCA over
/// each channel's nearest neighbours), applied on the device.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops.spatial", frozen)]
pub struct Whitening {
    pub(crate) model: SpatialWhitening,
}

#[gen_stub_pymethods]
#[pymethods]
impl Whitening {
    /// Global ZCA `U (Λ + ε)^-1/2 Uᵀ` of `data` `(channels, samples)`.
    #[staticmethod]
    #[pyo3(signature = (data, *, epsilon, runtime))]
    fn fit_zca(py: Python<'_>, data: DeviceInput, epsilon: f32, runtime: Option<&str>) -> PyResult<Self> {
        let (channels, samples) = data.dim();
        let values: Vec<f32> = data.host_f32();
        let target = runtime::target(runtime)?;
        let model = py
            .detach(move || runtime::open_client(target).map(|c| SpatialWhitening::fit_zca::<f32>(&c, &values, channels, samples, epsilon)))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { model })
    }

    /// Local ZCA: each channel whitened over its `k` nearest channels (`positions` `(channels, 2)`),
    /// as Kilosort4.
    #[staticmethod]
    #[pyo3(signature = (data, positions, k, *, epsilon, runtime))]
    fn fit_local_knn(py: Python<'_>, data: DeviceInput, positions: PyReadonlyArray2<'_, f32>, k: usize, epsilon: f32, runtime: Option<&str>) -> PyResult<Self> {
        let (channels, samples) = data.dim();
        let positions = positions_of(&positions, channels)?;
        if k == 0 || k > channels {
            return Err(PyValueError::new_err(format!("k must be in 1..={channels}, got {k}")));
        }
        let values: Vec<f32> = data.host_f32();
        let target = runtime::target(runtime)?;
        let model = py
            .detach(move || {
                runtime::open_client(target).map(|c| SpatialWhitening::fit_local_knn::<f32>(&c, &values, channels, samples, &positions, k, epsilon))
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { model })
    }

    /// A whitening from an explicit `(channels, channels)` matrix.
    #[staticmethod]
    fn from_matrix(matrix: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        Ok(Self { model: SpatialWhitening::from_matrix(square(&matrix)?, matrix.as_array().iter().copied().collect()) })
    }

    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        square_array(py, &self.model.matrix, self.model.num_channels)
    }

    /// `matrix @ data` on the device.
    #[pyo3(signature = (data, *, runtime))]
    fn apply<'py>(&self, py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        check_channels(&data, self.model.num_channels)?;
        let model = self.model.clone();
        device::run(py, data, runtime, move |client, input, output, channels, samples| {
            model.apply_gpu::<f32>(client, input, output, channels, samples);
            Ok(())
        })
    }
}

/// Surface Laplacian `(channels, channels)`: each channel minus the mean of its neighbours (grid)
/// or a distance-weighted mean of its `k` nearest (positions), applied on the device.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops.spatial", frozen)]
pub struct Laplacian {
    pub(crate) model: SurfaceLaplacian,
}

#[gen_stub_pymethods]
#[pymethods]
impl Laplacian {
    /// 5-point Laplacian on a `rows × cols` grid (channel `row * cols + col`).
    #[staticmethod]
    fn from_grid(rows: usize, cols: usize) -> PyResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("rows and cols must be at least 1"));
        }
        Ok(Self { model: SurfaceLaplacian::from_grid_2d(rows, cols) })
    }

    /// Distance-weighted Hjorth Laplacian over each channel's `k` nearest (`positions` `(channels, 2)`).
    #[staticmethod]
    fn from_positions(positions: PyReadonlyArray2<'_, f32>, k: usize) -> PyResult<Self> {
        let channels = positions.as_array().dim().0;
        let positions = positions_of(&positions, channels)?;
        Ok(Self { model: SurfaceLaplacian::from_coordinates_knn(&positions, k) })
    }

    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        square_array(py, &self.model.matrix, self.model.num_channels)
    }

    /// `matrix @ data` on the device.
    #[pyo3(signature = (data, *, runtime))]
    fn apply<'py>(&self, py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        check_channels(&data, self.model.num_channels)?;
        let model = self.model.clone();
        device::run(py, data, runtime, move |client, input, output, channels, samples| {
            model.apply_gpu::<f32>(client, input, output, channels, samples);
            Ok(())
        })
    }
}

fn positions_of(positions: &PyReadonlyArray2<'_, f32>, channels: usize) -> PyResult<Vec<[f32; 2]>> {
    let view = positions.as_array();
    if view.dim() != (channels, 2) {
        return Err(PyValueError::new_err(format!("positions must have shape ({channels}, 2), got {:?}", view.dim())));
    }
    Ok(view.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

fn square(matrix: &PyReadonlyArray2<'_, f32>) -> PyResult<usize> {
    let (r, c) = matrix.as_array().dim();
    if r != c || r == 0 {
        return Err(PyValueError::new_err(format!("matrix must be square (channels, channels), got ({r}, {c})")));
    }
    Ok(r)
}

fn square_array<'py>(py: Python<'py>, values: &[f32], n: usize) -> Bound<'py, PyArray2<f32>> {
    numpy::ndarray::Array2::from_shape_vec((n, n), values.to_vec()).expect("n × n").into_pyarray(py)
}

pyo3_stub_gen::module_variable!("dspk._native.ops.spatial", "DEFAULT_WHITENING_EPSILON", f32);
