//! `dspk.ops.linalg`: models fitted on the device (`dspk_ops::linalg`): PCA and probabilistic PCA.
//!
//! Fitting computes the covariance and its eigendecomposition on the device; `transform` projects
//! on the device. FastICA follows once its iterations run on the device.

use dspk_ops::linalg::{PcaModel, PpcaModel};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::device;
use crate::runtime;

/// `data` as a host `[channels, samples]` vector, checked for a fit.
fn fit_input(data: &PyReadonlyArray2<'_, f32>, n_components: usize) -> PyResult<(Vec<f32>, usize, usize)> {
    let (channels, samples) = data.as_array().dim();
    if channels == 0 || samples < 2 {
        return Err(PyValueError::new_err(format!("need at least one channel and two samples, got {channels} × {samples}")));
    }
    if n_components == 0 || n_components > channels {
        return Err(PyValueError::new_err(format!("n_components must be in 1..={channels}, got {n_components}")));
    }
    Ok((data.as_array().iter().copied().collect(), channels, samples))
}

fn rows<'py>(py: Python<'py>, values: &[f32], cols: usize) -> Bound<'py, PyArray2<f32>> {
    numpy::ndarray::Array2::from_shape_vec((values.len() / cols, cols), values.to_vec()).expect("rows × cols").into_pyarray(py)
}

/// Principal component analysis of `(channels, samples)` data: channels are the features.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops.linalg", frozen)]
pub struct Pca {
    model: PcaModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl Pca {
    /// Fits `n_components` components on `data` `(channels, samples)`.
    #[staticmethod]
    #[pyo3(signature = (data, n_components, *, runtime))]
    fn fit(py: Python<'_>, data: PyReadonlyArray2<'_, f32>, n_components: usize, runtime: Option<&str>) -> PyResult<Self> {
        let (values, channels, samples) = fit_input(&data, n_components)?;
        let target = runtime::target(runtime)?;
        let model = py
            .detach(move || runtime::open_client(target).map(|client| PcaModel::fit::<f32>(&client, &values, channels, samples, n_components)))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { model })
    }

    /// `(channels, n_components)`: one component per column.
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        rows(py, &self.model.components, self.model.num_components)
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.model.mean.clone().into_pyarray(py)
    }

    #[getter]
    fn explained_variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.model.explained_variance.clone().into_pyarray(py)
    }

    #[getter]
    fn explained_variance_ratio<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.model.explained_variance_ratio.clone().into_pyarray(py)
    }

    /// Projects `data` `(channels, samples)` onto the components: `(n_components, samples)`.
    #[pyo3(signature = (data, *, runtime))]
    fn transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        check_channels(&data, self.model.num_channels)?;
        let (model, samples) = (self.model.clone(), data.as_array().dim().1);
        device::run_shaped(py, data, runtime, (model.num_components, samples), move |client, input, output, channels, samples| {
            model.project_gpu::<f32>(client, input, output, channels, samples);
            Ok(())
        })
    }
}

/// Probabilistic PCA (Tipping & Bishop): closed-form maximum-likelihood fit; `transform` gives the
/// posterior mean of the latent variables.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops.linalg", frozen)]
pub struct Ppca {
    model: PpcaModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl Ppca {
    /// Fits `n_components` latent dimensions on `data` `(channels, samples)`.
    #[staticmethod]
    #[pyo3(signature = (data, n_components, *, runtime))]
    fn fit(py: Python<'_>, data: PyReadonlyArray2<'_, f32>, n_components: usize, runtime: Option<&str>) -> PyResult<Self> {
        let (values, channels, samples) = fit_input(&data, n_components)?;
        let target = runtime::target(runtime)?;
        let model = py
            .detach(move || runtime::open_client(target).map(|client| PpcaModel::fit::<f32>(&client, &values, channels, samples, n_components)))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { model })
    }

    /// Loadings `W` `(channels, n_components)`.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        rows(py, &self.model.weights, self.model.num_components)
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.model.mean.clone().into_pyarray(py)
    }

    /// Isotropic noise variance σ².
    #[getter]
    fn noise_variance(&self) -> f32 {
        self.model.noise_variance
    }

    /// Posterior mean `E[z | x]` of `data` `(channels, samples)`: `(n_components, samples)`.
    #[pyo3(signature = (data, *, runtime))]
    fn transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        check_channels(&data, self.model.num_channels)?;
        let (model, samples) = (self.model.clone(), data.as_array().dim().1);
        device::run_shaped(py, data, runtime, (model.num_components, samples), move |client, input, output, channels, samples| {
            model.project_gpu::<f32>(client, input, output, channels, samples);
            Ok(())
        })
    }
}

pub(crate) fn check_channels(data: &PyReadonlyArray2<'_, f32>, expected: usize) -> PyResult<()> {
    let channels = data.as_array().dim().0;
    if channels != expected {
        return Err(PyValueError::new_err(format!("the model was fitted on {expected} channels, data has {channels}")));
    }
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pca>()?;
    m.add_class::<Ppca>()?;
    Ok(())
}
