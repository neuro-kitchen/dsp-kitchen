//! `dsp_kitchen.linalg`: PCA, probabilistic PCA and FastICA, fitted on the device (covariance and
//! eigendecomposition). Data is `[channels, samples]`: channels are the features, samples the
//! observations. Defaults follow scikit-learn (`n_components=None`: all; FastICA `max_iter=200`,
//! `tol=1e-4`, `fun="logcosh"`).

use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::CubeElement;
use dsp_base::linalg::{FastIcaModel, IcaContrast, PcaModel, PpcaModel};
use dsp_core::compute::ComputeTask;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::array::{runtime_error, to_numpy, F32Array};
use crate::runtime::target;

/// scikit-learn's FastICA defaults.
const ICA_DEFAULT_MAX_ITER: usize = 200;
const ICA_DEFAULT_TOL: f32 = 1e-4;
const ICA_DEFAULT_FUN: &str = "logcosh";

/// What to fit, on the task's device.
enum Model {
    Pca,
    Ppca,
    Ica { contrast: IcaContrast, max_iter: usize, tol: f32 },
}

enum Fitted {
    Pca(PcaModel),
    Ppca(PpcaModel),
    Ica(FastIcaModel),
}

struct FitTask<'a> {
    model: Model,
    x: &'a [f32],
    channels: usize,
    samples: usize,
    components: usize,
}

impl ComputeTask for FitTask<'_> {
    type Output = Fitted;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Fitted {
        let (x, c, s, k) = (self.x, self.channels, self.samples, self.components);
        match self.model {
            Model::Pca => Fitted::Pca(PcaModel::fit::<R, f32>(&client, x, c, s, k)),
            Model::Ppca => Fitted::Ppca(PpcaModel::fit::<R, f32>(&client, x, c, s, k)),
            Model::Ica { contrast, max_iter, tol } => Fitted::Ica(FastIcaModel::fit::<R, f32>(&client, x, c, s, k, contrast, max_iter, tol)),
        }
    }
}

/// Fits `model` to `data` with `n_components` (all channels when `None`).
fn fit(py: Python<'_>, data: &Bound<'_, PyAny>, model: Model, n_components: Option<usize>, runtime: Option<&str>) -> PyResult<Fitted> {
    let input = F32Array::new(data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let task = FitTask { model, x: input.slice(), channels, samples, components: n_components.unwrap_or(channels) };
    let target = target(runtime)?;
    py.detach(|| target.run(task)).map_err(runtime_error)
}

/// Device projection `[components, samples]` of `data` with a fitted PCA / PPCA model.
struct ProjectTask<'a> {
    model: &'a Fitted,
    x: &'a [f32],
    channels: usize,
    samples: usize,
}

impl ComputeTask for ProjectTask<'_> {
    type Output = Vec<f32>;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Vec<f32> {
        let input = client.create_from_slice(f32::as_bytes(self.x));
        let components = match self.model {
            Fitted::Pca(m) => m.num_components,
            Fitted::Ppca(m) => m.num_components,
            Fitted::Ica(m) => m.num_components,
        };
        let out = client.empty(components * self.samples * std::mem::size_of::<f32>());
        match self.model {
            Fitted::Pca(m) => m.project_gpu::<R, f32>(&client, &input, &out, self.channels, self.samples),
            Fitted::Ppca(m) => m.project_gpu::<R, f32>(&client, &input, &out, self.channels, self.samples),
            Fitted::Ica(_) => unreachable!("ICA transforms on the host"),
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    }
}

/// `data` checked against a model of `channels` channels: `(values, samples)`.
fn checked<'a>(input: &'a F32Array<'_>, channels: usize) -> PyResult<(&'a [f32], usize)> {
    let (c, samples) = input.channels_samples(None)?;
    if c != channels {
        return Err(PyValueError::new_err(format!("the model has {channels} channels, data has {c}")));
    }
    Ok((input.slice(), samples))
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("fit the model first")
}

fn vector<'py>(py: Python<'py>, v: &[f32]) -> PyResult<Bound<'py, PyAny>> {
    to_numpy(py, v.to_vec(), &[v.len()])
}

#[pyclass(name = "PCA")]
pub struct PyPca {
    model: Option<Fitted>,
    n_components: Option<usize>,
}

#[pymethods]
impl PyPca {
    #[new]
    #[pyo3(signature = (n_components=None))]
    fn new(n_components: Option<usize>) -> Self {
        Self { model: None, n_components }
    }

    /// Fits on `data` (`[channels, samples]`); returns the model, so calls chain.
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        slf.model = Some(fit(data.py(), &data, Model::Pca, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// `[components, samples]` coordinates of `data`, projected on the device.
    #[pyo3(signature = (data, *, runtime=None))]
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let Some(model @ Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let target = target(runtime)?;
        let out = py.detach(|| target.run(ProjectTask { model, x, channels: m.num_channels, samples })).map_err(runtime_error)?;
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// `[components, channels]` principal axes.
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        to_numpy(py, m.components.clone(), &[m.num_components, m.num_channels])
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.mean)
    }

    #[getter]
    fn explained_variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.explained_variance)
    }

    #[getter]
    fn explained_variance_ratio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.explained_variance_ratio)
    }

    fn __repr__(&self) -> String {
        format!("PCA(n_components={:?})", self.n_components)
    }
}

#[pyclass(name = "PPCA")]
pub struct PyPpca {
    model: Option<Fitted>,
    n_components: Option<usize>,
}

#[pymethods]
impl PyPpca {
    #[new]
    #[pyo3(signature = (n_components=None))]
    fn new(n_components: Option<usize>) -> Self {
        Self { model: None, n_components }
    }

    /// Fits on `data` (`[channels, samples]`) (Tipping & Bishop maximum likelihood).
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        slf.model = Some(fit(data.py(), &data, Model::Ppca, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// `[components, samples]` posterior means of the latent coordinates, on the device.
    #[pyo3(signature = (data, *, runtime=None))]
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let Some(model @ Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let target = target(runtime)?;
        let out = py.detach(|| target.run(ProjectTask { model, x, channels: m.num_channels, samples })).map_err(runtime_error)?;
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// `[channels, samples]` reconstruction of latent coordinates `z` (`[components, samples]`).
    fn reconstruct<'py>(&self, py: Python<'py>, z: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&z)?;
        let (z, samples) = checked(&input, m.num_components)?;
        let out = py.detach(|| m.reconstruct_cpu(z, samples));
        to_numpy(py, out, &[m.num_channels, samples])
    }

    #[getter]
    fn noise_variance(&self) -> PyResult<f32> {
        let Some(Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        Ok(m.noise_variance)
    }

    fn __repr__(&self) -> String {
        format!("PPCA(n_components={:?})", self.n_components)
    }
}

#[pyclass(name = "FastICA")]
pub struct PyFastIca {
    model: Option<Fitted>,
    n_components: Option<usize>,
    contrast: IcaContrast,
    max_iter: usize,
    tol: f32,
}

#[pymethods]
impl PyFastIca {
    /// `fun`: `"logcosh"`, `"cube"` or `"skew"`.
    #[new]
    #[pyo3(signature = (n_components=None, *, fun=ICA_DEFAULT_FUN, max_iter=ICA_DEFAULT_MAX_ITER, tol=ICA_DEFAULT_TOL))]
    fn new(n_components: Option<usize>, fun: &str, max_iter: usize, tol: f32) -> PyResult<Self> {
        let contrast = match fun {
            "logcosh" => IcaContrast::LogCosh,
            "cube" => IcaContrast::Cube,
            "skew" => IcaContrast::Skew,
            other => return Err(PyValueError::new_err(format!("fun must be 'logcosh', 'cube' or 'skew', got '{other}'"))),
        };
        Ok(Self { model: None, n_components, contrast, max_iter, tol })
    }

    /// Fits on `data` (`[channels, samples]`): whitening on the device, iterations on the host.
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        let model = Model::Ica { contrast: slf.contrast, max_iter: slf.max_iter, tol: slf.tol };
        slf.model = Some(fit(data.py(), &data, model, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// `[components, samples]` sources of `data` (on the host).
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Ica(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let out = py.detach(|| m.transform_cpu(x, m.num_channels, samples));
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// `[components, channels]` unmixing matrix.
    #[getter]
    fn unmixing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Ica(m)) = &self.model else { return Err(not_fitted()) };
        to_numpy(py, m.unmixing.clone(), &[m.num_components, m.num_channels])
    }

    fn __repr__(&self) -> String {
        format!("FastICA(n_components={:?}, max_iter={}, tol={})", self.n_components, self.max_iter, self.tol)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPca>()?;
    m.add_class::<PyPpca>()?;
    m.add_class::<PyFastIca>()?;
    Ok(())
}
