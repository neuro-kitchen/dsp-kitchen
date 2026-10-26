//! `dsp_kitchen.linalg`: PCA, probabilistic PCA and FastICA, fitted on the device (covariance and
//! eigendecomposition). Data is `[channels, samples]`: channels are the features, samples the
//! observations. Defaults follow scikit-learn (`n_components=None`: all; FastICA `max_iter=200`,
//! `tol=1e-4`, `fun="logcosh"`).

use cubecl::prelude::Client;
use cubecl::CubeElement;
use dsp_base::linalg::{FastIcaModel, IcaContrast, PcaModel, PpcaModel};
use dsp_core::compute::ComputeTask;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

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
    fn run(self, client: Client) -> Fitted {
        let (x, c, s, k) = (self.x, self.channels, self.samples, self.components);
        match self.model {
            Model::Pca => Fitted::Pca(PcaModel::fit::<f32>(&client, x, c, s, k)),
            Model::Ppca => Fitted::Ppca(PpcaModel::fit::<f32>(&client, x, c, s, k)),
            Model::Ica { contrast, max_iter, tol } => Fitted::Ica(FastIcaModel::fit::<f32>(&client, x, c, s, k, contrast, max_iter, tol)),
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
    fn run(self, client: Client) -> Vec<f32> {
        let input = client.create_from_slice(f32::as_bytes(self.x));
        let components = match self.model {
            Fitted::Pca(m) => m.num_components,
            Fitted::Ppca(m) => m.num_components,
            Fitted::Ica(m) => m.num_components,
        };
        let out = client.empty(components * self.samples * std::mem::size_of::<f32>());
        match self.model {
            Fitted::Pca(m) => m.project_gpu::<f32>(&client, &input, &out, self.channels, self.samples),
            Fitted::Ppca(m) => m.project_gpu::<f32>(&client, &input, &out, self.channels, self.samples),
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

/// Principal component analysis, fitted on the device (scikit-learn's `PCA` semantics: centred data).
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Components kept; default: as many as channels.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.linalg import PCA
/// >>> pca = PCA(3).fit(x)                 # x: [channels, samples]
/// >>> z = pca.transform(x)                # [3, samples]
/// >>> pca.explained_variance_ratio
#[gen_stub_pyclass]
#[pyclass(name = "PCA")]
pub struct PyPca {
    model: Option<Fitted>,
    n_components: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPca {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (n_components=None))]
    fn new(n_components: Option<usize>) -> Self {
        Self { model: None, n_components }
    }

    /// Fits the model on `data` and returns it, so calls chain (`PCA(3).fit(x).transform(x)`).
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`: channels are the features, samples the observations (the transpose
    ///     of scikit-learn's `[n_samples, n_features]`).
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        slf.model = Some(fit(data.py(), &data, Model::Pca, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// Coordinates of `data` on the components, `[components, samples]` float32 (on the device).
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, with the channels the model was fitted on.
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    #[pyo3(signature = (data, *, runtime=None))]
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let Some(model @ Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let target = target(runtime)?;
        let out = py.detach(|| target.run(ProjectTask { model, x, channels: m.num_channels, samples })).map_err(runtime_error)?;
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// Principal axes, `[components, channels]` float32 (unit vectors).
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        to_numpy(py, m.components.clone(), &[m.num_components, m.num_channels])
    }

    /// Mean of each channel over the fitted samples, `[channels]` float32.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.mean)
    }

    /// Variance along each component, `[components]` float32 (in the data's unit squared).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn explained_variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.explained_variance)
    }

    /// Fraction of the total variance along each component, `[components]` float32.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn explained_variance_ratio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Pca(m)) = &self.model else { return Err(not_fitted()) };
        vector(py, &m.explained_variance_ratio)
    }

    fn __repr__(&self) -> String {
        format!("PCA(n_components={:?})", self.n_components)
    }
}

/// Probabilistic PCA (Tipping & Bishop, maximum likelihood), fitted on the device: PCA with an
/// isotropic noise model, so it reconstructs data and gives the noise variance.
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Latent dimensions; default: as many as channels.
#[gen_stub_pyclass]
#[pyclass(name = "PPCA")]
pub struct PyPpca {
    model: Option<Fitted>,
    n_components: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPpca {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (n_components=None))]
    fn new(n_components: Option<usize>) -> Self {
        Self { model: None, n_components }
    }

    /// Fits the model on `data` and returns it, so calls chain (`PPCA(3).fit(x).transform(x)`).
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`: channels are the features, samples the observations (the transpose
    ///     of scikit-learn's `[n_samples, n_features]`).
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        slf.model = Some(fit(data.py(), &data, Model::Ppca, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// Posterior means of the latent coordinates, `[components, samples]` float32 (on the device).
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, with the channels the model was fitted on.
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    #[pyo3(signature = (data, *, runtime=None))]
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let Some(model @ Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let target = target(runtime)?;
        let out = py.detach(|| target.run(ProjectTask { model, x, channels: m.num_channels, samples })).map_err(runtime_error)?;
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// Data reconstructed from latent coordinates.
    ///
    /// Parameters
    /// ----------
    /// z : numpy.ndarray
    ///     `[components, samples]` (e.g. from `transform`).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     `[channels, samples]` float32.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn reconstruct<'py>(&self, py: Python<'py>, z: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&z)?;
        let (z, samples) = checked(&input, m.num_components)?;
        let out = py.detach(|| m.reconstruct_cpu(z, samples));
        to_numpy(py, out, &[m.num_channels, samples])
    }

    /// Variance of the isotropic noise (σ² of the model), in the data's unit squared.
    #[getter]
    fn noise_variance(&self) -> PyResult<f32> {
        let Some(Fitted::Ppca(m)) = &self.model else { return Err(not_fitted()) };
        Ok(m.noise_variance)
    }

    fn __repr__(&self) -> String {
        format!("PPCA(n_components={:?})", self.n_components)
    }
}

/// Independent component analysis (FastICA, scikit-learn's defaults): whitening on the device,
/// fixed-point iterations on the host.
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Sources estimated; default: as many as channels.
/// fun : {"logcosh", "cube", "skew"}, default "logcosh"
///     Contrast function (`logcosh` for general sources, `cube` for super-Gaussian ones, `skew` for
///     skewed ones).
/// max_iter : int, default 200
/// tol : float, default 1e-4
///     Convergence tolerance on the unmixing vectors.
#[gen_stub_pyclass]
#[pyclass(name = "FastICA")]
pub struct PyFastIca {
    model: Option<Fitted>,
    n_components: Option<usize>,
    contrast: IcaContrast,
    max_iter: usize,
    tol: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastIca {
    /// See the class docs.
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

    /// Fits the model on `data` and returns it, so calls chain (`FastICA(3).fit(x).transform(x)`).
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`: channels are the features, samples the observations (the transpose
    ///     of scikit-learn's `[n_samples, n_features]`).
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[pyo3(signature = (data, *, runtime=None))]
    fn fit<'py>(mut slf: PyRefMut<'py, Self>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<PyRefMut<'py, Self>> {
        let model = Model::Ica { contrast: slf.contrast, max_iter: slf.max_iter, tol: slf.tol };
        slf.model = Some(fit(data.py(), &data, model, slf.n_components, runtime)?);
        Ok(slf)
    }

    /// Estimated sources of `data`, `[components, samples]` float32.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, with the channels the model was fitted on.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn transform<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let Some(Fitted::Ica(m)) = &self.model else { return Err(not_fitted()) };
        let input = F32Array::new(&data)?;
        let (x, samples) = checked(&input, m.num_channels)?;
        let out = py.detach(|| m.transform_cpu(x, m.num_channels, samples));
        to_numpy(py, out, &[m.num_components, samples])
    }

    /// Unmixing matrix, `[components, channels]` float32 (`sources = W · (x − mean)`).
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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
