//! Spatial whitening (ZCA, global or over each channel's nearest neighbours), fitted on the
//! device. `epsilon` (added to eigenvalues) has no default: it depends on the data's scale.

use cubecl::prelude::Client;
use dsp_base::pipeline::PipelineStage;
use dsp_base::spatial::SpatialWhitening;
use dsp_core::compute::ComputeTask;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::array::{runtime_error, to_numpy, F32Array};
use crate::pipeline::run_stage;
use crate::runtime::target;

/// Spatial (ZCA) whitening fitted on data: decorrelates channels and scales them to unit variance,
/// `W = U (Λ + ε I)^(−1/2) Uᵀ` from the channel covariance (eigenvalues `Λ`, eigenvectors `U`).
///
/// Fit it with `fit_zca` (all channels together) or `fit_local_knn` (each channel over its nearest
/// contacts, as Kilosort4 does), then use it as a pipeline stage or with `run`.
///
/// Examples
/// --------
/// >>> w = SpatialWhitening.fit_zca(x, epsilon=1e-6)
/// >>> y = w.run(x)                                   # or Pipeline([..., w])
#[gen_stub_pyclass]
#[pyclass(name = "SpatialWhitening", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpatialWhitening {
    pub inner: SpatialWhitening,
}

/// Which whitening to fit.
enum Fit<'a> {
    Zca,
    LocalKnn { positions: &'a [[f32; 2]], k_neighbors: usize },
}

struct FitTask<'a> {
    fit: Fit<'a>,
    x: &'a [f32],
    channels: usize,
    samples: usize,
    epsilon: f32,
}

impl ComputeTask for FitTask<'_> {
    type Output = SpatialWhitening;
    fn run(self, client: Client) -> SpatialWhitening {
        match self.fit {
            Fit::Zca => SpatialWhitening::fit_zca::<f32>(&client, self.x, self.channels, self.samples, self.epsilon),
            Fit::LocalKnn { positions, k_neighbors } => {
                SpatialWhitening::fit_local_knn::<f32>(&client, self.x, self.channels, self.samples, positions, k_neighbors, self.epsilon)
            }
        }
    }
}

fn fit(py: Python<'_>, data: &Bound<'_, PyAny>, fit: Fit<'_>, epsilon: f32, runtime: Option<&str>) -> PyResult<PySpatialWhitening> {
    let input = F32Array::new(data)?;
    let (channels, samples) = input.channels_samples(None)?;
    if let Fit::LocalKnn { positions, .. } = &fit
        && positions.len() != channels
    {
        return Err(PyValueError::new_err(format!("{} positions for {channels} channels", positions.len())));
    }
    let target = target(runtime)?;
    let task = FitTask { fit, x: input.slice(), channels, samples, epsilon };
    let inner = py.detach(|| target.run(task)).map_err(runtime_error)?;
    Ok(PySpatialWhitening { inner })
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpatialWhitening {
    /// Fits ZCA whitening over all channels.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, converted to float32.
    /// epsilon : float
    ///     Added to the covariance eigenvalues (regularization; avoids blowing up near-silent
    ///     directions). Required: its right value depends on the data's scale.
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[staticmethod]
    #[pyo3(signature = (data, *, epsilon, runtime=None))]
    fn fit_zca(py: Python<'_>, data: Bound<'_, PyAny>, epsilon: f32, runtime: Option<&str>) -> PyResult<Self> {
        fit(py, &data, Fit::Zca, epsilon, runtime)
    }

    /// Fits local ZCA whitening: each channel is whitened over its `k_neighbors` nearest contacts.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, converted to float32.
    /// positions : list of (float, float)
    ///     `(x, y)` of each channel's contact, µm.
    /// k_neighbors : int
    ///     Contacts in each neighbourhood (the channel included).
    /// epsilon : float
    ///     Added to the covariance eigenvalues (see `fit_zca`).
    /// runtime : str, optional
    ///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
    #[staticmethod]
    #[pyo3(signature = (data, positions, k_neighbors, *, epsilon, runtime=None))]
    fn fit_local_knn(py: Python<'_>, data: Bound<'_, PyAny>, positions: Vec<[f32; 2]>, k_neighbors: usize, epsilon: f32, runtime: Option<&str>) -> PyResult<Self> {
        fit(py, &data, Fit::LocalKnn { positions: &positions, k_neighbors }, epsilon, runtime)
    }

    #[getter]
    /// Number of channels it whitens.
    fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    /// The whitening matrix, `[channels, channels]` float32 (`y = W · x`).
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.inner.num_channels;
        to_numpy(py, self.inner.matrix.clone(), &[c, c])
    }

    /// Whitens data on the device (`y = W · x`).
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
        run_stage(py, PipelineStage::SpatialWhitening(self.inner.clone()), &data, None, runtime)
    }

    fn __repr__(&self) -> String {
        format!("SpatialWhitening(channels={})", self.inner.num_channels)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpatialWhitening>()
}
