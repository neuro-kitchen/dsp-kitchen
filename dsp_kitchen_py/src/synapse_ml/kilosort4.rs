use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use dsp_core::ComputeTarget;
use dsp_synapse::SpikeDetector;
use dsp_synapse_ml::{Kilosort4BasisEmbedder, Kilosort4TemplateMatcher};

use crate::array::runtime_error;
use crate::synapse::PySpikeEvent;

fn parse_optional_target(backend: Option<&str>) -> PyResult<Option<ComputeTarget>> {
    match backend {
        Some(name) => ComputeTarget::parse(name)
            .map(Some)
            .map_err(|e| PyValueError::new_err(e.to_string())),
        None => Ok(None),
    }
}

#[pyclass(name = "Kilosort4BasisEmbedder", skip_from_py_object)]
#[derive(Clone)]
pub struct PyKilosort4BasisEmbedder {
    pub inner: Kilosort4BasisEmbedder,
}

#[pymethods]
impl PyKilosort4BasisEmbedder {
    #[new]
    #[pyo3(signature = (path=None, backend=None, z_score_per_channel=false))]
    pub fn new(
        path: Option<&str>,
        backend: Option<&str>,
        z_score_per_channel: bool,
    ) -> PyResult<Self> {
        let _ = z_score_per_channel;
        let target = parse_optional_target(backend)?;
        let inner = match path {
            Some(p) => Kilosort4BasisEmbedder::from_npy(p, target).map_err(runtime_error)?,
            None => Kilosort4BasisEmbedder::from_hub(target).map_err(runtime_error)?,
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (backend=None, z_score_per_channel=false))]
    pub fn from_hub(backend: Option<&str>, z_score_per_channel: bool) -> PyResult<Self> {
        Self::new(None, backend, z_score_per_channel)
    }

    #[staticmethod]
    #[pyo3(signature = (path, backend=None, z_score_per_channel=false))]
    pub fn from_npy(
        path: &str,
        backend: Option<&str>,
        z_score_per_channel: bool,
    ) -> PyResult<Self> {
        Self::new(Some(path), backend, z_score_per_channel)
    }

    #[getter]
    pub fn num_components(&self) -> usize {
        self.inner.num_components()
    }

    #[getter]
    pub fn window_len(&self) -> usize {
        self.inner.window_len()
    }

    #[getter]
    pub fn source_path(&self) -> String {
        self.inner.source_path().to_string_lossy().into_owned()
    }

    /// Returns the `[num_components, window_len]` temporal basis matrix as a 2D NumPy float32 array.
    pub fn basis_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat = self.inner.basis().to_vec().into_pyarray(py);
        flat.reshape([self.inner.num_components(), self.inner.window_len()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Alias for `basis_matrix()`.
    pub fn basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.basis_matrix(py)
    }

    /// Projects a 2D `[num_waveforms, window_len]` float32 array onto `[num_waveforms, num_components]`.
    pub fn project<'py>(
        &self,
        py: Python<'py>,
        waveforms: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = waveforms.shape();
        let (n, t) = (shape[0], shape[1]);
        let slice = waveforms
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("waveforms must be C-contiguous: {e}")))?;
        let coeffs = self.inner.project_2d(slice, n, t).map_err(runtime_error)?;
        coeffs
            .into_pyarray(py)
            .reshape([n, self.inner.num_components()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Projects a 3D `[num_spikes, num_channels, window_len]` snippet batch onto `[num_spikes, num_channels * num_components]`.
    pub fn embed<'py>(
        &self,
        py: Python<'py>,
        snippets: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = snippets.shape();
        let (n, k, t) = (shape[0], shape[1], shape[2]);
        let slice = snippets
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("snippets must be C-contiguous: {e}")))?;
        let coeffs = self
            .inner
            .project_2d(slice, n * k, t)
            .map_err(runtime_error)?;
        coeffs
            .into_pyarray(py)
            .reshape([n, k * self.inner.num_components()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Denoises a 3D `[num_spikes, num_channels, window_len]` snippet batch via rank-R orthonormal temporal basis projection and reconstruction.
    pub fn denoise<'py>(
        &self,
        py: Python<'py>,
        snippets: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
        let shape = snippets.shape();
        let (n, k, t) = (shape[0], shape[1], shape[2]);
        let slice = snippets
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("snippets must be C-contiguous: {e}")))?;
        let coeffs = self
            .inner
            .project_2d(slice, n * k, t)
            .map_err(runtime_error)?;
        let recon = self
            .inner
            .reconstruct_2d(&coeffs, n * k, self.inner.num_components())
            .map_err(runtime_error)?;
        recon
            .into_pyarray(py)
            .reshape([n, k, self.inner.window_len()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstructs `[num_waveforms, window_len]` waveforms from `[num_waveforms, num_components]` coefficients (or directly from `[num_waveforms, window_len]` waveforms).
    pub fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n, cols) = (shape[0], shape[1]);
        let slice = input
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("input must be C-contiguous: {e}")))?;

        let recon = if cols == self.inner.num_components() {
            self.inner
                .reconstruct_2d(slice, n, cols)
                .map_err(runtime_error)?
        } else if cols == self.inner.window_len() {
            let coeffs = self
                .inner
                .project_2d(slice, n, cols)
                .map_err(runtime_error)?;
            self.inner
                .reconstruct_2d(&coeffs, n, self.inner.num_components())
                .map_err(runtime_error)?
        } else {
            return Err(PyValueError::new_err(format!(
                "input second dimension ({cols}) must match either num_components ({}) or window_len ({})",
                self.inner.num_components(),
                self.inner.window_len()
            )));
        };

        recon
            .into_pyarray(py)
            .reshape([n, self.inner.window_len()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "Kilosort4Detector", skip_from_py_object)]
#[derive(Clone)]
pub struct PyKilosort4Detector {
    pub inner: Kilosort4TemplateMatcher,
}

#[pymethods]
impl PyKilosort4Detector {
    #[new]
    #[pyo3(signature = (threshold_sigma=4.5, refractory_samples=30, path=None, backend=None))]
    pub fn new(
        threshold_sigma: f32,
        refractory_samples: usize,
        path: Option<&str>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let target = parse_optional_target(backend)?;
        let inner = match path {
            Some(p) => Kilosort4TemplateMatcher::from_npy(
                p,
                threshold_sigma,
                refractory_samples,
                target,
            )
            .map_err(runtime_error)?,
            None => Kilosort4TemplateMatcher::from_hub(
                threshold_sigma,
                refractory_samples,
                target,
            )
            .map_err(runtime_error)?,
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (threshold_sigma=4.5, refractory_samples=30, backend=None))]
    pub fn from_hub(
        threshold_sigma: f32,
        refractory_samples: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Self::new(threshold_sigma, refractory_samples, None, backend)
    }

    #[staticmethod]
    #[pyo3(signature = (path, threshold_sigma=4.5, refractory_samples=30, backend=None))]
    pub fn from_npy(
        path: &str,
        threshold_sigma: f32,
        refractory_samples: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Self::new(threshold_sigma, refractory_samples, Some(path), backend)
    }

    #[getter]
    pub fn num_templates(&self) -> usize {
        self.inner.num_templates()
    }

    #[getter]
    pub fn window_len(&self) -> usize {
        self.inner.window_len()
    }

    #[getter]
    pub fn center_offset(&self) -> usize {
        self.inner.center_offset()
    }

    #[getter]
    pub fn source_path(&self) -> String {
        self.inner.source_path().to_string_lossy().into_owned()
    }

    /// Returns the `[num_templates, window_len]` L2-normalized universal templates as a 2D NumPy float32 array.
    pub fn templates_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat = self.inner.templates().to_vec().into_pyarray(py);
        flat.reshape([self.inner.num_templates(), self.inner.window_len()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Alias for `templates_matrix()`.
    pub fn templates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.templates_matrix(py)
    }

    /// Computes the `[channels, samples]` universal template matched-filter energy envelope.
    pub fn filter_energy<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = data.shape();
        let (channels, samples) = (shape[0], shape[1]);
        let slice = data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("data must be C-contiguous: {e}")))?;
        let energy = self
            .inner
            .filter_energy(slice, channels, samples)
            .map_err(runtime_error)?;
        energy
            .into_pyarray(py)
            .reshape([channels, samples])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Detects spikes across `[channels, samples]` using Kilosort4 universal templates.
    #[pyo3(signature = (data, sample_rate_hz=30000.0))]
    pub fn detect(
        &self,
        data: PyReadonlyArray2<'_, f32>,
        sample_rate_hz: f64,
    ) -> PyResult<Vec<PySpikeEvent>> {
        let shape = data.shape();
        let (channels, samples) = (shape[0], shape[1]);
        let slice = data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("data must be C-contiguous: {e}")))?;
        let events = self
            .inner
            .detect(slice, channels, samples, sample_rate_hz)
            .map_err(runtime_error)?;
        Ok(events
            .into_iter()
            .map(|s| PySpikeEvent {
                channel_id: s.channel_id,
                sample_index: s.sample_index,
                peak_amplitude_uv: s.peak_amplitude_uv,
            })
            .collect())
    }
}
