use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use dsp_core::ComputeTarget;
use dsp_synapse::SpikeDetector;
use dsp_synapse_ml::{
    MyomatrixBasisEmbedder, MyomatrixDetector, MyomatrixLatencyAligner, MyomatrixProbeKind,
    MyomatrixSortConfig,
};

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

/// Specialized configuration parameters for Myomatrix / EMUsort spike sorting.
#[pyclass(name = "MyomatrixSortConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMyomatrixSortConfig {
    pub inner: MyomatrixSortConfig,
}

#[pymethods]
impl PyMyomatrixSortConfig {
    #[new]
    #[pyo3(signature = (
        template_samples=150,
        threshold_sigma=6.5,
        refractory_samples=60,
        dedup_window_samples=36,
        spatial_radius_um=6000.0,
        min_conduction_velocity=2.5,
        max_conduction_velocity=6.0,
        num_temporal_pcs=12,
        min_clusters=3,
        max_clusters=10,
    ))]
    pub fn new(
        template_samples: usize,
        threshold_sigma: f32,
        refractory_samples: usize,
        dedup_window_samples: usize,
        spatial_radius_um: f32,
        min_conduction_velocity: f32,
        max_conduction_velocity: f32,
        num_temporal_pcs: usize,
        min_clusters: usize,
        max_clusters: usize,
    ) -> Self {
        Self {
            inner: MyomatrixSortConfig {
                template_samples,
                threshold_sigma,
                refractory_samples,
                dedup_window_samples,
                spatial_radius_um,
                min_conduction_velocity,
                max_conduction_velocity,
                num_temporal_pcs,
                min_clusters,
                max_clusters,
                probe_kind: MyomatrixProbeKind::Grid32,
            },
        }
    }

    #[staticmethod]
    pub fn preset_32ch_grid() -> Self {
        Self {
            inner: MyomatrixSortConfig::preset_32ch_grid(),
        }
    }

    #[staticmethod]
    pub fn preset_64ch_grid() -> Self {
        Self {
            inner: MyomatrixSortConfig::preset_64ch_grid(),
        }
    }

    #[staticmethod]
    pub fn preset_8ch_thread() -> Self {
        Self {
            inner: MyomatrixSortConfig::preset_8ch_thread(),
        }
    }

    #[getter]
    pub fn template_samples(&self) -> usize {
        self.inner.template_samples
    }

    #[getter]
    pub fn threshold_sigma(&self) -> f32 {
        self.inner.threshold_sigma
    }

    #[getter]
    pub fn refractory_samples(&self) -> usize {
        self.inner.refractory_samples
    }

    #[getter]
    pub fn dedup_window_samples(&self) -> usize {
        self.inner.dedup_window_samples
    }

    #[getter]
    pub fn spatial_radius_um(&self) -> f32 {
        self.inner.spatial_radius_um
    }

    #[getter]
    pub fn num_temporal_pcs(&self) -> usize {
        self.inner.num_temporal_pcs
    }

    #[getter]
    pub fn min_conduction_velocity(&self) -> f32 {
        self.inner.min_conduction_velocity
    }

    #[getter]
    pub fn max_conduction_velocity(&self) -> f32 {
        self.inner.max_conduction_velocity
    }

    #[getter]
    pub fn min_clusters(&self) -> usize {
        self.inner.min_clusters
    }

    #[getter]
    pub fn max_clusters(&self) -> usize {
        self.inner.max_clusters
    }
}

/// 150-sample, 12-component spatiotemporal muscle basis embedder.
#[pyclass(name = "MyomatrixBasisEmbedder", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMyomatrixBasisEmbedder {
    pub inner: MyomatrixBasisEmbedder,
}

#[pymethods]
impl PyMyomatrixBasisEmbedder {
    #[new]
    #[pyo3(signature = (path=None, backend=None))]
    pub fn new(path: Option<&str>, backend: Option<&str>) -> PyResult<Self> {
        let target = parse_optional_target(backend)?;
        let inner = match path {
            Some(p) => MyomatrixBasisEmbedder::from_npy(p, target).map_err(runtime_error)?,
            None => MyomatrixBasisEmbedder::from_canonical(target).map_err(runtime_error)?,
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (backend=None))]
    pub fn from_canonical(backend: Option<&str>) -> PyResult<Self> {
        Self::new(None, backend)
    }

    #[staticmethod]
    #[pyo3(signature = (backend=None))]
    pub fn from_hub(backend: Option<&str>) -> PyResult<Self> {
        Self::new(None, backend)
    }

    #[staticmethod]
    #[pyo3(signature = (path, backend=None))]
    pub fn from_npy(path: &str, backend: Option<&str>) -> PyResult<Self> {
        Self::new(Some(path), backend)
    }

    #[getter]
    pub fn num_components(&self) -> usize {
        self.inner.num_components()
    }

    #[getter]
    pub fn window_len(&self) -> usize {
        self.inner.window_len()
    }

    /// Returns the `[num_components, window_len]` temporal basis matrix as a 2D NumPy array.
    pub fn basis_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat = self.inner.basis().to_vec().into_pyarray(py);
        flat.reshape([self.inner.num_components(), self.inner.window_len()])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Projects 1D `[window_len]` or 2D `[num_waveforms, window_len]` onto 12-PC temporal subspace.
    pub fn project<'py>(
        &self,
        py: Python<'py>,
        waveforms: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let wl = self.inner.window_len();
        let nc = self.inner.num_components();
        let basis = self.inner.basis();
        if let Ok(arr1) = waveforms.extract::<PyReadonlyArray1<'py, f32>>() {
            let slice = arr1
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if slice.len() != wl {
                return Err(PyValueError::new_err(format!(
                    "Expected length {wl}, got {}",
                    slice.len()
                )));
            }
            let mut out = vec![0.0f32; nc];
            for c in 0..nc {
                let mut sum = 0.0f32;
                for t in 0..wl {
                    sum += slice[t] * basis[c * wl + t];
                }
                out[c] = sum;
            }
            return Ok(out.into_pyarray(py).into_any());
        }
        if let Ok(arr2) = waveforms.extract::<PyReadonlyArray2<'py, f32>>() {
            let shape = arr2.shape();
            let (n, t) = (shape[0], shape[1]);
            if t != wl {
                return Err(PyValueError::new_err(format!(
                    "Expected {wl} samples, got {t}"
                )));
            }
            let slice = arr2
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let mut out = vec![0.0f32; n * nc];
            for i in 0..n {
                let w = &slice[i * wl..(i + 1) * wl];
                for c in 0..nc {
                    let mut sum = 0.0f32;
                    for k in 0..wl {
                        sum += w[k] * basis[c * wl + k];
                    }
                    out[i * nc + c] = sum;
                }
            }
            let flat = out.into_pyarray(py);
            return flat
                .reshape([n, nc])
                .map(|a| a.into_any())
                .map_err(|e| PyValueError::new_err(e.to_string()));
        }
        Err(PyValueError::new_err("Expected 1D or 2D array"))
    }

    /// Reconstructs waveforms from PCA projection coefficients.
    pub fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        coeffs: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let wl = self.inner.window_len();
        let nc = self.inner.num_components();
        let basis = self.inner.basis();
        if let Ok(arr1) = coeffs.extract::<PyReadonlyArray1<'py, f32>>() {
            let slice = arr1
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if slice.len() != nc {
                return Err(PyValueError::new_err(format!(
                    "Expected {nc} coefficients, got {}",
                    slice.len()
                )));
            }
            let mut out = vec![0.0f32; wl];
            for t in 0..wl {
                let mut sum = 0.0f32;
                for c in 0..nc {
                    sum += slice[c] * basis[c * wl + t];
                }
                out[t] = sum;
            }
            return Ok(out.into_pyarray(py).into_any());
        }
        if let Ok(arr2) = coeffs.extract::<PyReadonlyArray2<'py, f32>>() {
            let shape = arr2.shape();
            let (n, c_len) = (shape[0], shape[1]);
            if c_len != nc {
                return Err(PyValueError::new_err(format!(
                    "Expected {nc} coefficients, got {c_len}"
                )));
            }
            let slice = arr2
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let mut out = vec![0.0f32; n * wl];
            for i in 0..n {
                let c_row = &slice[i * nc..(i + 1) * nc];
                for t in 0..wl {
                    let mut sum = 0.0f32;
                    for c in 0..nc {
                        sum += c_row[c] * basis[c * wl + t];
                    }
                    out[i * wl + t] = sum;
                }
            }
            let flat = out.into_pyarray(py);
            return flat
                .reshape([n, wl])
                .map(|a| a.into_any())
                .map_err(|e| PyValueError::new_err(e.to_string()));
        }
        Err(PyValueError::new_err("Expected 1D or 2D array"))
    }

    /// Projects `[num_snippets, channels, window_len]` waveforms onto `[num_snippets, channels * num_components]`.
    pub fn embed<'py>(
        &self,
        py: Python<'py>,
        waveforms: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = waveforms.shape();
        let (n, k, t) = (shape[0], shape[1], shape[2]);
        let slice = waveforms
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Input waveforms must be contiguous: {e}")))?;

        let feats = self
            .inner
            .project_raw(slice, n, k, t)
            .map_err(runtime_error)?;

        let dim = k * self.inner.num_components();
        let flat = feats.into_pyarray(py);
        flat.reshape([n, dim])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// 150-sample universal MUAP matched-filter template detector.
#[pyclass(name = "MyomatrixDetector", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMyomatrixDetector {
    pub inner: MyomatrixDetector,
}

#[pymethods]
impl PyMyomatrixDetector {
    #[new]
    #[pyo3(signature = (threshold_sigma=6.5, refractory_samples=60, path=None, backend=None))]
    pub fn new(
        threshold_sigma: f32,
        refractory_samples: usize,
        path: Option<&str>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let target = parse_optional_target(backend)?;
        let inner = match path {
            Some(p) => MyomatrixDetector::from_npy(p, threshold_sigma, refractory_samples, target)
                .map_err(runtime_error)?,
            None => MyomatrixDetector::from_canonical(
                threshold_sigma,
                refractory_samples,
                target,
            )
            .map_err(runtime_error)?,
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (threshold_sigma=6.5, refractory_samples=60, backend=None))]
    pub fn from_canonical(
        threshold_sigma: f32,
        refractory_samples: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Self::new(threshold_sigma, refractory_samples, None, backend)
    }

    #[staticmethod]
    #[pyo3(signature = (threshold_sigma=6.5, refractory_samples=60, backend=None))]
    pub fn from_hub(
        threshold_sigma: f32,
        refractory_samples: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Self::new(threshold_sigma, refractory_samples, None, backend)
    }

    #[staticmethod]
    #[pyo3(signature = (path, threshold_sigma=6.5, refractory_samples=60, backend=None))]
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
    pub fn threshold_sigma(&self) -> f32 {
        self.inner.threshold_sigma
    }

    #[getter]
    pub fn refractory_samples(&self) -> usize {
        self.inner.refractory_samples
    }

    /// Returns the `[num_templates, window_len]` canonical templates as a 2D NumPy float32 array.
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

    /// Detects MUAPs in continuous multi-channel `[channels, samples]` recording.
    #[pyo3(signature = (data, sample_rate_hz=24414.0625))]
    pub fn detect<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        sample_rate_hz: f64,
    ) -> PyResult<Vec<PySpikeEvent>> {
        let shape = data.shape();
        let (ch, s) = (shape[0], shape[1]);
        let slice = data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Data array must be contiguous: {e}")))?;

        let detector = self.inner.clone();
        let events = py
            .detach(move || detector.detect(slice, ch, s, sample_rate_hz))
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

/// Cross-channel conduction latency aligner for Motor Unit Action Potentials.
#[pyclass(name = "MyomatrixLatencyAligner", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMyomatrixLatencyAligner {
    pub inner: MyomatrixLatencyAligner,
}

#[pymethods]
impl PyMyomatrixLatencyAligner {
    #[new]
    #[pyo3(signature = (max_lag_samples=25))]
    pub fn new(max_lag_samples: usize) -> Self {
        Self {
            inner: MyomatrixLatencyAligner::new(max_lag_samples),
        }
    }

    /// Estimates integer sample lag of each channel relative to `ref_ch`.
    #[pyo3(signature = (snippet, ref_ch=0))]
    pub fn estimate_channel_lags<'py>(
        &self,
        snippet: PyReadonlyArray2<'py, f32>,
        ref_ch: usize,
    ) -> PyResult<Vec<isize>> {
        let shape = snippet.shape();
        let (ch, s) = (shape[0], shape[1]);
        let slice = snippet
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Snippet must be contiguous: {e}")))?;
        if ref_ch >= ch {
            return Err(PyValueError::new_err(format!(
                "ref_ch {ref_ch} out of bounds for {ch} channels"
            )));
        }
        Ok(self.inner.estimate_channel_lags(slice, ch, s, ref_ch))
    }

    /// Time-shifts each channel's row in `snippet` by `-lag` to align with the reference channel.
    #[pyo3(signature = (snippet, lags))]
    pub fn align_snippet<'py>(
        &self,
        py: Python<'py>,
        snippet: PyReadonlyArray2<'py, f32>,
        lags: Vec<isize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = snippet.shape();
        let (ch, s) = (shape[0], shape[1]);
        if lags.len() != ch {
            return Err(PyValueError::new_err(format!(
                "lags length {} does not match channel count {ch}",
                lags.len()
            )));
        }
        let slice = snippet
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Snippet must be contiguous: {e}")))?;

        let aligned = self.inner.align_snippet(slice, ch, s, &lags);
        let flat = aligned.into_pyarray(py);
        flat.reshape([ch, s])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
