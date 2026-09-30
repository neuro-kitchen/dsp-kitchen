//! PyO3 bindings for `dsp-synapse-ml` native neural architectures (`DartsortVaeEmbedder`,
//! `ContrastiveWaveformEmbedder`, `SpatiotemporalUnetDenoiser`, `SingleChannelDenoiser`,
//! `UnitQualityClassifier`) and `.safetensors` serialization.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use dsp_synapse::{FeatureEmbedder, SnippetBatch, WaveformDenoiser};
use dsp_synapse_ml::{
    ContrastiveWaveformEmbedder, DartsortVaeEmbedder, SafetensorsMap, SingleChannelDenoiser,
    SpatiotemporalUnetDenoiser, SynapseMlDevice, UnitQualityClassifier, UnitQualityFeatures,
};
use crate::synapse::PyWaveformSnippet;

/// Helper to pack a slice of `PyWaveformSnippet` or a 3D float32 numpy array `[N, K, T]` into a `SnippetBatch`.
pub(crate) fn numpy_3d_to_snippet_batch<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<SnippetBatch> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 3D float32 array [num_spikes, num_channels, num_samples], got shape {:?}",
            shape
        )));
    }
    let [n, k, t] = [shape[0], shape[1], shape[2]];
    let py_bytes = arr.call_method0("tobytes")?;
    let raw_bytes: &[u8] = py_bytes.extract()?;
    let float_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            raw_bytes.as_ptr() as *const f32,
            raw_bytes.len() / std::mem::size_of::<f32>(),
        )
    };
    let mut channel_ids = Vec::with_capacity(n * k);
    for _ in 0..n {
        for ch in 0..k {
            channel_ids.push(ch);
        }
    }
    Ok(SnippetBatch::from_raw_parts(
        float_slice.to_vec(),
        n,
        k,
        t,
        vec![0; n],
        vec![0; n],
        vec![0.0; n],
        channel_ids,
    ))
}

pub(crate) fn snippet_batch_to_numpy_3d<'py>(
    py: Python<'py>,
    batch: &SnippetBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let [n, k, t] = batch.shape();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            batch.data.as_ptr() as *const u8,
            batch.data.len() * std::mem::size_of::<f32>(),
        )
    };
    let py_bytes = PyBytes::new(py, bytes);
    let flat = np.call_method1("frombuffer", (py_bytes, "float32"))?;
    flat.call_method1("reshape", ((n, k, t),))
}

pub(crate) fn vec_f32_to_numpy_2d<'py>(
    py: Python<'py>,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    let py_bytes = PyBytes::new(py, bytes);
    let flat = np.call_method1("frombuffer", (py_bytes, "float32"))?;
    flat.call_method1("reshape", ((rows, cols),))
}

/// Native 1D Spatio-Temporal UNet Waveform Denoiser (`[N, K, T] -> [N, K, T]`).
#[pyclass(name = "SpatiotemporalUnetDenoiser", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpatiotemporalUnetDenoiser {
    pub inner: SpatiotemporalUnetDenoiser,
}

#[pymethods]
impl PySpatiotemporalUnetDenoiser {
    #[new]
    #[pyo3(signature = (num_channels=4, num_samples=40, base_filters=8, seed=42))]
    pub fn new(
        num_channels: usize,
        num_samples: usize,
        base_filters: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: SpatiotemporalUnetDenoiser::new(
                num_channels,
                num_samples,
                base_filters,
                seed,
                SynapseMlDevice::Cpu,
            ),
        }
    }

    /// Denoises a 3D float32 numpy array `[N, K, T]` and returns a denoised `[N, K, T]` numpy array.
    pub fn denoise<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
        let out = self.inner.denoise(&batch);
        snippet_batch_to_numpy_3d(py, &out)
    }

    pub fn save_safetensors(&self, path: &str) -> PyResult<()> {
        let mut map = SafetensorsMap::new();
        self.inner.save_weights(&mut map);
        map.save_to_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub fn load_safetensors(&mut self, path: &str) -> PyResult<()> {
        let map = SafetensorsMap::from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        self.inner
            .load_weights(&map)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Native Single-Channel 1D Conv Waveform Denoiser (`[N, K, T] -> [N, K, T]`).
#[pyclass(name = "SingleChannelDenoiser", skip_from_py_object)]
#[derive(Clone)]
pub struct PySingleChannelDenoiser {
    pub inner: SingleChannelDenoiser,
}

#[pymethods]
impl PySingleChannelDenoiser {
    #[new]
    #[pyo3(signature = (hidden_channels=16, seed=42))]
    pub fn new(hidden_channels: usize, seed: u64) -> Self {
        Self {
            inner: SingleChannelDenoiser::new(hidden_channels, seed, SynapseMlDevice::Cpu),
        }
    }

    pub fn denoise<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
        let out = self.inner.denoise(&batch);
        snippet_batch_to_numpy_3d(py, &out)
    }
}

/// DARTsort-style Variational Autoencoder (VAE) waveform feature embedder (`[N, K, T] -> [N, D]`).
#[pyclass(name = "DartsortVaeEmbedder", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDartsortVaeEmbedder {
    pub inner: DartsortVaeEmbedder,
}

#[pymethods]
impl PyDartsortVaeEmbedder {
    #[new]
    #[pyo3(signature = (num_channels=4, num_samples=40, latent_dim=8, seed=42))]
    pub fn new(
        num_channels: usize,
        num_samples: usize,
        latent_dim: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: DartsortVaeEmbedder::new(
                num_channels,
                num_samples,
                latent_dim,
                seed,
                SynapseMlDevice::Cpu,
            ),
        }
    }

    /// Projects a 3D float32 numpy array `[N, K, T]` into a 2D latent matrix `[N, latent_dim]`.
    pub fn embed<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
        let (flat, dim) = self.inner.embed(&batch);
        vec_f32_to_numpy_2d(py, &flat, batch.num_spikes, dim)
    }
}

/// Contrastive SimCLR / CEBRA-style waveform feature embedder (`[N, K, T] -> [N, D]` on unit hypersphere).
#[pyclass(name = "ContrastiveWaveformEmbedder", skip_from_py_object)]
#[derive(Clone)]
pub struct PyContrastiveWaveformEmbedder {
    pub inner: ContrastiveWaveformEmbedder,
}

#[pymethods]
impl PyContrastiveWaveformEmbedder {
    #[new]
    #[pyo3(signature = (num_channels=4, proj_dim=8, seed=42))]
    pub fn new(num_channels: usize, proj_dim: usize, seed: u64) -> Self {
        Self {
            inner: ContrastiveWaveformEmbedder::new(num_channels, proj_dim, seed, SynapseMlDevice::Cpu),
        }
    }

    pub fn embed<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
        let (flat, dim) = self.inner.embed(&batch);
        vec_f32_to_numpy_2d(py, &flat, batch.num_spikes, dim)
    }
}

/// Automated Allen/IBL Single-Unit Quality Classifier (`SUA`, `MUA`, `Noise`).
#[pyclass(name = "UnitQualityClassifier", skip_from_py_object)]
#[derive(Clone)]
pub struct PyUnitQualityClassifier {
    pub inner: UnitQualityClassifier,
}

#[pymethods]
impl PyUnitQualityClassifier {
    #[new]
    #[pyo3(signature = (seed=42))]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: UnitQualityClassifier::new(seed, SynapseMlDevice::Cpu),
        }
    }

    /// Classifies an `[N, 8]` float32 feature array `[snr, isi_viol_pct, firing_rate_hz, amp_cutoff, presence_ratio, half_width_ms, trough_to_peak_ms, repol_slope]`
    /// and returns a list of `(label_str, p_sua, p_mua, p_noise)` tuples.
    pub fn classify<'py>(
        &self,
        py: Python<'py>,
        features: Bound<'py, PyAny>,
    ) -> PyResult<Vec<(String, f32, f32, f32)>> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (features, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 2 || shape[1] != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected [N, 8] float32 array of UnitQualityFeatures",
            ));
        }
        let n = shape[0];
        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;
        let s: &[f32] = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr() as *const f32,
                raw_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let mut units = Vec::with_capacity(n);
        for i in 0..n {
            let r = &s[i * 8..(i + 1) * 8];
            units.push(UnitQualityFeatures {
                snr: r[0],
                isi_violation_rate_pct: r[1],
                firing_rate_hz: r[2],
                amplitude_cutoff: r[3],
                presence_ratio: r[4],
                half_width_ms: r[5],
                trough_to_peak_ms: r[6],
                repolarization_slope: r[7],
            });
        }
        let preds = self.inner.classify_units(&units);
        let _ = PyWaveformSnippet::primary_channel; // keep import alive
        Ok(preds
            .into_iter()
            .map(|p| {
                let lbl = match p.label {
                    dsp_synapse::UnitQualityLabel::SingleUnit => "SingleUnit".to_string(),
                    dsp_synapse::UnitQualityLabel::MultiUnit => "MultiUnit".to_string(),
                    dsp_synapse::UnitQualityLabel::Noise => "Noise".to_string(),
                };
                (lbl, p.p_single_unit, p.p_multi_unit, p.p_noise)
            })
            .collect())
    }
}
