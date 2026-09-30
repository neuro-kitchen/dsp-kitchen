//! PyO3 bindings for `dsp-synapse-ml`'s Burn-ONNX (`onnx-ir = "0.21.0"`) runtime engine
//! (`OnnxGraphRunner`) and external sorter profiles (**Kilosort4**, **DARTsort**, **CEBRA**, **Bombcell**).

use pyo3::prelude::*;
use dsp_synapse::{FeatureEmbedder, WaveformDenoiser};
use dsp_synapse_ml::{
    CebraProfile, DartsortProfile, Kilosort4Profile, OnnxFeatureEmbedder, OnnxGraphRunner,
    OnnxWaveformDenoiser, SynapseMlDevice, Tensor2D, Tensor3D,
};
use super::models::{numpy_3d_to_snippet_batch, snippet_batch_to_numpy_3d};
use crate::array::{runtime_error, to_numpy, F32Array};

/// Burn-ONNX (`onnx-ir = "0.21.0"`) graph runner exposed to Python.
#[pyclass(name = "OnnxModelRunner", skip_from_py_object)]
#[derive(Clone)]
pub struct PyOnnxModelRunner {
    pub inner: OnnxGraphRunner,
}

#[pymethods]
impl PyOnnxModelRunner {
    /// Loads and simplifies a `.onnx` file from disk using zero-copy `memmap2`.
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let runner = OnnxGraphRunner::from_file(path, SynapseMlDevice::Cpu)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: runner })
    }

    /// Parses and simplifies an in-memory `.onnx` byte buffer.
    #[staticmethod]
    pub fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let runner = OnnxGraphRunner::from_bytes(bytes, SynapseMlDevice::Cpu)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: runner })
    }

    #[getter]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Runs a 2D float32 numpy array `[N, F_in]` through the ONNX graph and returns `[N, F_out]`.
    pub fn run_2d<'py>(&self, py: Python<'py>, input: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let arr = F32Array::new(&input)?;
        let [n, f] = arr.shape()[..] else {
            return Err(pyo3::exceptions::PyValueError::new_err("run_2d expects a 2D float32 numpy array"));
        };
        let t_in = Tensor2D::from_floats(arr.slice().to_vec(), [n, f], SynapseMlDevice::Cpu);
        let runner = &self.inner;
        let t_out = py
            .detach(|| runner.run_2d(&t_in))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let shape = t_out.shape;
        to_numpy(py, t_out.data, &shape)
    }

    /// Runs a 3D float32 numpy array `[N, C_in, T_in]` through the ONNX graph and returns `[N, C_out, T_out]`.
    pub fn run_3d<'py>(&self, py: Python<'py>, input: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let arr = F32Array::new(&input)?;
        let [n, c, t] = arr.shape()[..] else {
            return Err(pyo3::exceptions::PyValueError::new_err("run_3d expects a 3D float32 numpy array"));
        };
        let t_in = Tensor3D::from_floats(arr.slice().to_vec(), [n, c, t], SynapseMlDevice::Cpu);
        let runner = &self.inner;
        let t_out = py
            .detach(|| runner.run_3d(&t_in))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let shape = t_out.shape;
        to_numpy(py, t_out.data, &shape)
    }

    /// Embeds a 3D `[N, K, T]` snippet array using an external sorter profile
    /// (`"kilosort4"`, `"dartsort"`, or `"cebra"`).
    #[pyo3(signature = (snippets, sorter="cebra", embedding_dim=8))]
    pub fn embed_with_profile<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
        sorter: &str,
        embedding_dim: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(&snippets)?;
        let [_, k, t] = batch.shape();
        let embedder: OnnxFeatureEmbedder = match sorter.to_ascii_lowercase().as_str() {
            "kilosort4" | "ks4" => {
                Kilosort4Profile::neuropixels_default(k, t).wrap_embedder(self.inner.clone())
            }
            "dartsort" => {
                DartsortProfile::new(k, t, embedding_dim).wrap_embedder(self.inner.clone())
            }
            "cebra" => CebraProfile::new(k, t, embedding_dim).wrap_embedder(self.inner.clone()),
            _ => OnnxFeatureEmbedder::new(self.inner.clone(), embedding_dim),
        };
        let (flat, actual_dim) = py.detach(|| embedder.embed(&batch)).map_err(runtime_error)?;
        to_numpy(py, flat, &[batch.num_spikes, actual_dim])
    }

    /// Denoises a 3D `[N, K, T]` snippet array using DARTsort's peak-normalized ONNX denoiser profile.
    pub fn denoise_dartsort<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(&snippets)?;
        let [_, k, t] = batch.shape();
        let denoiser: OnnxWaveformDenoiser =
            DartsortProfile::new(k, t, 8).wrap_denoiser(self.inner.clone());
        let out = py.detach(|| denoiser.denoise(&batch)).map_err(runtime_error)?;
        snippet_batch_to_numpy_3d(py, out)
    }
}
