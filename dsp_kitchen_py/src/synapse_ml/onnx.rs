//! PyO3 bindings for `dsp-synapse-ml`'s Burn-ONNX (`onnx-ir = "0.21.0"`) runtime engine
//! (`OnnxGraphRunner`) and external sorter profiles (**Kilosort4**, **DARTsort**, **CEBRA**, **Bombcell**).

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use dsp_synapse::{FeatureEmbedder, WaveformDenoiser};
use dsp_synapse_ml::{
    CebraProfile, DartsortProfile, Kilosort4Profile, OnnxFeatureEmbedder, OnnxGraphRunner,
    OnnxWaveformDenoiser, SynapseMlDevice, Tensor2D, Tensor3D,
};
use super::models::{numpy_3d_to_snippet_batch, snippet_batch_to_numpy_3d, vec_f32_to_numpy_2d};

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
    pub fn run_2d<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (input, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_2d expects a 2D float32 numpy array",
            ));
        }
        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;
        let slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr() as *const f32,
                raw_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let t_in = Tensor2D::from_floats(slice.to_vec(), [shape[0], shape[1]], SynapseMlDevice::Cpu);
        let t_out = self
            .inner
            .run_2d(&t_in)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        vec_f32_to_numpy_2d(py, &t_out.data, t_out.shape[0], t_out.shape[1])
    }

    /// Runs a 3D float32 numpy array `[N, C_in, T_in]` through the ONNX graph and returns `[N, C_out, T_out]`.
    pub fn run_3d<'py>(
        &self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (input, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_3d expects a 3D float32 numpy array",
            ));
        }
        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;
        let slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr() as *const f32,
                raw_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let t_in = Tensor3D::from_floats(
            slice.to_vec(),
            [shape[0], shape[1], shape[2]],
            SynapseMlDevice::Cpu,
        );
        let t_out = self
            .inner
            .run_3d(&t_in)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let out_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                t_out.data.as_ptr() as *const u8,
                t_out.data.len() * std::mem::size_of::<f32>(),
            )
        };
        let py_b = PyBytes::new(py, out_bytes);
        let flat = np.call_method1("frombuffer", (py_b, "float32"))?;
        flat.call_method1("reshape", ((t_out.shape[0], t_out.shape[1], t_out.shape[2]),))
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
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
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
        let (flat, actual_dim) = embedder.embed(&batch);
        vec_f32_to_numpy_2d(py, &flat, batch.num_spikes, actual_dim)
    }

    /// Denoises a 3D `[N, K, T]` snippet array using DARTsort's peak-normalized ONNX denoiser profile.
    pub fn denoise_dartsort<'py>(
        &self,
        py: Python<'py>,
        snippets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = numpy_3d_to_snippet_batch(py, snippets)?;
        let [_, k, t] = batch.shape();
        let denoiser: OnnxWaveformDenoiser =
            DartsortProfile::new(k, t, 8).wrap_denoiser(self.inner.clone());
        let out = denoiser.denoise(&batch);
        snippet_batch_to_numpy_3d(py, &out)
    }
}
