use std::fs::File;
use std::sync::Arc;
use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMemoryView};

use dsp_core::probe::ProbeLayout;
use dsp_base::pipeline::{Pipeline, PipelineStage};
use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

// ============================================================================
// Probe Geometry
// ============================================================================

/// Neural Probe Layout representation for Python (SpikeInterface compatible).
#[pyclass(name = "ProbeLayout", skip_from_py_object)]
#[derive(Clone)]
pub struct PyProbeLayout {
    inner: ProbeLayout,
}

#[pymethods]
impl PyProbeLayout {
    #[staticmethod]
    pub fn neuropixels_1_0() -> Self {
        Self {
            inner: ProbeLayout::neuropixels_1_0_standard(),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn total_channels(&self) -> usize {
        self.inner.total_channels()
    }

    #[getter]
    pub fn active_channels(&self) -> usize {
        self.inner.active_channels()
    }

    pub fn contact_positions(&self) -> Vec<[f32; 3]> {
        self.inner
            .contacts
            .iter()
            .map(|c| [c.position.x_um, c.position.y_um, c.position.z_um])
            .collect()
    }

    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.channel_id).collect()
    }

    pub fn shank_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.shank_id).collect()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.inner.name)?;
        dict.set_item("ndim", 3)?;
        dict.set_item("total_channels", self.inner.total_channels())?;
        dict.set_item("contact_positions", self.contact_positions())?;
        dict.set_item("channel_ids", self.channel_ids())?;
        dict.set_item("shank_ids", self.shank_ids())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbeLayout(name='{}', channels={}, active={})",
            self.inner.name,
            self.inner.total_channels(),
            self.inner.active_channels()
        )
    }
}

// ============================================================================
// Zero-Copy Binary Recording Mmap
// ============================================================================

/// Zero-copy Memory-Mapped Raw Binary Recording.
#[pyclass(name = "MmapRecording", skip_from_py_object)]
pub struct PyMmapRecording {
    path: String,
    mmap: Arc<Mmap>,
    channels: usize,
    samples: usize,
    sample_rate: f64,
}

#[pymethods]
impl PyMmapRecording {
    #[new]
    #[pyo3(signature = (path, channels=384, samples=0, sample_rate=30000.0))]
    pub fn new(path: &str, channels: usize, mut samples: usize, sample_rate: f64) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Failed to open recording file '{}': {}",
                path, e
            ))
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to memory-map file '{}': {}",
                    path, e
                ))
            })?
        };

        let total_bytes = mmap.len();
        let bytes_per_sample = std::mem::size_of::<f32>();

        if samples == 0 {
            if channels == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("channels must be > 0"));
            }
            samples = total_bytes / (channels * bytes_per_sample);
        }

        let expected_bytes = channels * samples * bytes_per_sample;
        if total_bytes < expected_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "File size ({} bytes) is smaller than required ({} bytes for {} ch x {} samples float32)",
                total_bytes, expected_bytes, channels, samples
            )));
        }

        Ok(Self {
            path: path.to_string(),
            mmap: Arc::new(mmap),
            channels,
            samples,
            sample_rate,
        })
    }

    #[getter]
    pub fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    pub fn samples(&self) -> usize {
        self.samples
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[getter]
    pub fn total_bytes(&self) -> usize {
        self.mmap.len()
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.channels, self.samples)
    }

    pub fn memoryview<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMemoryView>> {
        let expected_bytes = self.channels * self.samples * std::mem::size_of::<f32>();
        let slice = &self.mmap[..expected_bytes];

        unsafe {
            let ptr = pyo3::ffi::PyMemoryView_FromMemory(
                slice.as_ptr() as *mut std::ffi::c_char,
                slice.len() as isize,
                pyo3::ffi::PyBUF_READ,
            );
            if ptr.is_null() {
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
        }
    }

    pub fn to_numpy<'py>(self_: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let (channels, samples) = {
            let inner = self_.borrow();
            (inner.channels, inner.samples)
        };
        let memview = self_.borrow().memoryview(py)?;
        let flat_arr = np.call_method1("frombuffer", (memview, "float32"))?;
        let reshaped = flat_arr.call_method1("reshape", ((channels, samples),))?;
        Ok(reshaped)
    }

    fn __repr__(&self) -> String {
        format!(
            "MmapRecording(path='{}', shape=({}, {}), sample_rate={:.1}Hz, size={:.2}MB)",
            self.path,
            self.channels,
            self.samples,
            self.sample_rate,
            self.mmap.len() as f64 / 1_048_576.0
        )
    }
}

// ============================================================================
// Pipeline Stages
// ============================================================================

#[pyclass(name = "Scale", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScale {
    pub alpha: f32,
    pub beta: f32,
}

#[pymethods]
impl PyScale {
    #[new]
    #[pyo3(signature = (alpha=0.195, beta=0.0))]
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }

    fn __repr__(&self) -> String {
        format!("Scale(alpha={}, beta={})", self.alpha, self.beta)
    }
}

#[pyclass(name = "SubtractBaseline", skip_from_py_object)]
#[derive(Clone)]
pub struct PySubtractBaseline {
    pub baseline_uv: f32,
}

#[pymethods]
impl PySubtractBaseline {
    #[new]
    #[pyo3(signature = (baseline_uv=0.0))]
    pub fn new(baseline_uv: f32) -> Self {
        Self { baseline_uv }
    }

    fn __repr__(&self) -> String {
        format!("SubtractBaseline(baseline_uv={})", self.baseline_uv)
    }
}

#[pyclass(name = "NotchFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyNotchFilter {
    pub freq_hz: f64,
    pub q: f64,
}

#[pymethods]
impl PyNotchFilter {
    #[new]
    #[pyo3(signature = (freq_hz=None, q=30.0, freq=None))]
    pub fn new(freq_hz: Option<f64>, q: f64, freq: Option<f64>) -> Self {
        let f = freq_hz.or(freq).unwrap_or(60.0);
        Self { freq_hz: f, q }
    }

    fn __repr__(&self) -> String {
        format!("NotchFilter(freq_hz={}Hz, q={})", self.freq_hz, self.q)
    }
}

#[pyclass(name = "BandpassFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBandpassFilter {
    pub low_hz: f64,
    pub high_hz: f64,
}

#[pymethods]
impl PyBandpassFilter {
    #[new]
    #[pyo3(signature = (low_hz=None, high_hz=None, low=None, high=None))]
    pub fn new(low_hz: Option<f64>, high_hz: Option<f64>, low: Option<f64>, high: Option<f64>) -> Self {
        let l = low_hz.or(low).unwrap_or(300.0);
        let h = high_hz.or(high).unwrap_or(6000.0);
        Self { low_hz: l, high_hz: h }
    }

    fn __repr__(&self) -> String {
        format!("BandpassFilter(low_hz={}Hz, high_hz={}Hz)", self.low_hz, self.high_hz)
    }
}

#[pyclass(name = "CommonAverageReference", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCommonAverageReference;

#[pymethods]
impl PyCommonAverageReference {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "CommonAverageReference()".to_string()
    }
}

#[pyclass(name = "MedianFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMedianFilter;

#[pymethods]
impl PyMedianFilter {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "MedianFilter(points=9)".to_string()
    }
}

// ============================================================================
// Composable In-VRAM Pipeline Engine
// ============================================================================

#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    stages: Vec<PipelineStage>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (stages=None))]
    pub fn new(stages: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<Self> {
        let mut pipe = Self { stages: Vec::new() };
        if let Some(stage_list) = stages {
            for item in stage_list {
                pipe.add(item)?;
            }
        }
        Ok(pipe)
    }

    pub fn add(&mut self, item: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(scale) = item.extract::<PyRef<PyScale>>() {
            self.stages.push(PipelineStage::Scale { alpha: scale.alpha, beta: scale.beta });
        } else if let Ok(base) = item.extract::<PyRef<PySubtractBaseline>>() {
            self.stages.push(PipelineStage::SubtractBaseline { baseline_uv: base.baseline_uv });
        } else if let Ok(notch) = item.extract::<PyRef<PyNotchFilter>>() {
            self.stages.push(PipelineStage::Notch { freq_hz: notch.freq_hz, q: notch.q });
        } else if let Ok(bp) = item.extract::<PyRef<PyBandpassFilter>>() {
            self.stages.push(PipelineStage::Bandpass { low_hz: bp.low_hz, high_hz: bp.high_hz });
        } else if item.is_instance_of::<PyCommonAverageReference>() {
            self.stages.push(PipelineStage::CommonAverageReference);
        } else if item.is_instance_of::<PyMedianFilter>() {
            self.stages.push(PipelineStage::Median9p);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unrecognized pipeline stage: {}", item.repr()?
            )));
        }
        Ok(())
    }

    #[getter]
    pub fn stages(&self) -> Vec<String> {
        self.stages.iter().map(|s| format!("{:?}", s)).collect()
    }

    pub fn __len__(&self) -> usize {
        self.stages.len()
    }

    /// Executes the pipeline across 2D array [channels, samples] directly inside GPU VRAM.
    #[pyo3(signature = (data, fs=30000.0, channels=None))]
    pub fn run<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        fs: f64,
        channels: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;

        // Determine shape
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let (ch, samples) = match shape.len() {
            1 => {
                let c = channels.unwrap_or(1);
                let s = shape[0] / c;
                (c, s)
            }
            2 => (shape[0], shape[1]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Data must be a 1D or 2D float32 array [channels, samples]"
                ));
            }
        };

        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;

        let rust_pipeline = Pipeline::with_stages(self.stages.clone());

        let out_bytes = {
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);

            let in_handle = client.create_from_slice(raw_bytes);
            let out_handle = rust_pipeline.execute::<WgpuRuntime>(
                &client,
                &in_handle,
                ch,
                samples,
                fs,
                false,
            );

            client.read_one_unchecked(out_handle)
        };

        let out_py_bytes = pyo3::types::PyBytes::new(py, &out_bytes);
        let flat_arr = np.call_method1("frombuffer", (out_py_bytes, "float32"))?;
        let reshaped = flat_arr.call_method1("reshape", ((ch, samples),))?;
        Ok(reshaped)
    }

    fn __repr__(&self) -> String {
        format!("Pipeline(stages={:?})", self.stages)
    }
}

// ============================================================================
// Legacy compatibility DspSession
// ============================================================================

#[pyclass(name = "DspSession")]
pub struct PyDspSession {
    sample_rate: f64,
    channels: usize,
}

#[pymethods]
impl PyDspSession {
    #[new]
    #[pyo3(signature = (sample_rate=30000.0, channels=384))]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self { sample_rate, channels }
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[getter]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[pyo3(signature = (input_data, alpha=None, beta=None, notch_freq=None, notch_q=None))]
    pub fn run_pipeline_wgpu<'py>(
        &self,
        py: Python<'py>,
        input_data: Bound<'py, PyAny>,
        alpha: Option<f32>,
        beta: Option<f32>,
        notch_freq: Option<f32>,
        notch_q: Option<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut pipe = Pipeline::new();
        pipe.add(PipelineStage::Scale {
            alpha: alpha.unwrap_or(0.195),
            beta: beta.unwrap_or(0.0),
        });
        pipe.add(PipelineStage::Notch {
            freq_hz: notch_freq.unwrap_or(60.0) as f64,
            q: notch_q.unwrap_or(30.0) as f64,
        });

        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (input_data, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let (ch, samples) = match shape.len() {
            1 => (self.channels, shape[0] / self.channels),
            2 => (shape[0], shape[1]),
            _ => (self.channels, 0),
        };

        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;

        let out_bytes = {
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);

            let in_handle = client.create_from_slice(raw_bytes);
            let out_handle = pipe.execute::<WgpuRuntime>(
                &client,
                &in_handle,
                ch,
                samples,
                self.sample_rate,
                false,
            );

            client.read_one_unchecked(out_handle)
        };

        let out_py_bytes = pyo3::types::PyBytes::new(py, &out_bytes);
        let flat_arr = np.call_method1("frombuffer", (out_py_bytes, "float32"))?;
        let reshaped = flat_arr.call_method1("reshape", ((ch, samples),))?;
        Ok(reshaped)
    }

    pub fn info(&self) -> String {
        format!("DspSession(channels={}, sample_rate={:.1}Hz)", self.channels, self.sample_rate)
    }
}

// ============================================================================
// Direct Functional Helpers (Model 1)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, freq=60.0, q=30.0, fs=30000.0))]
pub fn notch_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, freq: f64, q: f64, fs: f64) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Notch { freq_hz: freq, q });
    let py_pipe = PyPipeline { stages: pipe.stages().to_vec() };
    py_pipe.run(py, data, fs, None)
}

#[pyfunction]
#[pyo3(signature = (data, low=300.0, high=6000.0, fs=30000.0))]
pub fn bandpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, low: f64, high: f64, fs: f64) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Bandpass { low_hz: low, high_hz: high });
    let py_pipe = PyPipeline { stages: pipe.stages().to_vec() };
    py_pipe.run(py, data, fs, None)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn common_average_reference<'py>(py: Python<'py>, data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::CommonAverageReference);
    let py_pipe = PyPipeline { stages: pipe.stages().to_vec() };
    py_pipe.run(py, data, 30000.0, None)
}

#[pyfunction]
#[pyo3(signature = (data, alpha=0.195, beta=0.0))]
pub fn scale_samples<'py>(py: Python<'py>, data: Bound<'py, PyAny>, alpha: f32, beta: f32) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Scale { alpha, beta });
    let py_pipe = PyPipeline { stages: pipe.stages().to_vec() };
    py_pipe.run(py, data, 30000.0, None)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn median_filter_9p<'py>(py: Python<'py>, data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Median9p);
    let py_pipe = PyPipeline { stages: pipe.stages().to_vec() };
    py_pipe.run(py, data, 30000.0, None)
}

// ============================================================================
// Root Module Registration
// ============================================================================

#[pymodule]
fn _dsp_kitchen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProbeLayout>()?;
    m.add_class::<PyMmapRecording>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PySubtractBaseline>()?;
    m.add_class::<PyNotchFilter>()?;
    m.add_class::<PyBandpassFilter>()?;
    m.add_class::<PyCommonAverageReference>()?;
    m.add_class::<PyMedianFilter>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyDspSession>()?;
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    m.add_function(wrap_pyfunction!(scale_samples, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_9p, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
