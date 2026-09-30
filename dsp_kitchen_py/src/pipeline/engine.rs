use pyo3::prelude::*;
use dsp_base::pipeline::{Pipeline, PipelineStage};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;

use crate::math::{PyScale, PySubtractBaseline, PyClamp};
use crate::filter::{PyNotchFilter, PyBandpassFilter, PyMedianFilter, PyTeagerKaiser};
use crate::spatial::PyCommonAverageReference;

#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    stages: Vec<PipelineStage>,
}

impl PyPipeline {
    pub fn from_stages(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    pub(crate) fn to_rust_pipeline(&self) -> Pipeline {
        Pipeline::with_stages(self.stages.clone())
    }
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
        } else if let Ok(clamp) = item.extract::<PyRef<PyClamp>>() {
            self.stages.push(PipelineStage::Clamp { min: clamp.min_val, max: clamp.max_val });
        } else if let Ok(notch) = item.extract::<PyRef<PyNotchFilter>>() {
            self.stages.push(PipelineStage::Notch { freq_hz: notch.freq_hz, q: notch.q });
        } else if let Ok(bp) = item.extract::<PyRef<PyBandpassFilter>>() {
            self.stages.push(PipelineStage::Bandpass { low_hz: bp.low_hz, high_hz: bp.high_hz });
        } else if item.is_instance_of::<PyCommonAverageReference>() {
            self.stages.push(PipelineStage::CommonAverageReference);
        } else if item.is_instance_of::<PyMedianFilter>() {
            self.stages.push(PipelineStage::Median9p);
        } else if item.is_instance_of::<PyTeagerKaiser>() {
            self.stages.push(PipelineStage::TeagerKaiser);
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

    /// Computes the required filter boundary settling length in samples at `fs` Hz.
    #[pyo3(signature = (fs=30000.0))]
    pub fn settling_samples(&self, fs: f64) -> usize {
        self.to_rust_pipeline().settling_samples(fs)
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
