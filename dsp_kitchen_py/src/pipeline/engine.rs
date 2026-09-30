use pyo3::prelude::*;
use dsp_base::pipeline::{Pipeline, PipelineStage};
use cubecl::prelude::ComputeClient;
use cubecl::{CubeElement, Runtime};
use dsp_base::{ComputeTarget, ComputeTask};

use crate::array::{to_numpy, value_error, F32Array};

use crate::math::{PyScale, PySubtractBaseline, PyClamp};
use crate::filter::{extract_filter_spec, PyMedianFilter, PyTeagerKaiser};
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
        } else if let Some(spec) = extract_filter_spec(&item) {
            self.stages.push(PipelineStage::Filter(spec));
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

    /// `(left, right)` context in samples a chunk needs at `fs` Hz so its interior matches
    /// whole-recording filtering (forward-backward filters need both sides).
    #[pyo3(signature = (fs=30000.0))]
    pub fn settling(&self, fs: f64) -> PyResult<(usize, usize)> {
        self.to_rust_pipeline()
            .settling(fs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
        run_pipeline(py, Pipeline::with_stages(self.stages.clone()), &data, fs, channels)
    }

    fn __repr__(&self) -> String {
        format!("Pipeline(stages={:?})", self.stages)
    }
}

/// Runs `pipeline` on a `[channels, samples]` array (or 1-D with `channels`) with the GIL released,
/// on the runtime selected by `DSP_KITCHEN_RUNTIME` (default: first compiled-in).
pub(crate) fn run_pipeline<'py>(
    py: Python<'py>,
    pipeline: Pipeline,
    data: &Bound<'py, PyAny>,
    fs: f64,
    channels: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    struct Task<'a> {
        pipeline: Pipeline,
        x: &'a [f32],
        channels: usize,
        samples: usize,
        fs: f64,
    }
    impl ComputeTask for Task<'_> {
        type Output = Result<Vec<f32>, dsp_base::filter::FilterError>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            let in_handle = client.create_from_slice(f32::as_bytes(self.x));
            self.pipeline
                .execute::<R>(&client, &in_handle, self.channels, self.samples, self.fs)
                .map(|h| f32::from_bytes(&client.read_one_unchecked(h)).to_vec())
        }
    }

    let input = F32Array::new(data)?;
    let (ch, samples) = input.channels_samples(channels)?;
    pipeline.validate(fs).map_err(value_error)?;
    let target = compute_target()?;
    let task = Task { pipeline, x: input.slice(), channels: ch, samples, fs };
    let out = py.detach(|| target.run(task)).map_err(value_error)?;
    to_numpy(py, out.map_err(value_error)?, &[ch, samples])
}

/// The compute runtime for native calls (`DSP_KITCHEN_RUNTIME`, default: first compiled-in).
pub(crate) fn compute_target() -> PyResult<ComputeTarget> {
    ComputeTarget::from_env().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
        pipe.add(PipelineStage::notch(
            notch_freq.unwrap_or(60.0) as f64,
            notch_q.unwrap_or(30.0) as f64,
        ));

        let flat = F32Array::new(&input_data)?.ndim() == 1;
        run_pipeline(py, pipe, &input_data, self.sample_rate, flat.then_some(self.channels))
    }

    pub fn info(&self) -> String {
        format!("DspSession(channels={}, sample_rate={:.1}Hz)", self.channels, self.sample_rate)
    }
}
