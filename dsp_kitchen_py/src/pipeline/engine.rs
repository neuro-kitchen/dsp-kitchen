//! `dsp_kitchen.pipeline`: stages chained on the device (`Pipeline`), and the runner every
//! stage function uses.

use cubecl::prelude::Client;
use cubecl::CubeElement;
use dsp_base::pipeline::{Pipeline, PipelineStage};
use dsp_core::compute::ComputeTask;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::{runtime_error, to_numpy, value_error, F32Array};
use crate::filter::fir::PyGaussianSmooth;
use crate::filter::iir::{PyBandpassFilter, PyBandstopFilter, PyChebyshevFilter, PyHighpassFilter, PyLowpassFilter, PyNotchFilter};
use crate::filter::non_linear::{PyMedianFilter, PyTeagerKaiser};
use crate::math::{PyClamp, PyScale, PySubtractBaseline};
use crate::runtime::target;
use crate::spatial::{PyCommonAverageReference, PySpatialWhitening, PySurfaceLaplacian};

/// Rate given to pipelines whose stages do not depend on it (only filters do).
const RATE_NOT_USED_HZ: f64 = 1.0;

/// The pipeline stage a Python stage object stands for.
fn stage_of(item: &Bound<'_, PyAny>) -> PyResult<PipelineStage> {
    macro_rules! try_stage {
        ($($class:ty => $get:expr),* $(,)?) => {
            $(if let Ok(s) = item.cast::<$class>() {
                let s = s.borrow();
                return Ok($get(&*s));
            })*
        };
    }
    try_stage!(
        PyScale => |s: &PyScale| s.stage.clone(),
        PySubtractBaseline => |s: &PySubtractBaseline| s.stage.clone(),
        PyClamp => |s: &PyClamp| s.stage.clone(),
        PyBandpassFilter => PyBandpassFilter::stage,
        PyHighpassFilter => PyHighpassFilter::stage,
        PyLowpassFilter => PyLowpassFilter::stage,
        PyBandstopFilter => PyBandstopFilter::stage,
        PyNotchFilter => PyNotchFilter::stage,
        PyChebyshevFilter => PyChebyshevFilter::stage,
        PyGaussianSmooth => |s: &PyGaussianSmooth| s.stage.clone(),
        PyMedianFilter => |s: &PyMedianFilter| s.stage.clone(),
        PyTeagerKaiser => |s: &PyTeagerKaiser| s.stage.clone(),
        PyCommonAverageReference => |_: &PyCommonAverageReference| PipelineStage::CommonAverageReference,
        PySpatialWhitening => |s: &PySpatialWhitening| PipelineStage::SpatialWhitening(s.inner.clone()),
        PySurfaceLaplacian => |s: &PySurfaceLaplacian| PipelineStage::SurfaceLaplacian(s.inner.clone()),
    );
    Err(PyTypeError::new_err(format!("not a pipeline stage: {}", item.repr()?)))
}

/// The sample rate for `pipeline`: required when a stage is a filter.
fn rate(pipeline: &Pipeline, fs: Option<f64>) -> PyResult<f64> {
    let needs_rate = pipeline.stages().iter().any(|s| matches!(s, PipelineStage::Filter(_)));
    match (needs_rate, fs) {
        (_, Some(fs)) => Ok(fs),
        (false, None) => Ok(RATE_NOT_USED_HZ),
        (true, None) => Err(PyValueError::new_err("fs (the sample rate in Hz) is required for filter stages")),
    }
}

/// Runs `pipeline` on `data` (`[channels, samples]`, or 1-D for one channel) with the GIL released,
/// on the runtime of `runtime=` or the current one; returns a new float32 array.
pub(crate) fn run_pipeline<'py>(py: Python<'py>, pipeline: Pipeline, data: &Bound<'py, PyAny>, fs: Option<f64>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    struct Task<'a> {
        pipeline: Pipeline,
        x: &'a [f32],
        channels: usize,
        samples: usize,
        fs: f64,
    }
    impl ComputeTask for Task<'_> {
        type Output = Result<Vec<f32>, dsp_base::filter::FilterError>;
        fn run(self, client: Client) -> Self::Output {
            let input = client.create_from_slice(f32::as_bytes(self.x));
            let out = self.pipeline.execute::<f32>(&client, &input, self.channels, self.samples, self.fs)?;
            Ok(f32::from_bytes(&client.read_one_unchecked(out)).to_vec())
        }
    }

    let input = F32Array::new(data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let fs = rate(&pipeline, fs)?;
    pipeline.validate(fs).map_err(value_error)?;
    let target = target(runtime)?;
    let task = Task { pipeline, x: input.slice(), channels, samples, fs };
    let out = py.detach(|| target.run(task)).map_err(runtime_error)?.map_err(value_error)?;
    to_numpy(py, out, input.shape())
}

/// Runs one stage (the stage functions: `bandpass_filter`, `median_filter`, …).
pub(crate) fn run_stage<'py>(py: Python<'py>, stage: PipelineStage, data: &Bound<'py, PyAny>, fs: Option<f64>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_pipeline(py, Pipeline::with_stages(vec![stage]), data, fs, runtime)
}

/// Stages run one after the other on the device; intermediate results never leave it.
#[pyclass(name = "Pipeline", skip_from_py_object)]
pub struct PyPipeline {
    pub(crate) stages: Vec<PipelineStage>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (stages=None))]
    fn new(stages: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<Self> {
        let stages = stages.unwrap_or_default().iter().map(stage_of).collect::<PyResult<_>>()?;
        Ok(Self { stages })
    }

    /// Appends a stage; returns the pipeline, so calls chain.
    fn add<'py>(mut slf: PyRefMut<'py, Self>, stage: Bound<'py, PyAny>) -> PyResult<PyRefMut<'py, Self>> {
        slf.stages.push(stage_of(&stage)?);
        Ok(slf)
    }

    #[getter]
    fn stages(&self) -> Vec<String> {
        self.stages.iter().map(|s| format!("{s:?}")).collect()
    }

    fn __len__(&self) -> usize {
        self.stages.len()
    }

    /// `(left, right)` samples of context a chunk needs at `fs` Hz so its interior equals
    /// whole-recording processing.
    #[pyo3(signature = (*, fs=None))]
    fn settling(&self, fs: Option<f64>) -> PyResult<(usize, usize)> {
        let pipeline = Pipeline::with_stages(self.stages.clone());
        let fs = rate(&pipeline, fs)?;
        pipeline.settling(fs).map_err(value_error)
    }

    /// Runs every stage on `data` (`[channels, samples]`, or 1-D); `fs` is required when a stage
    /// is a filter.
    #[pyo3(signature = (data, *, fs=None, runtime=None))]
    fn run<'py>(&self, py: Python<'py>, data: Bound<'py, PyAny>, fs: Option<f64>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        run_pipeline(py, Pipeline::with_stages(self.stages.clone()), &data, fs, runtime)
    }

    fn __repr__(&self) -> String {
        format!("Pipeline({:?})", self.stages)
    }
}

impl PyPipeline {
    pub(crate) fn pipeline(&self) -> Pipeline {
        Pipeline::with_stages(self.stages.clone())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipeline>()
}
