//! `dspk.ops.Pipeline`: stages chained on the device. Filters are designed and uploaded once,
//! device buffers are reused between chunks, and inside `process` the data never leaves the
//! device: one upload, every stage, one download.
//!
//! Native functions take every argument; the Python wrappers (`dspk.ops.stage`) hold the defaults,
//! taken from the constants exported here.

use std::sync::Mutex;

use dspk_ops::core::EdgeMode;
use dspk_ops::filter::fir::GAUSSIAN_DEFAULT_EDGE;
use dspk_ops::filter::non_linear::{MAX_MEDIAN_WIDTH, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dspk_ops::filter::{FilterDesign, FilterMode, FilterSpec};
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline as RustPipeline, PipelineStage};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::filter::{band, design, mode, sos_from_numpy, start};
use crate::runtime;

fn edge(name: &str) -> PyResult<EdgeMode> {
    match name {
        "zeros" => Ok(EdgeMode::Zeros),
        "odd" => Ok(EdgeMode::Odd),
        "reflect" => Ok(EdgeMode::Reflect),
        "nearest" => Ok(EdgeMode::Nearest),
        other => Err(PyValueError::new_err(format!("unknown edge {other:?} (zeros, odd, reflect, nearest)"))),
    }
}

fn edge_name(edge: EdgeMode) -> &'static str {
    match edge {
        EdgeMode::Zeros => "zeros",
        EdgeMode::Odd => "odd",
        EdgeMode::Reflect => "reflect",
        EdgeMode::Nearest => "nearest",
    }
}

/// One pipeline stage (build with the functions of `dspk.ops.stage`).
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops", frozen, from_py_object)]
#[derive(Clone)]
pub struct Stage {
    inner: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl Stage {
    fn __repr__(&self) -> String {
        format!("Stage({:?})", self.inner)
    }
}

fn stage(inner: PipelineStage) -> Stage {
    Stage { inner }
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_scale(alpha: f32, beta: f32) -> Stage {
    stage(PipelineStage::Scale { alpha, beta })
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_subtract_baseline(baseline: f32) -> Stage {
    stage(PipelineStage::SubtractBaseline { baseline })
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_clamp(min: f32, max: f32) -> PyResult<Stage> {
    if min > max {
        return Err(PyValueError::new_err(format!("clamp min {min} is above max {max}")));
    }
    Ok(stage(PipelineStage::Clamp { min, max }))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn stage_iir(band: &str, cutoff: Vec<f64>, order: usize, design: &str, ripple_db: Option<f64>, mode: &str, start: &str) -> PyResult<Stage> {
    let spec = FilterSpec { design: self::design(design, order, self::band(band, &cutoff)?, ripple_db)?, mode: self::mode(mode)?, start: self::start(start)? };
    Ok(stage(PipelineStage::Filter(spec)))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_notch(freq: f64, q: f64, mode: &str, start: &str) -> PyResult<Stage> {
    let spec = FilterSpec { design: FilterDesign::Notch { freq_hz: freq, q }, mode: self::mode(mode)?, start: self::start(start)? };
    Ok(stage(PipelineStage::Filter(spec)))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_sos(sos: PyReadonlyArray2<'_, f64>, mode: &str, start: &str) -> PyResult<Stage> {
    let spec = FilterSpec { design: FilterDesign::Sos(sos_from_numpy(sos)?), mode: self::mode(mode)?, start: self::start(start)? };
    Ok(stage(PipelineStage::Filter(spec)))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_car() -> Stage {
    stage(PipelineStage::CommonAverageReference)
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_cmr() -> Stage {
    stage(PipelineStage::CommonMedianReference)
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_gaussian_smooth(sigma_samples: f32, edge: &str) -> PyResult<Stage> {
    if !(sigma_samples > 0.0) {
        return Err(PyValueError::new_err(format!("sigma_samples must be positive, got {sigma_samples}")));
    }
    Ok(stage(PipelineStage::GaussianSmooth { sigma_samples, edge: self::edge(edge)? }))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_median(width: usize, edge: &str) -> PyResult<Stage> {
    if width % 2 == 0 || width > MAX_MEDIAN_WIDTH {
        return Err(PyValueError::new_err(format!("median width must be odd and at most {MAX_MEDIAN_WIDTH}, got {width}")));
    }
    Ok(stage(PipelineStage::Median { width, edge: self::edge(edge)? }))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_teager_kaiser(edge: &str) -> PyResult<Stage> {
    Ok(stage(PipelineStage::TeagerKaiser { edge: self::edge(edge)? }))
}

/// Stages run on the device over `(channels, samples)` chunks.
///
/// `stream=False`: every chunk on its own (zero-phase filters allowed; give chunks enough context,
/// see `settling`). `stream=True`: chunks are consecutive pieces of one signal; filter state carries
/// over, so chunked output equals whole-signal output (forward filters only).
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.ops", frozen)]
pub struct Pipeline {
    stages: RustPipeline,
    sample_rate: f64,
    stream: bool,
    target: dspk_core::ComputeTarget,
    /// Created on the first chunk (its channel count fixes the workspace's).
    workspace: Mutex<Option<(usize, PipelineWorkspace<f32>)>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Pipeline {
    #[new]
    #[pyo3(signature = (stages, sample_rate, *, stream, runtime))]
    fn new(stages: Vec<Stage>, sample_rate: f64, stream: bool, runtime: Option<&str>) -> PyResult<Self> {
        let stages = RustPipeline::with_stages(stages.into_iter().map(|s| s.inner).collect());
        stages.validate(sample_rate).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let zero_phase = |s: &PipelineStage| matches!(s, PipelineStage::Filter(spec) if spec.mode == FilterMode::ForwardBackward);
        if stream && stages.stages().iter().any(zero_phase) {
            return Err(PyValueError::new_err("stream=True needs forward filters (mode=\"forward\"): zero-phase filters need future samples"));
        }
        Ok(Self { stages, sample_rate, stream, target: runtime::target(runtime)?, workspace: Mutex::new(None) })
    }

    /// Samples of context `(left, right)` a chunk needs so its interior matches whole-signal output
    /// (`stream=False`).
    #[getter]
    fn settling(&self) -> PyResult<(usize, usize)> {
        self.stages.settling(self.sample_rate).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[getter]
    fn stream(&self) -> bool {
        self.stream
    }

    /// The runtime the pipeline runs on.
    #[getter]
    fn runtime(&self) -> &'static str {
        self.target.name()
    }

    fn __len__(&self) -> usize {
        self.stages.len()
    }

    /// Runs every stage over `data` `(channels, samples)` on the device; returns the result.
    fn process<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (channels, samples) = data.as_array().dim();
        let values: Vec<f32> = data.as_array().iter().copied().collect();
        let out = py.detach(|| -> Result<Vec<f32>, String> {
            let mut slot = self.workspace.lock().map_err(|_| "pipeline lock poisoned".to_string())?;
            if let Some((ch, _)) = slot.as_ref() {
                if *ch != channels {
                    return Err(format!("this pipeline processes {ch} channels; got {channels} (call reset() first)"));
                }
            } else {
                let client = self.target.client().map_err(|e| e.to_string())?;
                let stages = self.stages.clone();
                let workspace = if self.stream {
                    PipelineWorkspace::new_stateful(client, stages, channels, samples, self.sample_rate)
                } else {
                    PipelineWorkspace::new(client, stages, channels, samples, self.sample_rate)
                }
                .map_err(|e| e.to_string())?;
                *slot = Some((channels, workspace));
            }
            let (_, workspace) = slot.as_mut().expect("created above");
            let mut out = vec![0.0f32; values.len()];
            workspace.process_chunk(&values, samples, &mut out);
            Ok(out)
        });
        let out = out.map_err(|e| if e.starts_with("this pipeline") { PyValueError::new_err(e) } else { PyRuntimeError::new_err(e) })?;
        Ok(Array2::from_shape_vec((channels, samples), out).expect("one value per input value").into_pyarray(py))
    }

    /// Forgets the device workspace: the next chunk starts a new stream (and may change channels).
    fn reset(&self) {
        *self.workspace.lock().expect("pipeline lock") = None;
    }

    fn __repr__(&self) -> String {
        format!("Pipeline({} stages, sample_rate={} Hz, stream={}, runtime={})", self.stages.len(), self.sample_rate, self.stream, self.target.name())
    }
}

pyo3_stub_gen::module_variable!("dspk._native.ops", "MEDIAN_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops", "GAUSSIAN_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops", "TEAGER_KAISER_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops", "MAX_MEDIAN_WIDTH", usize);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Stage>()?;
    m.add_class::<Pipeline>()?;
    m.add_function(wrap_pyfunction!(stage_scale, m)?)?;
    m.add_function(wrap_pyfunction!(stage_subtract_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(stage_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(stage_iir, m)?)?;
    m.add_function(wrap_pyfunction!(stage_notch, m)?)?;
    m.add_function(wrap_pyfunction!(stage_sos, m)?)?;
    m.add_function(wrap_pyfunction!(stage_car, m)?)?;
    m.add_function(wrap_pyfunction!(stage_cmr, m)?)?;
    m.add_function(wrap_pyfunction!(stage_gaussian_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(stage_median, m)?)?;
    m.add_function(wrap_pyfunction!(stage_teager_kaiser, m)?)?;
    m.add("MEDIAN_DEFAULT_EDGE", edge_name(MEDIAN_DEFAULT_EDGE))?;
    m.add("GAUSSIAN_DEFAULT_EDGE", edge_name(GAUSSIAN_DEFAULT_EDGE))?;
    m.add("TEAGER_KAISER_DEFAULT_EDGE", edge_name(TEAGER_KAISER_DEFAULT_EDGE))?;
    m.add("MAX_MEDIAN_WIDTH", MAX_MEDIAN_WIDTH)?;
    Ok(())
}
