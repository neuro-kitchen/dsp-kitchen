//! `dspk.ops.Pipeline`: stages chained on the device. Filters are designed and uploaded once,
//! device buffers are reused between chunks, and inside `process` the data never leaves the
//! device: one upload, every stage, one download.
//!
//! Native functions take every argument; the Python wrappers (`dspk.ops.stage`) hold the defaults,
//! taken from the constants `dspk.ops.filter` exports.

use crate::ops::device::DeviceInput;
use std::sync::Mutex;

use dspk_ops::core::buffer;
use dspk_ops::filter::non_linear::MAX_MEDIAN_WIDTH;
use dspk_ops::filter::{FilterDesign, FilterMode, FilterSpec};
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline as RustPipeline, PipelineStage};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::filter::{band, design, edge, mode, sos_from_numpy, start};
use crate::runtime;


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
fn stage_fir(taps: Vec<f32>, centered: bool, edge: &str) -> PyResult<Stage> {
    if taps.is_empty() {
        return Err(PyValueError::new_err("taps is empty"));
    }
    if centered && taps.len().is_multiple_of(2) {
        return Err(PyValueError::new_err(format!("centered FIR needs an odd number of taps, got {}", taps.len())));
    }
    Ok(stage(PipelineStage::Fir { taps, centered, edge: self::edge(edge)? }))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_whiten(whitening: PyRef<'_, super::spatial::Whitening>) -> Stage {
    stage(PipelineStage::SpatialWhitening(whitening.model.clone()))
}

#[gen_stub_pyfunction(module = "dspk._native.ops")]
#[pyfunction]
fn stage_laplacian(laplacian: PyRef<'_, super::spatial::Laplacian>) -> Stage {
    stage(PipelineStage::SurfaceLaplacian(laplacian.model.clone()))
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
    if width.is_multiple_of(2) || width > MAX_MEDIAN_WIDTH {
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

    /// Runs every stage over `data` `(channels, samples)` on the device; returns the result
    /// (float32). Integer arrays cross as their integers and are converted on the device.
    fn process<'py>(&self, py: Python<'py>, data: DeviceInput) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (channels, samples) = data.dim();
        let out = py.detach(|| -> Result<Vec<f32>, String> {
            let mut slot = self.workspace.lock().map_err(|_| "pipeline lock poisoned".to_string())?;
            if let Some((ch, _)) = slot.as_ref() {
                if *ch != channels {
                    return Err(format!("this pipeline processes {ch} channels; got {channels} (call reset() first)"));
                }
            } else {
                // Fitted spatial stages hold a channel count; dspk-ops asserts it
                for stage in self.stages.stages() {
                    let fitted = match stage {
                        PipelineStage::SpatialWhitening(w) => Some(("whitening", w.num_channels)),
                        PipelineStage::SurfaceLaplacian(l) => Some(("laplacian", l.num_channels)),
                        _ => None,
                    };
                    if let Some((name, n)) = fitted.filter(|(_, n)| *n != channels) {
                        return Err(format!("this pipeline's {name} stage is for {n} channels; got {channels}"));
                    }
                }
                let client = runtime::open_client(self.target)?;
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
            // Integer arrays cross as their stored integers and are converted on the device
            let handle = match data.stored() {
                Some((format, bytes)) => {
                    workspace.set_stored_scaling(&vec![1.0; channels], &vec![0.0; channels]);
                    workspace.process_stored_chunk_in_vram(bytes, format, samples).map_err(|e| e.to_string())?
                }
                None => workspace.process_chunk_in_vram(&data.host_f32(), samples),
            };
            Ok(buffer::download_prefix::<f32>(workspace.client(), handle, data.len()))
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


pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Stage>()?;
    m.add_class::<Pipeline>()?;
    m.add_function(wrap_pyfunction!(stage_scale, m)?)?;
    m.add_function(wrap_pyfunction!(stage_subtract_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(stage_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(stage_iir, m)?)?;
    m.add_function(wrap_pyfunction!(stage_notch, m)?)?;
    m.add_function(wrap_pyfunction!(stage_sos, m)?)?;
    m.add_function(wrap_pyfunction!(stage_fir, m)?)?;
    m.add_function(wrap_pyfunction!(stage_whiten, m)?)?;
    m.add_function(wrap_pyfunction!(stage_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(stage_car, m)?)?;
    m.add_function(wrap_pyfunction!(stage_cmr, m)?)?;
    m.add_function(wrap_pyfunction!(stage_gaussian_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(stage_median, m)?)?;
    m.add_function(wrap_pyfunction!(stage_teager_kaiser, m)?)?;
    Ok(())
}
