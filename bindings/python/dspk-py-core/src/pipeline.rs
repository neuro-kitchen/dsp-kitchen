//! `dspk.pipeline`: pipeline specs (JSON or TOML) checked and run by dspk-pipeline.
//!
//! Specs cross the boundary as canonical JSON text; the Python wrapper turns dicts and files into
//! it. A spec without `execution.runtime` runs on the runtime `dspk.runtime` selects, like every
//! other call.

use std::path::PathBuf;

use dspk_pipeline::{DataMovement, ExecutedMode, PipelineError, PipelineSpec, Plan, Progress};
use pyo3::exceptions::{PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::runtime;

fn to_py(e: PipelineError) -> PyErr {
    match e {
        PipelineError::Parse(_) | PipelineError::Invalid(_) => PyValueError::new_err(e.to_string()),
        PipelineError::Io(_) => PyOSError::new_err(e.to_string()),
        PipelineError::Dsp(e) => crate::error::to_py(e),
        PipelineError::Compute(_) => PyRuntimeError::new_err(e.to_string()),
        PipelineError::Cancelled => PyKeyboardInterrupt::new_err("pipeline run cancelled"),
    }
}

fn parse(text: &str, toml: bool) -> PyResult<PipelineSpec> {
    let spec = if toml { PipelineSpec::from_toml(text) } else { PipelineSpec::from_json(text) }.map_err(to_py)?;
    spec.chain().map_err(to_py)?;
    for (id, node) in &spec.chain().map_err(to_py)?.stages {
        node.stage().map_err(|e| PyValueError::new_err(format!("node {id:?}: {e}")))?;
    }
    Ok(spec)
}

/// Checks a spec (`toml=True` for TOML text) and returns its canonical JSON (every default
/// written out).
#[gen_stub_pyfunction(module = "dspk._native.pipeline")]
#[pyfunction]
#[pyo3(signature = (text, toml=false))]
pub fn canonical(text: &str, toml: bool) -> PyResult<String> {
    Ok(parse(text, toml)?.to_json())
}

/// What running a spec would do, per input, without touching the device: dicts with input, output,
/// mode, halo, stored_upload, channels, input_samples, input_rate, output_samples, output_rate,
/// windows, max_window_samples, bytes {read, uploaded}.
#[gen_stub_pyfunction(module = "dspk._native.pipeline")]
#[pyfunction]
#[pyo3(signature = (text, base_dir, toml=false))]
pub fn plan<'py>(py: Python<'py>, text: &str, base_dir: PathBuf, toml: bool) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let spec = parse(text, toml)?;
    let plans = py.detach(|| dspk_pipeline::plan(&spec, &base_dir)).map_err(to_py)?;
    plans.into_iter().map(|plan| plan_dict(py, plan)).collect()
}

/// Runs a spec (JSON text, or TOML with `toml=True`); relative paths resolve against `base_dir`.
/// Returns one report per input, each a dict (mode, halo, runtime, stored_upload, channels, samples,
/// sample_rate, windows, bytes {read, uploaded, downloaded, written}, output, provenance, seconds).
///
/// `progress(input, inputs, windows_done, windows)` is called after each window; returning `False`
/// cancels (KeyboardInterrupt, the unfinished output removed). Ctrl+C cancels too.
#[gen_stub_pyfunction(module = "dspk._native.pipeline")]
#[pyfunction]
#[pyo3(signature = (text, base_dir, toml=false, progress=None))]
pub fn run<'py>(
    py: Python<'py>,
    text: &str,
    base_dir: PathBuf,
    toml: bool,
    progress: Option<Py<PyAny>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut spec = parse(text, toml)?;
    let target = match &spec.execution.runtime {
        Some(name) => runtime::target(Some(name))?,
        None => runtime::target(None)?,
    };
    spec.execution.runtime = Some(target.name().to_string());
    // Opens the device first so a runtime without one raises RuntimeError (the engine reuses it)
    runtime::open_client(target).map_err(PyRuntimeError::new_err)?;
    // The first Python error from the callback or a signal; it replaces the engine's Cancelled
    let mut raised: Option<PyErr> = None;
    let observe = |p: &Progress| {
        Python::attach(|py| {
            let keep_going = py.check_signals().and_then(|()| match &progress {
                Some(f) => f.call1(py, (p.input, p.inputs, p.windows_done, p.windows))?.extract::<Option<bool>>(py).map(|r| r != Some(false)),
                None => Ok(true),
            });
            keep_going.unwrap_or_else(|e| {
                raised = Some(e);
                false
            })
        })
    };
    let reports = py.detach(|| dspk_pipeline::run_with(&spec, &base_dir, dspk_pipeline::default_open, observe));
    let reports = match (reports, raised) {
        (Err(PipelineError::Cancelled), Some(e)) => return Err(e),
        (reports, _) => reports.map_err(to_py)?,
    };
    reports.into_iter().map(|report| report_dict(py, report)).collect()
}

fn bytes_dict<'py>(py: Python<'py>, bytes: DataMovement) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("read", bytes.read)?;
    out.set_item("uploaded", bytes.uploaded)?;
    out.set_item("downloaded", bytes.downloaded)?;
    out.set_item("written", bytes.written)?;
    Ok(out)
}

fn mode_parts(mode: ExecutedMode) -> (&'static str, (usize, usize)) {
    match mode {
        ExecutedMode::Stream => ("stream", (0, 0)),
        ExecutedMode::Windows { left, right } => ("windows", (left, right)),
    }
}

fn plan_dict<'py>(py: Python<'py>, plan: Plan) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let (mode, halo) = mode_parts(plan.mode);
    out.set_item("input", plan.input)?;
    out.set_item("output", plan.output)?;
    out.set_item("mode", mode)?;
    out.set_item("halo", halo)?;
    out.set_item("stored_upload", plan.stored_upload)?;
    out.set_item("channels", plan.channels)?;
    out.set_item("input_samples", plan.input_samples)?;
    out.set_item("input_rate", plan.input_rate_hz)?;
    out.set_item("output_samples", plan.output_samples)?;
    out.set_item("output_rate", plan.output_rate_hz)?;
    out.set_item("windows", plan.windows)?;
    out.set_item("max_window_samples", plan.max_window_samples)?;
    let bytes = PyDict::new(py);
    bytes.set_item("read", plan.bytes_read)?;
    bytes.set_item("uploaded", plan.bytes_uploaded)?;
    out.set_item("bytes", bytes)?;
    Ok(out)
}

fn report_dict<'py>(py: Python<'py>, report: dspk_pipeline::RunReport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let (mode, halo) = mode_parts(report.mode);
    out.set_item("mode", mode)?;
    out.set_item("halo", halo)?;
    out.set_item("runtime", report.runtime)?;
    out.set_item("stored_upload", report.stored_upload)?;
    out.set_item("bytes", bytes_dict(py, report.bytes)?)?;
    out.set_item("channels", report.channels)?;
    out.set_item("samples", report.samples)?;
    out.set_item("sample_rate", report.sample_rate_hz)?;
    out.set_item("windows", report.windows)?;
    out.set_item("output", report.output)?;
    out.set_item("provenance", report.provenance)?;
    out.set_item("seconds", report.seconds)?;
    Ok(out)
}

/// JSON Schema of the spec, as JSON text.
#[gen_stub_pyfunction(module = "dspk._native.pipeline")]
#[pyfunction]
pub fn schema() -> String {
    PipelineSpec::json_schema()
}

pyo3_stub_gen::module_variable!("dspk._native.pipeline", "SPEC_VERSION", u32);
pyo3_stub_gen::module_variable!("dspk._native.pipeline", "DEFAULT_CHUNK_SAMPLES", usize);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical, m)?)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    m.add("SPEC_VERSION", dspk_pipeline::SPEC_VERSION)?;
    m.add("DEFAULT_CHUNK_SAMPLES", dspk_pipeline::DEFAULT_CHUNK_SAMPLES)?;
    Ok(())
}
