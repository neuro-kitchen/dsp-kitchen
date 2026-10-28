//! `dspk.pipeline`: pipeline specs (JSON or TOML) checked and run by dspk-pipeline.
//!
//! Specs cross the boundary as canonical JSON text; the Python wrapper turns dicts and files into
//! it. A spec without `execution.runtime` runs on the runtime `dspk.runtime` selects, like every
//! other call.

use std::path::PathBuf;

use dspk_pipeline::{ExecutedMode, PipelineError, PipelineSpec};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
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

/// Runs a spec (JSON text, or TOML with `toml=True`); relative paths resolve against `base_dir`.
/// Returns one report per input, each a dict (mode, halo, runtime, stored_upload, channels, samples,
/// sample_rate, windows, bytes {read, uploaded, downloaded, written}, output, provenance, seconds).
#[gen_stub_pyfunction(module = "dspk._native.pipeline")]
#[pyfunction]
#[pyo3(signature = (text, base_dir, toml=false))]
pub fn run<'py>(py: Python<'py>, text: &str, base_dir: PathBuf, toml: bool) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut spec = parse(text, toml)?;
    let target = match &spec.execution.runtime {
        Some(name) => runtime::target(Some(name))?,
        None => runtime::target(None)?,
    };
    spec.execution.runtime = Some(target.name().to_string());
    // Opens the device first so a runtime without one raises RuntimeError (the engine reuses it)
    runtime::open_client(target).map_err(PyRuntimeError::new_err)?;
    let reports = py.detach(|| dspk_pipeline::run(&spec, &base_dir)).map_err(to_py)?;
    reports.into_iter().map(|report| report_dict(py, report)).collect()
}

fn report_dict<'py>(py: Python<'py>, report: dspk_pipeline::RunReport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let (mode, left, right) = match report.mode {
        ExecutedMode::Stream => ("stream", 0, 0),
        ExecutedMode::Windows { left, right } => ("windows", left, right),
    };
    out.set_item("mode", mode)?;
    out.set_item("halo", (left, right))?;
    out.set_item("runtime", report.runtime)?;
    out.set_item("stored_upload", report.stored_upload)?;
    let bytes = PyDict::new(py);
    bytes.set_item("read", report.bytes.read)?;
    bytes.set_item("uploaded", report.bytes.uploaded)?;
    bytes.set_item("downloaded", report.bytes.downloaded)?;
    bytes.set_item("written", report.bytes.written)?;
    out.set_item("bytes", bytes)?;
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
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    m.add("SPEC_VERSION", dspk_pipeline::SPEC_VERSION)?;
    m.add("DEFAULT_CHUNK_SAMPLES", dspk_pipeline::DEFAULT_CHUNK_SAMPLES)?;
    Ok(())
}
