//! `dspk.ops`: signal-processing operations of dspk-ops. Each call uploads its input to the chosen
//! runtime, runs there and downloads the result; device-resident pipelines come separately.

pub mod filter;

use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let filter = PyModule::new(m.py(), "filter")?;
    filter::register(&filter)?;
    m.add_submodule(&filter)?;
    Ok(())
}
