//! `dspk.ops`: signal-processing operations of dspk-ops. Each call uploads its input to the chosen
//! runtime, runs there and downloads the result; device-resident pipelines come separately.

mod device;
pub mod filter;
pub mod linalg;
pub mod math;
pub mod peaks;
pub mod pipeline;
pub mod resample;
pub mod spatial;

use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let filter = PyModule::new(m.py(), "filter")?;
    filter::register(&filter)?;
    m.add_submodule(&filter)?;
    let spatial = PyModule::new(m.py(), "spatial")?;
    spatial::register(&spatial)?;
    m.add_submodule(&spatial)?;
    for (name, register) in [("resample", resample::register as fn(&Bound<'_, PyModule>) -> PyResult<()>), ("math", math::register), ("peaks", peaks::register), ("linalg", linalg::register)] {
        let sub = PyModule::new(m.py(), name)?;
        register(&sub)?;
        m.add_submodule(&sub)?;
    }
    pipeline::register(m)?;
    Ok(())
}
