//! `dsp_kitchen_bindings`: the native module behind the `dsp_kitchen` SDK. It is flat; the SDK's
//! Python packages (`filter`, `spatial`, `math`, `linalg`, `pipeline`, `io`, `synapse`, `runtime`)
//! arrange its names.

pub mod array;
pub mod buffer;
pub mod filter;
pub mod linalg;
pub mod math;
pub mod pipeline;
pub mod runtime;
pub mod spatial;
pub mod synapse;

use pyo3::prelude::*;

#[pymodule]
fn dsp_kitchen_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::register(m)?;
    buffer::register(m)?;
    math::register(m)?;
    filter::register(m)?;
    spatial::register(m)?;
    linalg::register(m)?;
    pipeline::register(m)?;
    synapse::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
