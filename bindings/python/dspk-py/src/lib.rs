//! `dspk._native`: the one extension module behind the `dspk` Python package. Binding crates add
//! their submodules here; one extension keeps one CubeCL device per runtime, so device buffers move
//! between modules without a copy through the host.

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dspk_py_core::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
