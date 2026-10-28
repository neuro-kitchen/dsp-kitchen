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

pyo3_stub_gen::module_variable!("dspk._native", "__version__", String);

/// Stub information for `src/bin/stub_gen.rs`, read from `bindings/python/pyproject.toml` (one
/// level above this crate: module name, Python source folder).
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let project = manifest_dir.parent().expect("dspk-py sits in bindings/python");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(project.join("pyproject.toml"))
}
