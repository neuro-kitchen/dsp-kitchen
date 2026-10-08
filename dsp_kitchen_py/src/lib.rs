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

/// What `src/bin/stub_gen.rs` writes into `dsp_kitchen_bindings.pyi`: every class and function
/// annotated for `pyo3-stub-gen`, with types, defaults and docstrings. The project's
/// `pyproject.toml` (module name, Python source folder) is at the repository root, one level above
/// this crate.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let root = manifest_dir.parent().expect("dsp_kitchen_py sits in the workspace root");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(root.join("pyproject.toml"))
}
