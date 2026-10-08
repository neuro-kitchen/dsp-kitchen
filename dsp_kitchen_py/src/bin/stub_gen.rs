//! Writes `dsp_kitchen_bindings/dsp_kitchen_bindings.pyi` (the native module's stubs) from the
//! annotated bindings. Run after changing a binding:
//!
//! ```text
//! cargo run -p dsp_kitchen_py --bin stub_gen
//! ```
//!
//! `pyo3-stub-gen` writes a mixed Rust / Python project's stubs as `<module>/__init__.pyi`; the
//! stub is moved next to the compiled module as the single file `<module>.pyi`, the standard place
//! editors and type checkers look.

use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dsp_kitchen_bindings::stub_info()?.generate()?;
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("dsp_kitchen_bindings");
    let generated = package.join("dsp_kitchen_bindings");
    fs::rename(generated.join("__init__.pyi"), package.join("dsp_kitchen_bindings.pyi"))?;
    fs::remove_dir(&generated)?;
    Ok(())
}
