//! Writes the `.pyi` stubs of `dspk._native` into `bindings/python/python/dspk/_native/` from the
//! annotated bindings. Run after changing a binding (the test `tests/test_stubs.py` fails when the
//! stubs and the extension disagree):
//!
//! ```text
//! cargo run --bin stub_gen
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    _native::stub_info()?.generate()?;
    Ok(())
}
