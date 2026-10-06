pub mod car;
pub mod laplacian;
pub mod whitening;

pub use car::PyCommonAverageReference;
pub use laplacian::PySurfaceLaplacian;
pub use whitening::PySpatialWhitening;

/// Adds the spatial classes and functions to the (flat) native module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    car::register(m)?;
    laplacian::register(m)?;
    whitening::register(m)
}
