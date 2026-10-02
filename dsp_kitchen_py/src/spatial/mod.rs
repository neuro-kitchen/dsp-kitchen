pub mod car;
pub mod laplacian;
pub mod whitening;

pub use car::{common_average_reference, PyCommonAverageReference};
pub use laplacian::PySurfaceLaplacian;
pub use whitening::PySpatialWhitening;
