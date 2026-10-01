pub mod car;
pub mod whitening;

pub use car::{direct_car_kernel, subtract_common_average_kernel};
pub use whitening::spatial_matrix_multiply_kernel;
