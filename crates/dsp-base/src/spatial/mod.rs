//! Operators that mix channels: each output channel is a weighted sum of input channels.
//!
//! - [`car`]: common average reference (subtracts the mean over channels at every sample).
//! - [`SpatialWhitening`]: ZCA whitening, global or over each channel's nearest contacts.
//! - [`SurfaceLaplacian`]: each channel minus the (distance-weighted) mean of its neighbours (HD-EMG,
//!   ECoG grids).
//! - [`sparse`]: the `[channels, channels]` matrix behind whitening and the Laplacian, applied dense
//!   or as sparse rows (CAR has its own single-pass kernel).

pub mod car;
pub mod kernels;
pub mod laplacian;
pub mod sparse;
pub mod whitening;

pub use car::{direct_car_kernel, execute_direct_car};
pub use laplacian::SurfaceLaplacian;
pub use sparse::{execute_sparse_rows_multiply, execute_spatial_matrix_multiply, DeviceSpatialMatrix, SparseRows, SPARSE_MAX_ROW_FILL};
pub use whitening::SpatialWhitening;
