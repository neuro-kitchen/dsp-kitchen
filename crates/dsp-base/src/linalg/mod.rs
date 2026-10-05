pub mod covariance;
pub mod eigen;
pub mod projection;
pub mod pca;
pub mod ppca;
pub mod ica;
pub mod kernels;

pub use covariance::{covariance, covariance_of_host};
pub use eigen::{symmetric_eigen, symmetric_eigen_batched, symmetric_eigen_host, EigenOptions, SymmetricEigen};
pub use pca::PcaModel;
pub use projection::DeviceProjection;
pub use ppca::PpcaModel;
pub use ica::{FastIcaModel, IcaContrast};
