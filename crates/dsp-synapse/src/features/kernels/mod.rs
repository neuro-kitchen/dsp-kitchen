//! Device kernels of the feature extractors.

pub mod local_svd;

pub use local_svd::{gather_fit_waveforms_kernel, project_local_kernel, NO_CHANNEL};
