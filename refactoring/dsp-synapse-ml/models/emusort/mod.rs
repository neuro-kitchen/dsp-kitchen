//! EMUsort and Myomatrix specialized spike-sorting engines for muscle electrophysiology.
//!
//! Provides:
//! - [`EmusortSortConfig`]: Presets for 8-channel threads, 32-channel, and 64-channel arrays.
//! - [`EmusortTemplateMatcher`] / [`EmusortDetector`]: 150-sample universal MUAP matched filtering.
//! - [`EmusortBasisEmbedder`]: 150-sample, 12-component spatiotemporal muscle basis projection.
//! - [`EmusortLatencyAligner`]: Cross-channel conduction velocity delay estimation and alignment.

pub mod basis;
pub mod config;
pub mod latency;
pub mod matcher;

pub use basis::{
    DEFAULT_MUAP_BASIS_COMPONENTS, DEFAULT_MUAP_BASIS_WINDOW_LEN, EmusortBasisEmbedder, MyomatrixBasisEmbedder,
};
pub use config::{EmusortProbeKind, EmusortSortConfig, MyomatrixProbeKind, MyomatrixSortConfig};
pub use latency::{EmusortLatencyAligner, MyomatrixLatencyAligner};
pub use matcher::{
    DEFAULT_MUAP_WINDOW_LEN, EmusortDetector, EmusortTemplateMatcher,
    MyomatrixDetector, MyomatrixTemplateMatcher,
};
