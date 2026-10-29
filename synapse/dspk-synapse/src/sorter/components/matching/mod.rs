//! Template matching (peelers): SpyKING CIRCUS 2's circus-omp and Tridesclous 2's tdc peeler.

pub mod kernels;
pub mod omp;
pub mod tdc_peeler;

pub use omp::{CircusOmp, OmpOptions, OmpSpike};
pub use tdc_peeler::{fine_prototype, FineFilter, TdcPeeler, TdcPeelerOptions, TdcSpike};
