pub mod device;
pub mod kernels;
pub mod subtraction;
#[cfg(test)]
mod reference;

pub use subtraction::{TemplateFilter, DEFAULT_MAX_LAG, MIN_TEMPLATE_ENERGY};
