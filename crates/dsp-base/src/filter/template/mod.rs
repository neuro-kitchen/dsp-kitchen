pub mod device;
pub mod kernels;
pub mod subtraction;

pub use subtraction::{TemplateFilter, subtract_template_1d, subtract_template_multichannel};
