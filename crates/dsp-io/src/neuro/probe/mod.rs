//! Physical probe geometry: contact positions, shanks, and standard probe presets.

pub mod site;
pub mod sensor;
mod source;

pub use site::{Position3D, SensorSite};
pub use sensor::SensorLayout;
pub use source::{probe_of, ProbeSource};
