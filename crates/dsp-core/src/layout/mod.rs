pub mod site;
pub mod sensor;

pub use site::{Position3D, SensorSite};
pub use sensor::SensorLayout;

pub type ProbeLayout = SensorLayout;
pub type ContactPosition = Position3D;
pub type ChannelContact = SensorSite;
