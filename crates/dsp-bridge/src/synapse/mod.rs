pub mod probe;
pub mod detection;

pub use probe::PyProbeLayout;
pub use detection::{PySpikeEvent, detect_spikes, estimate_noise};
