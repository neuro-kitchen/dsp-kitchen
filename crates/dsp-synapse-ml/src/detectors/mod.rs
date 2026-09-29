pub mod ensor_artifact;
pub mod spikedeeptector;
pub mod yass_detector;

pub use ensor_artifact::EnsorArtifactRejector;
pub use spikedeeptector::{SpikeClassProbabilities, SpikeDeeptector};
pub use yass_detector::YassNeuralDetector;
