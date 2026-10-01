pub mod bands;
pub mod events;
pub mod snippets;
pub mod template;
pub mod traits;

pub use bands::NeuralBand;
pub use events::{DeduplicatedSpike, MatchedSpike, SpikeEvent};
pub use snippets::{SnippetBatch, WaveformSnippet};
pub use template::{UnitQualityLabel, WaveformTemplate, compute_mean_template};
pub use traits::{
    FeatureEmbedder, PeakLocalizer, SpikeDetector, SpikeMatcher, WaveformDenoiser,
};
