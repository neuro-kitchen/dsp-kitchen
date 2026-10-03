pub mod bands;
pub mod events;
pub mod snippets;
pub mod sorting_output;
pub mod template;
pub mod traits;

pub use bands::NeuralBand;
pub use events::{DeduplicatedSpike, MatchedSpike, SpikeEvent};
pub use snippets::{SnippetBatch, WaveformSnippet};
pub use sorting_output::{RecordingMeta, SortedUnit, SortingOutput};
pub use template::{
    DenseTemplates, TemplateAxisOrder, UnitQualityLabel, WaveformTemplate, compute_mean_template,
};
pub use traits::{
    FeatureEmbedder, PeakLocalizer, SpikeDetector, SpikeMatcher, WaveformDenoiser,
};

