pub mod bands;
pub mod events;
pub mod kernels;
pub mod snippets;
pub mod sorting_output;
pub mod template;
pub mod traits;

pub use bands::NeuralBand;
pub use events::{DeduplicatedSpike, MatchedSpike, SpikeEvent};
pub use snippets::{SnippetBatch, WaveformSnippet};
pub use sorting_output::{RecordingMeta, SortedUnit, SortingOutput};
pub use template::{
    TEMPLATE_STD_DDOF, UnitQualityLabel, UnitTemplateAccumulator, WaveformTemplate, compute_mean_template,
    dense_waveform, pack_templates, unpack_template,
};
pub use traits::{
    FeatureEmbedder, PeakLocalizer, SpikeDetector, SpikeMatcher, WaveformDenoiser,
};

