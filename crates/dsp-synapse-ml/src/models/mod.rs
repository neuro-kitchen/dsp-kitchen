//! Modular pretrained model families for `dsp-synapse-ml`.
//!
//! Each subfolder corresponds 1-to-1 with a model family in `catalog/models.json`
//! and executes verified pretrained artifacts via [`crate::runtime`].

pub mod dartsort;
pub mod kilosort4;
pub mod myomatrix;
pub mod spikenet2;
pub mod unitrefine;

pub use dartsort::{
    DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder,
    DartsortWaveformDenoiser,
};
pub use kilosort4::{
    KILOSORT4_BASIS_MODEL_ID, KILOSORT4_TEMPLATES_MODEL_ID, Kilosort4BasisEmbedder,
    Kilosort4Detector, Kilosort4TemplateMatcher,
};
pub use myomatrix::{
    DEFAULT_MUAP_BASIS_COMPONENTS, DEFAULT_MUAP_BASIS_WINDOW_LEN, DEFAULT_MUAP_WINDOW_LEN,
    MYOMATRIX_BASIS_MODEL_ID, MYOMATRIX_TEMPLATES_MODEL_ID, MyomatrixBasisEmbedder,
    MyomatrixDetector, MyomatrixLatencyAligner, MyomatrixProbeKind, MyomatrixSortConfig,
    MyomatrixTemplateMatcher,
};
pub use spikenet2::{SPIKENET2_IED_MODEL_ID, SpikeNet2Detector};
pub use unitrefine::{
    UNITREFINE_BOMBCELL_MODEL_ID, UNITREFINE_CURATION_MODEL_ID, UnitCurationResult,
    UnitRefineClassifier,
};
