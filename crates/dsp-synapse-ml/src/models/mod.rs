//! Modular pretrained model families for `dsp-synapse-ml`.
//!
//! Each subfolder corresponds 1-to-1 with a model family in `catalog/models.json`
//! and executes verified pretrained artifacts via [`crate::runtime`].

pub mod dartsort;
pub mod emusort;
pub mod kilosort4;
pub mod spikenet2;
pub mod unitrefine;

// Backward-compatible module alias
pub use emusort as myomatrix;

pub use dartsort::{
    DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder,
    DartsortWaveformDenoiser,
};
pub use emusort::{
    DEFAULT_MUAP_BASIS_COMPONENTS, DEFAULT_MUAP_BASIS_WINDOW_LEN, DEFAULT_MUAP_WINDOW_LEN,
    EMUSORT_BASIS_MODEL_ID, EMUSORT_TEMPLATES_MODEL_ID, EmusortBasisEmbedder,
    EmusortDetector, EmusortLatencyAligner, EmusortProbeKind, EmusortSortConfig,
    EmusortTemplateMatcher, MYOMATRIX_BASIS_MODEL_ID, MYOMATRIX_TEMPLATES_MODEL_ID,
    MyomatrixBasisEmbedder, MyomatrixDetector, MyomatrixLatencyAligner, MyomatrixProbeKind,
    MyomatrixSortConfig, MyomatrixTemplateMatcher,
};
pub use kilosort4::{
    KILOSORT4_BASIS_MODEL_ID, KILOSORT4_TEMPLATES_MODEL_ID, Kilosort4BasisEmbedder,
    Kilosort4Detector, Kilosort4TemplateMatcher,
};
pub use spikenet2::{SPIKENET2_IED_MODEL_ID, SpikeNet2Detector};
pub use unitrefine::{
    UNITREFINE_BOMBCELL_MODEL_ID, UNITREFINE_CURATION_MODEL_ID, UnitCurationResult,
    UnitRefineClassifier,
};
