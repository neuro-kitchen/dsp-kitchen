//! Pretrained networks run as their authors released them (artifacts through the catalog,
//! [`crate::hub`]). None has a validated artifact yet: their catalog entries are parked until
//! verified (see `refactoring/dsp-synapse-ml/`). Sorters reimplemented from papers live in
//! [`crate::sorters`].

pub mod dartsort;
pub mod spikenet2;
pub mod unitrefine;

pub use dartsort::{DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder, DartsortWaveformDenoiser};
pub use spikenet2::{SPIKENET2_IED_MODEL_ID, SpikeNet2Detector};
pub use unitrefine::{UNITREFINE_CURATION_MODEL_ID, UnitCurationResult, UnitRefineClassifier};
