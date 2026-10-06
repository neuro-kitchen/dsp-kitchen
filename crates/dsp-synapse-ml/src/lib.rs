//! `dsp-synapse-ml`: neuro models and sorters from the literature, each traceable to its authors.
//!
//! - [`sorters`]: spike sorters reimplemented from their papers (Kilosort4, EMUsort).
//! - [`models`]: pretrained networks run as released (parked until their artifacts are verified).
//! - [`hub`]: the catalog of published artifacts (with feature `hub`, downloads through
//!   `dsp-synapse-hub`).
//! - [`provenance`]: paper, code, license and download source of every sorter and model.
//! - [`runtime`]: tensor execution for the pretrained models.

pub mod hub;
pub mod models;
pub mod provenance;
pub mod runtime;
pub mod sorters;

pub use provenance::{ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

pub use hub::{
    ArraySpec, ModelCatalog, ModelFormat, ModelManifest, SafetensorEntryHeader, SafetensorsMap,
    TensorIoSpec, TensorPortSpec,
};
#[cfg(feature = "hub")]
pub use hub::{ModelHub, ModelHubEntry};
pub use models::{
    DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder, DartsortWaveformDenoiser,
    SPIKENET2_IED_MODEL_ID, SpikeNet2Detector, UNITREFINE_CURATION_MODEL_ID, UnitCurationResult,
    UnitRefineClassifier,
};
pub use runtime::{OnnxRuntimeSession, RuntimeTensor, burn_conv1d, burn_linear_2d, validate_tensor_port};
pub use sorters::{
    Emusort, EmusortConfig, EmusortResult, EmusortRunner, Kilosort4, Kilosort4Config,
    Kilosort4Result, Kilosort4Runner,
};
