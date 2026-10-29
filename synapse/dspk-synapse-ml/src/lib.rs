//! `dspk-synapse-ml`: pretrained neuro models from the literature, each traceable to its authors.
//! The spike sorters live in their own crates (`synapse/sorters/`: `dspk-ks4`, `dspk-ms5`,
//! `dspk-sc2`, `dspk-tdc2`), each under its upstream's licence.
//!
//! - [`models`]: pretrained networks run as released (parked until their artifacts are verified).
//! - [`hub`]: the catalog of published artifacts (with feature `hub`, downloads through
//!   `dspk-synapse-hub`).
//! - [`provenance`]: paper, code, license and download source of every sorter and model
//!   (from `dspk-synapse`).
//! - [`runtime`]: tensor execution for the pretrained models.

pub mod hub;
pub mod models;
pub mod runtime;

// Provenance lives in dspk-synapse (shared by the models here and the sorter crates)
pub use dspk_synapse::provenance;
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
