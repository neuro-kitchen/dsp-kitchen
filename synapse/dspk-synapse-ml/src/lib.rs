//! `dspk-synapse-ml`: pretrained neuro models from the literature, each traceable to its authors.
//! The spike sorters live in their own crates, one per upstream project, each under its upstream's
//! licence (`dspk-sorter-kilosort`, `dspk-sorter-ms5`, `dspk-sorter-si`).
//!
//! - [`models`]: pretrained networks run as released (parked until their artifacts are verified).
//! - [`manifest`]: how to run each published model (format, tensors, provenance); with feature
//!   `hub` its file is downloaded through the `dspk-hub` catalog.
//! - [`provenance`]: paper, code, license and download source of every sorter and model
//!   (from `dspk-synapse`).
//! - [`runtime`]: tensor execution for the pretrained models.

pub mod manifest;
pub mod models;
pub mod runtime;

// Provenance lives in dspk-synapse (shared by the models here and the sorter crates)
pub use dspk_synapse::provenance;
pub use provenance::{ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

pub use manifest::{
    ArraySpec, ModelCatalog, ModelFormat, ModelManifest, SafetensorEntryHeader, SafetensorsMap,
    TensorIoSpec, TensorPortSpec,
};
pub use models::{
    DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder, DartsortWaveformDenoiser,
    SPIKENET2_IED_MODEL_ID, SpikeNet2Detector, UNITREFINE_CURATION_MODEL_ID, UnitCurationResult,
    UnitRefineClassifier,
};
pub use runtime::{OnnxRuntimeSession, RuntimeTensor, burn_conv1d, burn_linear_2d, validate_tensor_port};
