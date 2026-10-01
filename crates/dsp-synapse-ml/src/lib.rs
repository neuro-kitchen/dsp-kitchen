//! `dsp-synapse-ml` (`synapseml`): Pretrained model hub and neural inference engine
//! for electrophysiology spike sorting, denoising, latent embeddings, and automated curation.

pub mod hub;
pub mod models;
pub mod runtime;

pub use hub::{
    HubCache, ModelCatalog, ModelFormat, ModelHub, ModelHubEntry, ModelManifest, ModelPresetConfig,
    ModelStatus, ModelVerifyReport, NpyTensorF32, ProbePreset, PyTorchRemapRule,
    PyTorchWeightAdapter, SafetensorEntryHeader, SafetensorsMap, TensorIoSpec, TensorPortSpec,
    WeightTransform, transpose_2d_slice,
};
pub use models::{
    DARTSORT_DENOISER_MODEL_ID, DARTSORT_VAE_MODEL_ID, DartsortVaeEmbedder,
    DartsortWaveformDenoiser, KILOSORT4_BASIS_MODEL_ID, KILOSORT4_TEMPLATES_MODEL_ID,
    Kilosort4BasisEmbedder, Kilosort4Detector, Kilosort4TemplateMatcher, SPIKENET2_IED_MODEL_ID,
    SpikeNet2Detector, UNITREFINE_BOMBCELL_MODEL_ID, UNITREFINE_CURATION_MODEL_ID,
    UnitCurationResult, UnitRefineClassifier,
};
pub use runtime::{
    ComputeError, ComputeTarget, ComputeTask, LaunchGeometry, OnnxRuntimeSession, RuntimeTensor,
    burn_conv1d, burn_linear_2d, default_compute_target, validate_tensor_port,
};



