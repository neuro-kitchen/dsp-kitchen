//! `dsp-synapse-ml` (`synapseml`): Pretrained model hub and neural inference engine
//! for electrophysiology spike sorting, denoising, latent embeddings, and automated curation.

pub mod hub;

pub use hub::{
    HubCache, ModelCatalog, ModelFormat, ModelHub, ModelHubEntry, ModelManifest, ModelPresetConfig,
    ModelStatus, ModelVerifyReport, NpyTensorF32, ProbePreset, PyTorchRemapRule,
    PyTorchWeightAdapter, SafetensorEntryHeader, SafetensorsMap, TensorIoSpec, TensorPortSpec,
    WeightTransform, transpose_2d_slice,
};

