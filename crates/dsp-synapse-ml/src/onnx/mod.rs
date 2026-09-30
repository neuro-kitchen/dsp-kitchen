//! Burn-ONNX (`onnx-ir = "0.21.0"`) external sorter bridge for `dsp-synapse-ml`.
//!
//! Enables loading and running standard `.onnx` models exported from **Kilosort4**,
//! **DARTsort**, **CEBRA**, and **Bombcell / UnitMatch** directly inside the Rust
//! `dsp-synapse` pipeline via polymorphic trait adapters.

pub mod adapters;
pub mod ir_runner;
pub mod sorters;

pub use adapters::{
    OnnxFeatureEmbedder, OnnxNormalization, OnnxPeakLocalizer, OnnxSnippetLayout,
    OnnxSpikeDetector, OnnxUnitCurator, OnnxWaveformDenoiser, prepare_snippet_tensor,
};
pub use ir_runner::{DynTensor, OnnxGraphRunner, OnnxPortSpec, onnx_proto_builder};
pub use sorters::{
    BombcellProfile, CebraProfile, DartsortProfile, ExternalSorterFamily, ExternalSorterProfile,
    Kilosort4Profile,
};
