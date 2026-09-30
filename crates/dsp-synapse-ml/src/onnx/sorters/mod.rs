//! Pre-configured I/O, normalization, and PyTorch weight-remapping profiles for external
//! spike-sorting ecosystems (**Kilosort4**, **DARTsort**, **CEBRA**, **Bombcell / UnitMatch**).

pub mod bombcell;
pub mod cebra;
pub mod dartsort;
pub mod kilosort4;

use serde::{Deserialize, Serialize};
use crate::onnx::adapters::{OnnxNormalization, OnnxSnippetLayout};

pub use bombcell::BombcellProfile;
pub use cebra::CebraProfile;
pub use dartsort::DartsortProfile;
pub use kilosort4::Kilosort4Profile;

/// External spike-sorting ecosystem family bridged via Burn-ONNX (`onnx-ir = "0.21.0"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalSorterFamily {
    /// Kilosort4 (Pachitariu et al. 2024) — template projection & learned temporal PCs.
    Kilosort4,
    /// DARTsort (Boussard, Varol, Windolf, Paninski et al.) — pretrained denoiser, VAE & localizer.
    Dartsort,
    /// CEBRA (Schneider, Lee, Mathis 2023) — contrastive latent hypersphere embeddings.
    Cebra,
    /// Bombcell / UnitMatch (Fabre et al. / van Beest et al.) — automated unit quality curation.
    Bombcell,
}

/// Canonical I/O and tensor-normalization metadata for an external spike sorter model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalSorterProfile {
    pub family: ExternalSorterFamily,
    pub name: String,
    pub num_channels: usize,
    pub num_samples: usize,
    pub embedding_dim: usize,
    pub snippet_layout: OnnxSnippetLayout,
    pub normalization: OnnxNormalization,
    pub l2_normalize_embeddings: bool,
}
