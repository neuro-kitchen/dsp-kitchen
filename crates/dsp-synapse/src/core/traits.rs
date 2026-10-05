//! Unified polymorphic trait contracts for classical DSP (`dsp-synapse`) and
//! neural network inference (`dsp-synapse-ml`).
//!
//! Every stage returns a [`DspResult`]: learned models can fail at run time (an ONNX graph with an
//! unexpected input shape, an unsupported operator) and callers, including Python, get the error.

use dsp_core::DspResult;
use dsp_io::neuro::probe::SensorLayout;
use super::events::{MatchedSpike, SpikeEvent};
use super::snippets::SnippetBatch;
use super::template::WaveformTemplate;

/// Polymorphic contract for Stage 1: Spike Detection (Classical Threshold/NEO vs. 1D-CNN/YASS).
pub trait SpikeDetector: Send + Sync {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> DspResult<Vec<SpikeEvent>>;
}

/// Polymorphic contract for Stage 2: Waveform Denoising (Identity/Filter vs. 1D-UNet/Autoencoder).
pub trait WaveformDenoiser: Send + Sync {
    fn denoise(&self, batch: &SnippetBatch) -> DspResult<SnippetBatch>;
}

/// Polymorphic contract for Stage 3: Dimensionality Reduction & Latent Embedding (PCA vs. Conv-AE/VAE/SimCLR).
pub trait FeatureEmbedder: Send + Sync {
    /// Projects `[N, K, T]` snippets into a flat `[N, D]` embedding matrix and returns `(flat_embeddings, embedding_dim)`.
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)>;
}

/// Polymorphic contract for Stage 4: 3D Physical Source Localization (CoM/Monopolar vs. Dipole MLP).
pub trait PeakLocalizer: Send + Sync {
    /// Estimates 3D physical coordinates `[x_um, y_um, z_um]` for each spike in `batch`.
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> DspResult<Vec<[f32; 3]>>;
}

/// Polymorphic contract for Stage 5: Template Matching & Collision Deconvolution (matching pursuit or learned separators).
/// Template matching on a compute device (the caller chooses it; see `dsp_core::compute`).
pub trait SpikeMatcher: Send + Sync {
    fn match_spikes<R: cubecl::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        data: &[f32],
        channels: usize,
        samples: usize,
        templates: &[WaveformTemplate],
    ) -> DspResult<Vec<MatchedSpike>>;
}
