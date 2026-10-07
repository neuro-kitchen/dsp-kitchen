use cubecl::prelude::Client;
use dsp_base::linalg::{PcaModel, PpcaModel};
use dsp_core::compute::{ComputeTarget, ComputeTask};
use dsp_core::{DspError, DspResult};

use crate::core::{FeatureEmbedder, SnippetBatch, WaveformSnippet};

pub mod conduction;
pub mod morphology;
pub mod wavelet;

pub use conduction::{ConductionVelocityEstimate, estimate_hdemg_conduction_velocity};
pub use morphology::{SpikeMorphology, compute_morphology};
pub use wavelet::{WaveletFeatureEmbedder, haar_dwt_multilevel_1d};

/// Components the PCA / PPCA embedders keep unless told otherwise.
pub const DEFAULT_FEATURE_COMPONENTS: usize = 4;

/// Features × spikes matrix (`matrix[f · spikes + s]`), the layout the linear models fit.
fn feature_matrix(features: usize, spikes: usize, feature_of: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
    let mut matrix = vec![0.0f32; features * spikes];
    for s in 0..spikes {
        for (f, v) in feature_of(s).into_iter().enumerate().take(features) {
            matrix[f * spikes + s] = v;
        }
    }
    matrix
}

/// Projections `[component][spike]` as one row per spike (`out[s · d + k]`).
fn per_spike(projected: &[f32], d: usize, spikes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; spikes * d];
    for s in 0..spikes {
        for k in 0..d {
            out[s * d + k] = projected[k * spikes + s];
        }
    }
    out
}

/// Fits a PCA model (`dsp_base::linalg::PcaModel`) on `client` to spike waveforms (primary trace
/// of each snippet; snippets of another length are left at zero) and projects each spike.
pub fn extract_waveform_pca(
    client: &Client,
    snippets: &[WaveformSnippet],
    num_components: usize,
) -> Option<(PcaModel, Vec<Vec<f32>>)> {
    let first = snippets.first()?;
    let (samples, spikes) = (first.num_samples, snippets.len());
    let matrix = feature_matrix(samples, spikes, |s| {
        let w = &snippets[s].waveform;
        if w.len() == samples { w.clone() } else { Vec::new() }
    });
    let pca = PcaModel::fit::<f32>(client, &matrix, samples, spikes, num_components);
    let d = pca.num_components;
    let flat = per_spike(&pca.project_cpu(&matrix, samples, spikes), d, spikes);
    let rows = flat.chunks_exact(d.max(1)).map(<[f32]>::to_vec).collect();
    Some((pca, rows))
}

/// Which linear model an embedder fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearModel {
    Pca,
    Ppca,
}

/// Fits `model` to a batch on the task's device and returns `(rows, d)`.
struct EmbedTask<'a> {
    model: LinearModel,
    matrix: &'a [f32],
    features: usize,
    spikes: usize,
    components: usize,
}

impl ComputeTask for EmbedTask<'_> {
    type Output = (Vec<f32>, usize);
    fn run(self, client: Client) -> Self::Output {
        let (projected, d) = match self.model {
            LinearModel::Pca => {
                let m = PcaModel::fit::<f32>(&client, self.matrix, self.features, self.spikes, self.components);
                (m.project_cpu(self.matrix, self.features, self.spikes), m.num_components)
            }
            LinearModel::Ppca => {
                let m = PpcaModel::fit::<f32>(&client, self.matrix, self.features, self.spikes, self.components);
                (m.project_cpu(self.matrix, self.features, self.spikes), m.num_components)
            }
        };
        (per_spike(&projected, d, self.spikes), d)
    }
}

fn embed_linear(model: LinearModel, target: ComputeTarget, components: usize, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
    let d = components.max(1);
    if batch.num_spikes == 0 {
        return Ok((Vec::new(), d));
    }
    let features = batch.num_channels * batch.num_samples;
    let matrix = feature_matrix(features, batch.num_spikes, |s| batch.snippet_slice(s).to_vec());
    let task = EmbedTask { model, matrix: &matrix, features, spikes: batch.num_spikes, components: d };
    target.run(task).map_err(|e| DspError::ComputeError(e.to_string()))
}

/// Classical PCA feature embedder delegating to [`dsp_base::linalg::PcaModel`], fitted on
/// `target`.
#[derive(Debug, Clone, Copy)]
pub struct PcaFeatureEmbedder {
    pub num_components: usize,
    pub target: ComputeTarget,
}

impl PcaFeatureEmbedder {
    /// [`DEFAULT_FEATURE_COMPONENTS`] components on `target`.
    pub fn new(target: ComputeTarget) -> Self {
        Self { num_components: DEFAULT_FEATURE_COMPONENTS, target }
    }
}

impl FeatureEmbedder for PcaFeatureEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        embed_linear(LinearModel::Pca, self.target, self.num_components, batch)
    }
}

/// Probabilistic PCA (PPCA) embedder delegating to [`dsp_base::linalg::PpcaModel`], fitted on
/// `target`.
#[derive(Debug, Clone, Copy)]
pub struct PpcaFeatureEmbedder {
    pub num_components: usize,
    pub target: ComputeTarget,
}

impl PpcaFeatureEmbedder {
    /// [`DEFAULT_FEATURE_COMPONENTS`] components on `target`.
    pub fn new(target: ComputeTarget) -> Self {
        Self { num_components: DEFAULT_FEATURE_COMPONENTS, target }
    }
}

impl FeatureEmbedder for PpcaFeatureEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        embed_linear(LinearModel::Ppca, self.target, self.num_components, batch)
    }
}
