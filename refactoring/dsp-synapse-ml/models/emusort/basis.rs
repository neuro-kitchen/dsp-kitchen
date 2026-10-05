//! EMUsort / Myomatrix 12-PC spatiotemporal muscle basis embedder (`wPCA_EMG`).

use std::path::{Path, PathBuf};

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{FeatureEmbedder, SnippetBatch};

use crate::hub::NpyTensorF32;
use crate::runtime::{
    ProjectBasisTask, ReconstructBasisTask, default_compute_target, run_on_target,
};

pub const DEFAULT_MUAP_BASIS_COMPONENTS: usize = 12;
pub const DEFAULT_MUAP_BASIS_WINDOW_LEN: usize = 150;

/// EMUsort temporal muscle basis loaded from a `.npy` file (EMUsort learns it per recording; see
/// `refactoring/dsp-synapse-ml/REVIEW.md`).
#[derive(Debug, Clone)]
pub struct EmusortBasisEmbedder {
    /// Row-major `[num_components, window_len]` orthonormal temporal basis.
    basis: Vec<f32>,
    num_components: usize,
    window_len: usize,
    source_path: Option<PathBuf>,
    target: ComputeTarget,
}

impl EmusortBasisEmbedder {
    /// Loads a Myomatrix basis directly from a `.npy` file on disk.
    pub fn from_npy(path: impl AsRef<Path>, target: Option<ComputeTarget>) -> DspResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let npy = NpyTensorF32::from_file(&path_buf)
            .map_err(|e| DspError::Model(e.to_string()))?;
        Self::from_tensor(npy, Some(path_buf), target)
    }

    fn from_tensor(
        npy: NpyTensorF32,
        source_path: Option<PathBuf>,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        if npy.shape.len() != 2 || npy.shape[0] == 0 || npy.shape[1] == 0 {
            return Err(DspError::Model(format!(
                "MyomatrixBasisEmbedder expects non-empty 2D [num_components, window_len] .npy, got {:?}",
                npy.shape
            )));
        }

        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };

        let num_components = npy.shape[0];
        let window_len = npy.shape[1];

        Ok(Self {
            basis: npy.data,
            num_components,
            window_len,
            source_path,
            target: compute_target,
        })
    }

    pub fn num_components(&self) -> usize {
        self.num_components
    }

    pub fn window_len(&self) -> usize {
        self.window_len
    }

    pub fn basis(&self) -> &[f32] {
        &self.basis
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Projects 3D `[num_snippets, channels, window_len]` waveforms directly into `[num_snippets, channels * num_components]` features.
    pub fn project_raw(
        &self,
        waveforms: &[f32],
        num_snippets: usize,
        channels: usize,
        window_len: usize,
    ) -> DspResult<Vec<f32>> {
        if window_len != self.window_len {
            return Err(DspError::ShapeMismatch {
                expected: vec![self.window_len],
                actual: vec![window_len],
            });
        }
        if waveforms.len() != num_snippets * channels * window_len {
            return Err(DspError::ShapeMismatch {
                expected: vec![num_snippets, channels, window_len],
                actual: vec![waveforms.len()],
            });
        }

        run_on_target(
            self.target,
            ProjectBasisTask {
                snippets: waveforms,
                num_spikes: num_snippets,
                num_channels: channels,
                window_len,
                basis: &self.basis,
                num_components: self.num_components,
            },
        )
    }

    /// Projects and reconstructs a [`SnippetBatch`] (`W^T * W * x`) to evaluate basis compression fidelity.
    pub fn reconstruct(&self, batch: &SnippetBatch) -> DspResult<SnippetBatch> {
        let [n, k, t] = batch.shape();
        if t != self.window_len {
            return Err(DspError::Model(format!(
                "MyomatrixBasisEmbedder expects snippet window_len {}, got {}",
                self.window_len, t
            )));
        }
        let (coeffs, _) = self.embed(batch)?;
        let data = run_on_target(
            self.target,
            ReconstructBasisTask {
                coeffs: &coeffs,
                num_spikes: n,
                num_channels: k,
                num_components: self.num_components,
                basis: &self.basis,
                window_len: self.window_len,
            },
        )?;

        Ok(SnippetBatch {
            data,
            num_spikes: n,
            num_channels: k,
            num_samples: t,
            primary_channels: batch.primary_channels.clone(),
            center_samples: batch.center_samples.clone(),
            subsample_offsets: batch.subsample_offsets.clone(),
            channel_ids: batch.channel_ids.clone(),
        })
    }
}

impl FeatureEmbedder for EmusortBasisEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        let [n, k, t] = batch.shape();
        if t != self.window_len {
            return Err(DspError::Model(format!(
                "EmusortBasisEmbedder expects snippet window_len {}, got {}",
                self.window_len, t
            )));
        }
        let embed_dim = k * self.num_components;
        if n == 0 {
            return Ok((Vec::new(), embed_dim));
        }

        let coeffs = run_on_target(
            self.target,
            ProjectBasisTask {
                snippets: &batch.data,
                num_spikes: n,
                num_channels: k,
                window_len: self.window_len,
                basis: &self.basis,
                num_components: self.num_components,
            },
        )?;

        Ok((coeffs, embed_dim))
    }
}

/// Backward-compatible type alias for [`EmusortBasisEmbedder`].
pub type MyomatrixBasisEmbedder = EmusortBasisEmbedder;
