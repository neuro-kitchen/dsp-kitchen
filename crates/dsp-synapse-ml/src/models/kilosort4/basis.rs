//! Kilosort4 temporal PCA basis embedder (`kilosort4/temporal-basis-v1` / `wPCA.npy`).
//!
//! All dimensions (`num_components`, `window_len`) are read dynamically from the `.npy`
//! header and verified against the catalog manifest's [`crate::hub::TensorIoSpec`].

use std::path::{Path, PathBuf};

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{FeatureEmbedder, SnippetBatch};

use crate::hub::NpyTensorF32;
use crate::runtime::{
    ProjectBasisTask, ReconstructBasisTask, default_compute_target, pull_model, run_on_target,
};

pub const KILOSORT4_BASIS_MODEL_ID: &str = "kilosort4/temporal-basis-v1";

/// Pretrained Kilosort4 temporal PCA basis (`wPCA.npy`) executing projection and
/// reconstruction on the active [`ComputeTarget`] via CubeCL.
#[derive(Debug, Clone)]
pub struct Kilosort4BasisEmbedder {
    /// Row-major `[num_components, window_len]` orthonormal temporal basis.
    basis: Vec<f32>,
    num_components: usize,
    window_len: usize,
    source_path: PathBuf,
    target: ComputeTarget,
}

impl Kilosort4BasisEmbedder {
    /// Pulls and verifies `kilosort4/temporal-basis-v1` from [`crate::hub::ModelHub`] and binds to `target`
    /// (or [`default_compute_target`] when `None`).
    pub fn from_hub(target: Option<ComputeTarget>) -> DspResult<Self> {
        let (_manifest, path) = pull_model(KILOSORT4_BASIS_MODEL_ID)?;
        Self::from_npy(path, target)
    }

    /// Loads a Kilosort4 temporal basis directly from a `.npy` file on disk.
    pub fn from_npy(path: impl AsRef<Path>, target: Option<ComputeTarget>) -> DspResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let npy = NpyTensorF32::from_file(&path_buf)
            .map_err(|e| DspError::Model(e.to_string()))?;
        Self::from_tensor(npy, path_buf, target)
    }

    fn from_tensor(
        npy: NpyTensorF32,
        source_path: PathBuf,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        if npy.shape.len() != 2 || npy.shape[0] == 0 || npy.shape[1] == 0 {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder expects non-empty 2D [num_components, window_len] .npy, got {:?}",
                npy.shape
            )));
        }
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };

        Ok(Self {
            num_components: npy.shape[0],
            window_len: npy.shape[1],
            basis: npy.data,
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

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    /// Projects a 2D batch `[num_waveforms, window_len]` onto `[num_waveforms, num_components]`.
    pub fn project_2d(
        &self,
        waveforms: &[f32],
        num_waveforms: usize,
        window_len: usize,
    ) -> DspResult<Vec<f32>> {
        if window_len != self.window_len {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder expects window_len {}, got {}",
                self.window_len, window_len
            )));
        }
        if waveforms.len() != num_waveforms * window_len {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder input length {} != num_waveforms ({num_waveforms}) * window_len ({window_len})",
                waveforms.len()
            )));
        }

        run_on_target(
            self.target,
            ProjectBasisTask {
                snippets: waveforms,
                num_spikes: num_waveforms,
                num_channels: 1,
                window_len: self.window_len,
                basis: &self.basis,
                num_components: self.num_components,
            },
        )
    }

    /// Reconstructs `[num_waveforms, window_len]` waveforms from `[num_waveforms, num_components]` coefficients.
    pub fn reconstruct_2d(
        &self,
        coeffs: &[f32],
        num_waveforms: usize,
        num_components: usize,
    ) -> DspResult<Vec<f32>> {
        if num_components != self.num_components {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder expects num_components {}, got {}",
                self.num_components, num_components
            )));
        }
        if coeffs.len() != num_waveforms * num_components {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder coeffs length {} != num_waveforms ({num_waveforms}) * num_components ({num_components})",
                coeffs.len()
            )));
        }

        run_on_target(
            self.target,
            ReconstructBasisTask {
                coeffs,
                num_spikes: num_waveforms,
                num_channels: 1,
                num_components: self.num_components,
                basis: &self.basis,
                window_len: self.window_len,
            },
        )
    }

    /// Projects and reconstructs a [`SnippetBatch`] (`W^T * W * x`) to evaluate basis compression fidelity.
    pub fn reconstruct(&self, batch: &SnippetBatch) -> DspResult<SnippetBatch> {
        let [n, k, t] = batch.shape();
        if t != self.window_len {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder expects snippet window_len {}, got {}",
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

impl FeatureEmbedder for Kilosort4BasisEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        let [n, k, t] = batch.shape();
        if t != self.window_len {
            return Err(DspError::Model(format!(
                "Kilosort4BasisEmbedder expects snippet window_len {}, got {}",
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
