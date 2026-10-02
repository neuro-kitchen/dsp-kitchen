//! Myomatrix 12-PC spatiotemporal muscle basis embedder (`wPCA_EMG`).

use std::path::{Path, PathBuf};

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{FeatureEmbedder, SnippetBatch};

use crate::hub::NpyTensorF32;
use crate::runtime::{
    ProjectBasisTask, ReconstructBasisTask, default_compute_target, run_on_target,
};

pub const MYOMATRIX_BASIS_MODEL_ID: &str = "myomatrix/temporal-basis-150-12pc-v1";
pub const DEFAULT_MUAP_BASIS_COMPONENTS: usize = 12;
pub const DEFAULT_MUAP_BASIS_WINDOW_LEN: usize = 150;

/// Pretrained or canonical Myomatrix 12-component orthonormal temporal muscle basis.
#[derive(Debug, Clone)]
pub struct MyomatrixBasisEmbedder {
    /// Row-major `[num_components, window_len]` orthonormal temporal basis.
    basis: Vec<f32>,
    num_components: usize,
    window_len: usize,
    source_path: Option<PathBuf>,
    target: ComputeTarget,
}

impl MyomatrixBasisEmbedder {
    /// Creates a canonical 150-sample, 12-component orthonormal muscle basis without disk dependencies.
    pub fn from_canonical(target: Option<ComputeTarget>) -> DspResult<Self> {
        let num_components = DEFAULT_MUAP_BASIS_COMPONENTS;
        let window_len = DEFAULT_MUAP_BASIS_WINDOW_LEN;
        let mut basis = vec![0.0f32; num_components * window_len];

        // Construct 12 smooth spatiotemporal Hermite-Gaussian orthogonal functions
        let center = (window_len as f32) / 2.0;
        let sigma = (window_len as f32) * 0.18;

        for c in 0..num_components {
            let row = &mut basis[c * window_len..(c + 1) * window_len];
            for i in 0..window_len {
                let x = (i as f32 - center) / sigma;
                let envelope = (-0.5 * x * x).exp();
                // Hermite-like polynomial terms
                let poly = match c {
                    0 => 1.0,
                    1 => x,
                    2 => x * x - 1.0,
                    3 => x * x * x - 3.0 * x,
                    4 => x.powi(4) - 6.0 * x * x + 3.0,
                    5 => x.powi(5) - 10.0 * x.powi(3) + 15.0 * x,
                    6 => (2.0 * x).sin(),
                    7 => (2.0 * x).cos(),
                    8 => (3.0 * x).sin() * envelope,
                    9 => (3.0 * x).cos() * envelope,
                    10 => x.powi(6) - 15.0 * x.powi(4) + 45.0 * x * x - 15.0,
                    _ => (4.0 * x).sin(),
                };
                row[i] = poly * envelope;
            }
        }

        // Gram-Schmidt orthonormalization
        for i in 0..num_components {
            for j in 0..i {
                let dot: f32 = (0..window_len)
                    .map(|k| basis[i * window_len + k] * basis[j * window_len + k])
                    .sum();
                for k in 0..window_len {
                    let sub = dot * basis[j * window_len + k];
                    basis[i * window_len + k] -= sub;
                }
            }
            let norm: f32 = (0..window_len)
                .map(|k| basis[i * window_len + k].powi(2))
                .sum::<f32>()
                .sqrt();
            let inv_norm = if norm > 1e-9 { 1.0 / norm } else { 1.0 };
            for k in 0..window_len {
                basis[i * window_len + k] *= inv_norm;
            }
        }

        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };

        Ok(Self {
            basis,
            num_components,
            window_len,
            source_path: None,
            target: compute_target,
        })
    }

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

impl FeatureEmbedder for MyomatrixBasisEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        let [n, k, t] = batch.shape();
        if t != self.window_len {
            return Err(DspError::Model(format!(
                "MyomatrixBasisEmbedder expects snippet window_len {}, got {}",
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
