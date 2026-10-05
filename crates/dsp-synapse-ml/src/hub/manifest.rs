//! Declarative model manifest, tensor I/O specification, and local status types for the Model Hub.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// Serialization format of a pretrained model weight artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// Open Neural Network Exchange (`.onnx`) graph + weights.
    Onnx,
    /// HuggingFace Safetensors (`.safetensors`) zero-copy tensor map.
    Safetensors,
    /// NumPy array binary (`.npy`), e.g. Kilosort4 temporal basis `wPCA.npy`.
    Npy,
    /// PyTorch checkpoint / TorchScript archive (`.pt` / `.pth`).
    Pt,
}

impl ModelFormat {
    /// Canonical file extension (without leading dot).
    pub fn extension(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Safetensors => "safetensors",
            Self::Npy => "npy",
            Self::Pt => "pt",
        }
    }

    /// Parse a format string case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "onnx" => Some(Self::Onnx),
            "safetensors" => Some(Self::Safetensors),
            "npy" => Some(Self::Npy),
            "pt" | "pth" => Some(Self::Pt),
            _ => None,
        }
    }
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// Local installation and integrity state of a model in the Hub cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model weights are downloaded in the local cache and match the manifest SHA-256.
    Installed,
    /// Model is registered in the catalog and available for download (`dsp-cli hub pull`).
    Available,
    /// A cached weight file exists on disk, but its size or SHA-256 does not match the manifest.
    Corrupted,
}

impl ModelStatus {
    /// Badge string formatted for CLI table output.
    pub fn badge(self) -> &'static str {
        match self {
            Self::Installed => "[INSTALLED]",
            Self::Available => "[AVAILABLE]",
            Self::Corrupted => "[CORRUPTED]",
        }
    }
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.badge())
    }
}

/// Descriptor for a single input or output tensor port of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPortSpec {
    /// Tensor port name in the graph or weight map (e.g. `"noisy_waveforms"`, `"latent_mu"`).
    pub name: String,
    /// Element data type (e.g. `"float32"`, `"int64"`).
    pub dtype: String,
    /// Expected tensor dimensions; `-1` denotes a dynamic batch or temporal axis (e.g. `[-1, 1, 121]`).
    pub shape: Vec<i64>,
    /// Optional human-readable description of the tensor's physical meaning.
    #[serde(default)]
    pub description: Option<String>,
}

impl TensorPortSpec {
    /// Formats the tensor shape like `[-1, 1, 121]`.
    pub fn format_shape(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(ToString::to_string).collect();
        format!("[{}]", dims.join(", "))
    }
}

/// Complete input/output tensor specification, expected sampling rate, and normalization policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorIoSpec {
    /// Input tensor port specifications.
    pub inputs: Vec<TensorPortSpec>,
    /// Output tensor port specifications.
    pub outputs: Vec<TensorPortSpec>,
    /// Expected signal sampling rate in Hz (e.g. `30000.0` for extracellular AP, `30.0` for calcium).
    pub sample_rate_hz: f64,
    /// Expected input normalization rule (e.g. `"z_score_per_channel"`, `"peak_abs_normalized"`, `"raw_microvolts"`).
    pub normalization: String,
}

/// Declarative metadata manifest for a published electrophysiology model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Unique canonical model identifier (`"<family>/<model-slug>"`, e.g. `"unitrefine/curation-v1"`).
    pub id: String,
    /// Human-readable model title.
    pub name: String,
    /// Model family slug (`"spikenet2"`, `"unitrefine"`, `"dartsort"`, `"kilosort4"`, `"cebra"`, `"cascade"`, `"deepinterpolation"`).
    pub family: String,
    /// Pipeline task (`"detection"`, `"denoising"`, `"embedding"`, `"localization"`, `"curation"`).
    pub task: String,
    /// Semantic version of the model weights.
    pub version: String,
    /// Weight file serialization format.
    pub format: ModelFormat,
    /// Summary of the model architecture and electrophysiology domain.
    #[serde(default)]
    pub description: String,
    /// Author's upstream source repository URL.
    pub upstream_repo: String,
    /// Peer-reviewed publication or preprint DOI / URL.
    pub paper_url: String,
    /// SPDX license identifier for the model weights.
    pub license: String,
    /// Download source URI (`hf://...`, `gh://...`, `zenodo://...`, `https://...`, or `file://...`).
    pub weights_url: String,
    /// Expected lowercase hex SHA-256 digest of the weight file.
    pub sha256: String,
    /// Expected file size in bytes.
    pub size_bytes: u64,
    /// Input/output tensor signature and preprocessing requirements.
    pub io_spec: TensorIoSpec,
}

impl ModelManifest {
    /// Canonical filename used when storing this model's weights on disk.
    pub fn artifact_filename(&self) -> String {
        let slug = self.id.split('/').next_back().unwrap_or(&self.id);
        format!("{}.{}", slug, self.format.extension())
    }
}

/// Combined view of a [`ModelManifest`] and its current local cache status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelHubEntry {
    pub manifest: ModelManifest,
    pub status: ModelStatus,
    pub resolved_download_url: String,
    pub local_weights_path: PathBuf,
}
