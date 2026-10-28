//! Declarative model manifest, tensor I/O specification, and local status types for the Model Hub.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::provenance::Provenance;

/// Serialization format of a pretrained model weight artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// Open Neural Network Exchange (`.onnx`) graph + weights.
    Onnx,
    /// HuggingFace Safetensors (`.safetensors`) zero-copy tensor map.
    Safetensors,
    /// NumPy array binary (`.npy`).
    Npy,
    /// NumPy zip of named arrays (`.npz`), e.g. Kilosort4 `wTEMP.npz` (`wPCA`, `wTEMP`).
    Npz,
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
            Self::Npz => "npz",
            Self::Pt => "pt",
        }
    }

    /// Parse a format string case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "onnx" => Some(Self::Onnx),
            "safetensors" => Some(Self::Safetensors),
            "npy" => Some(Self::Npy),
            "npz" => Some(Self::Npz),
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

/// One array stored in a multi-array artifact (e.g. `wPCA` in Kilosort4's `wTEMP.npz`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArraySpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Catalog entry of a published artifact: what to download, how to verify it, what it holds, and
/// whom to credit ([`Provenance`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Unique identifier `"<family>/<slug>"` (e.g. `"kilosort4/wtemp-v1"`).
    pub id: String,
    /// Human-readable title.
    pub name: String,
    /// Sorter / model family slug (e.g. `"kilosort4"`).
    pub family: String,
    /// Pipeline task (`"detection"`, `"denoising"`, `"embedding"`, `"curation"`, …).
    pub task: String,
    /// Artifact file format.
    pub format: ModelFormat,
    #[serde(default)]
    pub description: String,
    /// Download URI (`hf://…`, `gh://…`, `zenodo://…`, `https://…`, or `file://…`).
    pub weights_url: String,
    /// Lowercase hex SHA-256 of the artifact (required: entries without one cannot be pulled).
    pub sha256: String,
    /// Artifact size in bytes (checked on download).
    pub size_bytes: u64,
    /// Arrays of a multi-array artifact (`.npz`).
    #[serde(default)]
    pub arrays: Vec<ArraySpec>,
    /// Tensor signature of a network artifact.
    #[serde(default)]
    pub io_spec: Option<TensorIoSpec>,
    /// Sampling rate the artifact was made for (Hz).
    #[serde(default)]
    pub sample_rate_hz: Option<f64>,
    /// Paper, code, license and download source to credit.
    pub provenance: Provenance,
}

impl ModelManifest {
    /// File name of the artifact: the published name ([`Provenance::artifacts`]), else
    /// `<slug>.<extension>`.
    pub fn file_name(&self) -> String {
        match self.provenance.artifacts.first() {
            Some(a) => a.name.clone(),
            None => format!("{}.{}", self.id.rsplit('/').next().unwrap_or(&self.id), self.format.extension()),
        }
    }

    /// What `dsp-synapse-hub` downloads and verifies for this entry.
    #[cfg(feature = "hub")]
    pub fn artifact(&self) -> dsp_synapse_hub::Artifact {
        dsp_synapse_hub::Artifact {
            id: self.id.clone(),
            family: self.family.clone(),
            file_name: self.file_name(),
            url: self.weights_url.clone(),
            sha256: self.sha256.clone(),
            size_bytes: self.size_bytes,
        }
    }
}
