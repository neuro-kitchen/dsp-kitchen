//! Model manifests: what a published model is and how to run it (format, tensor I/O, provenance).
//! Where its file is downloaded from, its SHA-256 and size are the `dspk-hub` catalog entry with
//! the same id.

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

/// How to run a published model: its format, tensors and provenance. The file itself (URL,
/// SHA-256, size) is the `dspk-hub` catalog entry with the same [`id`](Self::id).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Unique identifier `"<family>/<slug>"`, the id of the `dspk-hub` catalog entry.
    pub id: String,
    /// Human-readable title.
    pub name: String,
    /// Model family slug (e.g. `"dartsort"`).
    pub family: String,
    /// Pipeline task (`"detection"`, `"denoising"`, `"embedding"`, `"curation"`, …).
    pub task: String,
    /// Artifact file format.
    pub format: ModelFormat,
    #[serde(default)]
    pub description: String,
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
