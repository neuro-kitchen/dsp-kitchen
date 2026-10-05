//! Model Hub catalog loader supporting the embedded JSON catalog (`catalog/models.json`)
//! and optional user catalog overrides.

use dsp_core::{DspError, DspResult as Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::manifest::ModelManifest;

/// Embedded default JSON catalog shipped in `crates/dsp-synapse-ml/catalog/models.json`.
pub const EMBEDDED_MODELS_JSON: &str = include_str!("../../catalog/models.json");

/// Top-level JSON catalog document structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogDocument {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub models: Vec<ModelManifest>,
}

fn default_schema_version() -> u32 {
    1
}

/// In-memory catalog of pretrained electrophysiology model manifests.
#[derive(Debug, Clone)]
pub struct ModelCatalog {
    models: Vec<ModelManifest>,
}

impl ModelCatalog {
    /// Loads the embedded default catalog compiled into `dsp-synapse-ml`, plus any override
    /// file pointed to by `DSP_KITCHEN_HUB_CATALOG` if set.
    pub fn load_default() -> Result<Self> {
        let mut catalog = Self::from_json_str(EMBEDDED_MODELS_JSON)
            .map_err(|e| DspError::InvalidConfig(format!("embedded catalog/models.json: {e}")))?;

        if let Ok(custom_path) = std::env::var("DSP_KITCHEN_HUB_CATALOG") {
            let trimmed = custom_path.trim();
            if !trimmed.is_empty() {
                let p = Path::new(trimmed);
                if p.exists() {
                    catalog.merge_file(p)?;
                }
            }
        }

        Ok(catalog)
    }

    /// Parses a [`ModelCatalog`] from a JSON string (supports either `{ "models": [...] }` or a bare `[...]` array).
    pub fn from_json_str(json: &str) -> Result<Self> {
        if let Ok(doc) = serde_json::from_str::<CatalogDocument>(json) {
            return Ok(Self { models: doc.models });
        }
        let models: Vec<ModelManifest> =
            serde_json::from_str(json).map_err(|e| DspError::InvalidConfig(format!("invalid model catalog JSON: {e}")))?;
        Ok(Self { models })
    }

    /// Merges or overrides model manifests from a user JSON catalog file on disk.
    /// Models with matching `id` replace existing entries; new IDs are appended.
    pub fn merge_file(&mut self, path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DspError::Io(format!("user catalog override {}: {e}", path.display())))?;
        let other = Self::from_json_str(&content)
            .map_err(|e| DspError::InvalidConfig(format!("user catalog override {}: {e}", path.display())))?;
        self.merge(other);
        Ok(())
    }

    /// Merges another [`ModelCatalog`] into `self`, overriding entries with identical `id`s.
    pub fn merge(&mut self, other: ModelCatalog) {
        for incoming in other.models {
            if let Some(existing) = self.models.iter_mut().find(|m| m.id == incoming.id) {
                *existing = incoming;
            } else {
                self.models.push(incoming);
            }
        }
    }

    /// Returns all registered model manifests in catalog order.
    pub fn models(&self) -> &[ModelManifest] {
        &self.models
    }

    /// Looks up a model manifest by exact `id` (case-insensitive fallback).
    pub fn get(&self, id: &str) -> Option<&ModelManifest> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .or_else(|| self.models.iter().find(|m| m.id.eq_ignore_ascii_case(id)))
    }

    /// Filters manifests by model family (case-insensitive).
    pub fn by_family(&self, family: &str) -> Vec<&ModelManifest> {
        self.models
            .iter()
            .filter(|m| m.family.eq_ignore_ascii_case(family))
            .collect()
    }
}
