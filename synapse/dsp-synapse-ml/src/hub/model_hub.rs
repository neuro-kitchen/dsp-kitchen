//! [`ModelHub`]: the catalog of this crate plus `dsp-synapse-hub` downloads and cache.

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use dsp_synapse_hub::{ArtifactReport, ArtifactStatus, Hub};

use super::catalog::ModelCatalog;
use super::manifest::ModelManifest;

/// A catalog entry with its local state.
#[derive(Debug, Clone)]
pub struct ModelHubEntry {
    pub manifest: ModelManifest,
    pub status: ArtifactStatus,
    pub resolved_download_url: String,
    pub local_path: PathBuf,
}

/// Catalog lookups backed by a [`Hub`].
#[derive(Debug, Clone)]
pub struct ModelHub {
    catalog: ModelCatalog,
    hub: Hub,
}

impl ModelHub {
    /// Embedded catalog (plus `DSP_KITCHEN_HUB_CATALOG`) and the default cache.
    pub fn new() -> Result<Self> {
        Ok(Self { catalog: ModelCatalog::load_default()?, hub: Hub::from_env_or_default() })
    }

    pub fn with_catalog_and_hub(catalog: ModelCatalog, hub: Hub) -> Self {
        Self { catalog, hub }
    }

    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    pub fn hub(&self) -> &Hub {
        &self.hub
    }

    fn manifest(&self, model_id: &str) -> Result<&ModelManifest> {
        self.catalog.get(model_id).ok_or_else(|| anyhow!("unknown model id '{model_id}' in the dsp-synapse-ml catalog"))
    }

    fn entry_for(&self, manifest: &ModelManifest) -> ModelHubEntry {
        let artifact = manifest.artifact();
        ModelHubEntry {
            resolved_download_url: dsp_synapse_hub::resolve_weights_uri(&artifact.url).map(|(_, u)| u).unwrap_or_else(|_| artifact.url.clone()),
            status: self.hub.status(&artifact),
            local_path: self.hub.path(&artifact),
            manifest: manifest.clone(),
        }
    }

    pub fn list(&self) -> Vec<ModelHubEntry> {
        self.catalog.models().iter().map(|m| self.entry_for(m)).collect()
    }

    pub fn info(&self, model_id: &str) -> Result<ModelHubEntry> {
        Ok(self.entry_for(self.manifest(model_id)?))
    }

    /// Downloads and verifies `model_id`; returns its manifest and local path.
    pub fn pull(&self, model_id: &str, force: bool) -> Result<(ModelManifest, PathBuf)> {
        let manifest = self.manifest(model_id)?;
        Ok((manifest.clone(), self.hub.pull(&manifest.artifact(), force)?))
    }

    pub fn verify(&self, model_id: &str, check_link: bool) -> Result<ArtifactReport> {
        self.hub.verify(&self.manifest(model_id)?.artifact(), check_link)
    }

    pub fn verify_all(&self, check_link: bool) -> Result<Vec<ArtifactReport>> {
        self.catalog.models().iter().map(|m| self.hub.verify(&m.artifact(), check_link)).collect()
    }

    pub fn remove(&self, model_id: &str) -> Result<bool> {
        self.hub.remove(&self.manifest(model_id)?.artifact())
    }

    pub fn clean(&self) -> Result<u64> {
        self.hub.clean()
    }
}
