//! Declarative Model Hub Manager and weight loader utilities for `dsp-synapse-ml`.
//!
//! Tracks published electrophysiology models via human-readable JSON manifests (`catalog/models.json`)
//! without embedding binary `.onnx` or `.safetensors` weights in Git, and manages local caching
//! under `~/.cache/dsp-kitchen/hub/` (or `DSP_KITCHEN_HUB_DIR`).

pub mod cache;
pub mod catalog;
pub mod downloader;
pub mod manifest;
pub mod providers;
pub mod pytorch_remap;
pub mod registry;
pub mod safetensors;

pub use cache::{CachedModelRecord, HubCache, HubCacheIndex, compute_file_sha256};
pub use catalog::{CatalogDocument, EMBEDDED_MODELS_JSON, ModelCatalog};
pub use downloader::{DownloadOutcome, RemoteLinkCheck, check_remote_link, download_and_verify};
pub use manifest::{
    ModelFormat, ModelHubEntry, ModelManifest, ModelStatus, TensorIoSpec, TensorPortSpec,
};
pub use providers::{
    ProviderKind, resolve_gh_uri, resolve_hf_uri, resolve_weights_uri, resolve_zenodo_uri,
};
pub use pytorch_remap::{
    PyTorchRemapRule, PyTorchWeightAdapter, WeightTransform, flip_conv1d_kernel_slice,
    permute_kio_to_oik, transpose_2d_slice,
};
pub use registry::{ModelPresetConfig, ProbePreset};
pub use safetensors::{SafetensorEntryHeader, SafetensorsMap};

use anyhow::{Result, anyhow, bail};
use std::path::PathBuf;

/// Verification report for a single model in the Hub.
#[derive(Debug, Clone)]
pub struct ModelVerifyReport {
    pub id: String,
    pub status: ModelStatus,
    pub local_weights_path: PathBuf,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub remote_check: Option<RemoteLinkCheck>,
}

/// High-level Model Hub client combining the declarative [`ModelCatalog`] and local [`HubCache`].
#[derive(Debug, Clone)]
pub struct ModelHub {
    catalog: ModelCatalog,
    cache: HubCache,
}

impl ModelHub {
    /// Initializes the [`ModelHub`] using the embedded catalog (plus optional `DSP_KITCHEN_HUB_CATALOG`)
    /// and the default cache directory (`DSP_KITCHEN_HUB_DIR` or `~/.cache/dsp-kitchen/hub`).
    pub fn new() -> Result<Self> {
        Ok(Self {
            catalog: ModelCatalog::load_default()?,
            cache: HubCache::from_env_or_default(),
        })
    }

    /// Initializes a [`ModelHub`] with explicit catalog and cache instances (useful for testing).
    pub fn with_catalog_and_cache(catalog: ModelCatalog, cache: HubCache) -> Self {
        Self { catalog, cache }
    }

    /// Reference to the underlying [`ModelCatalog`].
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    /// Reference to the underlying [`HubCache`].
    pub fn cache(&self) -> &HubCache {
        &self.cache
    }

    /// Lists all models in the catalog along with their resolved download URLs and local cache statuses.
    pub fn list(&self) -> Vec<ModelHubEntry> {
        self.catalog
            .models()
            .iter()
            .map(|m| self.entry_for(m))
            .collect()
    }

    /// Returns detailed Hub entry information for a specific `model_id`.
    pub fn info(&self, model_id: &str) -> Result<ModelHubEntry> {
        let manifest = self
            .catalog
            .get(model_id)
            .ok_or_else(|| anyhow!("unknown model id '{model_id}' in dsp-synapse-ml catalog"))?;
        Ok(self.entry_for(manifest))
    }

    /// Pulls (downloads and SHA-256 verifies) a model's weights into the local cache.
    /// If already [`ModelStatus::Installed`] and `force` is `false`, returns the cached entry without re-downloading.
    pub fn pull(&self, model_id: &str, force: bool) -> Result<ModelHubEntry> {
        let manifest = self
            .catalog
            .get(model_id)
            .ok_or_else(|| anyhow!("unknown model id '{model_id}' in dsp-synapse-ml catalog"))?
            .clone();

        let dest = self.cache.weights_path(&manifest);
        if !force && self.cache.status(&manifest) == ModelStatus::Installed {
            return Ok(self.entry_for(&manifest));
        }

        let outcome = download_and_verify(&manifest.weights_url, &dest, &manifest.sha256)?;
        self.cache
            .record_installed(&manifest, &outcome.sha256, outcome.bytes_written)?;

        Ok(self.entry_for(&manifest))
    }

    /// Verifies the local SHA-256 integrity of a specific model (and optionally checks remote URL reachability).
    pub fn verify(&self, model_id: &str) -> Result<ModelVerifyReport> {
        self.verify_with_options(model_id, false)
    }

    /// Verifies a specific model with optional remote `HEAD` link inspection.
    pub fn verify_with_options(
        &self,
        model_id: &str,
        check_links: bool,
    ) -> Result<ModelVerifyReport> {
        let manifest = self
            .catalog
            .get(model_id)
            .ok_or_else(|| anyhow!("unknown model id '{model_id}' in dsp-synapse-ml catalog"))?;
        self.verify_manifest(manifest, check_links)
    }

    /// Verifies all models in the catalog (or all locally cached models).
    pub fn verify_all(&self, check_links: bool) -> Result<Vec<ModelVerifyReport>> {
        let mut reports = Vec::with_capacity(self.catalog.models().len());
        for manifest in self.catalog.models() {
            reports.push(self.verify_manifest(manifest, check_links)?);
        }
        Ok(reports)
    }

    /// Removes cached weights for `model_id`.
    pub fn remove(&self, model_id: &str) -> Result<bool> {
        let manifest = self
            .catalog
            .get(model_id)
            .ok_or_else(|| anyhow!("unknown model id '{model_id}' in dsp-synapse-ml catalog"))?;
        self.cache.remove_model(manifest)
    }

    /// Removes all cached model weights from the local Hub cache directory.
    pub fn clean(&self) -> Result<u64> {
        self.cache.clean_all()
    }

    fn entry_for(&self, manifest: &ModelManifest) -> ModelHubEntry {
        let resolved_download_url = resolve_weights_uri(&manifest.weights_url)
            .map(|(_, url)| url)
            .unwrap_or_else(|_| manifest.weights_url.clone());
        let local_weights_path = self.cache.weights_path(manifest);
        let status = self.cache.status(manifest);
        ModelHubEntry {
            manifest: manifest.clone(),
            status,
            resolved_download_url,
            local_weights_path,
        }
    }

    fn verify_manifest(
        &self,
        manifest: &ModelManifest,
        check_links: bool,
    ) -> Result<ModelVerifyReport> {
        let local_weights_path = self.cache.weights_path(manifest);
        let (status, actual_sha256) = if local_weights_path.exists() {
            match compute_file_sha256(&local_weights_path) {
                Ok((digest, _)) => {
                    if digest.eq_ignore_ascii_case(&manifest.sha256) {
                        (ModelStatus::Installed, Some(digest))
                    } else {
                        (ModelStatus::Corrupted, Some(digest))
                    }
                }
                Err(_) => (ModelStatus::Corrupted, None),
            }
        } else {
            (ModelStatus::Available, None)
        };

        let remote_check = if check_links {
            Some(check_remote_link(&manifest.weights_url)?)
        } else {
            None
        };

        if status == ModelStatus::Corrupted {
            bail!(
                "corrupted cached weights for '{}': expected sha256 {}, got {}",
                manifest.id,
                manifest.sha256,
                actual_sha256.as_deref().unwrap_or("<unreadable>")
            );
        }

        Ok(ModelVerifyReport {
            id: manifest.id.clone(),
            status,
            local_weights_path,
            expected_sha256: manifest.sha256.clone(),
            actual_sha256,
            remote_check,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_embedded_catalog_parses_all_target_families() {
        let catalog = ModelCatalog::load_default().unwrap();
        assert!(!catalog.models().is_empty());

        let required_families = [
            "spikenet2",
            "unitrefine",
            "dartsort",
            "kilosort4",
            "cebra",
            "cascade",
            "deepinterpolation",
        ];
        for family in required_families {
            assert!(
                !catalog.by_family(family).is_empty(),
                "missing family '{family}' in embedded catalog"
            );
        }

        // Verify specific expected IDs and URI resolution
        let ur = catalog.get("unitrefine/curation-v1").unwrap();
        assert_eq!(ur.format, ModelFormat::Onnx);
        let (_, resolved) = resolve_weights_uri(&ur.weights_url).unwrap();
        assert!(resolved.starts_with("https://huggingface.co/"));

        let ds = catalog.get("dartsort/singlechan-denoiser-v1").unwrap();
        assert_eq!(ds.format, ModelFormat::Safetensors);
        assert_eq!(ds.io_spec.inputs[0].shape, vec![-1, 1, 121]);
    }

    #[test]
    fn test_hub_pull_verify_and_remove_local_artifact() {
        let tmp_root = std::env::temp_dir().join(format!(
            "dsp_kitchen_hub_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_root);
        std::fs::create_dir_all(&tmp_root).unwrap();

        // Create a mock weight artifact on disk
        let dummy_weights = b"MOCK_ONNX_WEIGHT_BYTES_FOR_UNITREFINE";
        let expected_sha = hex::encode(Sha256::digest(dummy_weights));
        let src_file = tmp_root.join("source_weights.onnx");
        std::fs::write(&src_file, dummy_weights).unwrap();

        let custom_json = serde_json::json!({
            "schema_version": 1,
            "models": [{
                "id": "unitrefine/test-local-v1",
                "name": "Test Local UnitRefine",
                "family": "unitrefine",
                "task": "curation",
                "version": "1.0.0",
                "format": "onnx",
                "description": "Local test artifact",
                "upstream_repo": "https://github.com/SpikeInterface/UnitRefine",
                "paper_url": "https://doi.org/10.1101/2025.03.30.645770",
                "license": "MIT",
                "weights_url": format!("file://{}", src_file.display()),
                "sha256": expected_sha,
                "size_bytes": dummy_weights.len(),
                "io_spec": {
                    "inputs": [{ "name": "x", "dtype": "float32", "shape": [-1, 8] }],
                    "outputs": [{ "name": "y", "dtype": "float32", "shape": [-1, 3] }],
                    "sample_rate_hz": 30000.0,
                    "normalization": "unit_quality_standard"
                }
            }]
        });

        let catalog = ModelCatalog::from_json_str(&custom_json.to_string()).unwrap();
        let cache = HubCache::new(tmp_root.join("cache"));
        let hub = ModelHub::with_catalog_and_cache(catalog, cache);

        // Initially Available
        let info_before = hub.info("unitrefine/test-local-v1").unwrap();
        assert_eq!(info_before.status, ModelStatus::Available);

        // Pull into cache
        let pulled = hub.pull("unitrefine/test-local-v1", false).unwrap();
        assert_eq!(pulled.status, ModelStatus::Installed);
        assert!(pulled.local_weights_path.exists());

        // Verify local hash + link check
        let report = hub
            .verify_with_options("unitrefine/test-local-v1", true)
            .unwrap();
        assert_eq!(report.status, ModelStatus::Installed);
        assert!(report.remote_check.unwrap().reachable);

        // Remove from cache
        assert!(hub.remove("unitrefine/test-local-v1").unwrap());
        let info_after = hub.info("unitrefine/test-local-v1").unwrap();
        assert_eq!(info_after.status, ModelStatus::Available);

        let _ = std::fs::remove_dir_all(&tmp_root);
    }
}
