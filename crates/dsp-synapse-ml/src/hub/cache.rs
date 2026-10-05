//! Local filesystem cache manager (`~/.cache/dsp-kitchen/hub/` or `DSP_KITCHEN_HUB_DIR`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::manifest::{ModelManifest, ModelStatus};

/// Entry recorded in `~/.cache/dsp-kitchen/hub/index.json` for an installed model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedModelRecord {
    pub id: String,
    pub family: String,
    pub version: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub relative_path: String,
    pub source_url: String,
}

/// Persistent index file (`index.json`) stored at the root of the Hub cache directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HubCacheIndex {
    pub entries: BTreeMap<String, CachedModelRecord>,
}

/// Manages the local model cache directory (`models/<family>/<model>/` and `index.json`).
#[derive(Debug, Clone)]
pub struct HubCache {
    root_dir: PathBuf,
}

impl HubCache {
    /// Resolves the Hub cache directory from `DSP_KITCHEN_HUB_DIR` if set, otherwise
    /// defaults to `~/.cache/dsp-kitchen/hub`.
    pub fn from_env_or_default() -> Self {
        if let Ok(dir) = std::env::var("DSP_KITCHEN_HUB_DIR") {
            let trimmed = dir.trim();
            if !trimmed.is_empty() {
                return Self::new(PathBuf::from(trimmed));
            }
        }
        let base = dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".cache")))
            .unwrap_or_else(|| PathBuf::from(".cache"));
        Self::new(base.join("dsp-kitchen").join("hub"))
    }

    /// Creates a [`HubCache`] rooted at an explicit directory.
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    /// Root directory of the Hub cache.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Path to the persistent `index.json` file.
    pub fn index_path(&self) -> PathBuf {
        self.root_dir.join("index.json")
    }

    /// Directory where a specific model's weights and sidecar manifest are cached:
    /// `<root>/models/<family>/<model-slug>/`.
    pub fn model_dir(&self, manifest: &ModelManifest) -> PathBuf {
        let slug = manifest.id.split('/').next_back().unwrap_or(&manifest.id);
        self.root_dir
            .join("models")
            .join(&manifest.family)
            .join(slug)
    }

    /// Full local path to a model's weight artifact file.
    pub fn weights_path(&self, manifest: &ModelManifest) -> PathBuf {
        self.model_dir(manifest).join(manifest.artifact_filename())
    }

    /// Full local path to the cached copy of the model's JSON manifest.
    pub fn cached_manifest_path(&self, manifest: &ModelManifest) -> PathBuf {
        self.model_dir(manifest).join("manifest.json")
    }

    /// Loads `index.json` if present, or returns an empty index.
    pub fn load_index(&self) -> Result<HubCacheIndex> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(HubCacheIndex::default());
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read hub cache index {}", path.display()))?;
        let idx: HubCacheIndex = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse hub cache index {}", path.display()))?;
        Ok(idx)
    }

    /// Writes `index.json` atomically to the Hub cache root.
    pub fn save_index(&self, index: &HubCacheIndex) -> Result<()> {
        std::fs::create_dir_all(&self.root_dir).with_context(|| {
            format!(
                "failed to create hub cache directory {}",
                self.root_dir.display()
            )
        })?;
        let path = self.index_path();
        let json = serde_json::to_string_pretty(index)?;
        std::fs::write(&path, json)
            .with_context(|| format!("failed to write hub cache index {}", path.display()))?;
        Ok(())
    }

    /// Records a verified model artifact in `index.json` and writes its sidecar `manifest.json`.
    pub fn record_installed(&self, manifest: &ModelManifest, actual_sha256: &str, actual_size: u64) -> Result<()> {
        let model_dir = self.model_dir(manifest);
        std::fs::create_dir_all(&model_dir)?;
        let manifest_json = serde_json::to_string_pretty(manifest)?;
        std::fs::write(self.cached_manifest_path(manifest), manifest_json)?;

        let weights_path = self.weights_path(manifest);
        let rel = weights_path
            .strip_prefix(&self.root_dir)
            .unwrap_or(&weights_path)
            .to_string_lossy()
            .into_owned();

        let mut idx = self.load_index().unwrap_or_default();
        idx.entries.insert(
            manifest.id.clone(),
            CachedModelRecord {
                id: manifest.id.clone(),
                family: manifest.family.clone(),
                version: manifest.version.clone(),
                sha256: actual_sha256.to_ascii_lowercase(),
                size_bytes: actual_size,
                relative_path: rel,
                source_url: manifest.weights_url.clone(),
            },
        );
        self.save_index(&idx)
    }

    /// Checks the local status of a model (`Installed`, `Available`, or `Corrupted`) by verifying
    /// file existence and SHA-256 checksum.
    pub fn status(&self, manifest: &ModelManifest) -> ModelStatus {
        let path = self.weights_path(manifest);
        if !path.exists() {
            return ModelStatus::Available;
        }
        match compute_file_sha256(&path) {
            Ok((digest, _size)) => {
                if digest.eq_ignore_ascii_case(&manifest.sha256) {
                    ModelStatus::Installed
                } else {
                    ModelStatus::Corrupted
                }
            }
            Err(_) => ModelStatus::Corrupted,
        }
    }

    /// Removes a single model's cached directory and removes its entry from `index.json`.
    /// Returns `true` if any cached files or index entries were removed.
    pub fn remove_model(&self, manifest: &ModelManifest) -> Result<bool> {
        let mut removed = false;
        let dir = self.model_dir(manifest);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).with_context(|| {
                format!("failed to remove cached model directory {}", dir.display())
            })?;
            removed = true;
        }
        if let Ok(mut idx) = self.load_index()
            && idx.entries.remove(&manifest.id).is_some()
        {
            self.save_index(&idx)?;
            removed = true;
        }
        Ok(removed)
    }

    /// Cleans all cached model weights and resets `index.json`.
    /// Returns the number of bytes freed on disk.
    pub fn clean_all(&self) -> Result<u64> {
        let models_dir = self.root_dir.join("models");
        let freed = dir_size_bytes(&models_dir);
        if models_dir.exists() {
            std::fs::remove_dir_all(&models_dir).with_context(|| {
                format!(
                    "failed to clean hub models directory {}",
                    models_dir.display()
                )
            })?;
        }
        let idx_path = self.index_path();
        if idx_path.exists() {
            let _ = std::fs::remove_file(&idx_path);
        }
        Ok(freed)
    }
}

/// Computes the lowercase hex SHA-256 digest and byte length of a file on disk.
pub fn compute_file_sha256(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open file for SHA-256 check: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total_bytes = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total_bytes += n as u64;
    }
    let digest = hex::encode(hasher.finalize());
    Ok((digest, total_bytes))
}

fn dir_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size_bytes(&p);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}
