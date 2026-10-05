//! Local cache of downloaded artifacts (`~/.cache/dsp-kitchen/hub/` or `DSP_KITCHEN_HUB_DIR`):
//! `models/<family>/<slug>/<file>` plus an `index.json` recording each installed file's hash and
//! size, so status checks never re-read the files ([`HubCache::verify`] does).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Artifact;

/// Environment variable overriding the cache root.
pub const HUB_DIR_ENV: &str = "DSP_KITCHEN_HUB_DIR";

/// Read buffer of the hashing loop.
const HASH_CHUNK_BYTES: usize = 64 * 1024;

/// Local state of an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactStatus {
    /// Downloaded and recorded with the expected hash and size.
    Installed,
    /// Not downloaded.
    Available,
    /// A file is present but its record or size does not match the expected artifact.
    Corrupted,
}

impl ArtifactStatus {
    /// Badge for CLI tables.
    pub fn badge(self) -> &'static str {
        match self {
            Self::Installed => "[INSTALLED]",
            Self::Available => "[AVAILABLE]",
            Self::Corrupted => "[CORRUPTED]",
        }
    }
}

impl std::fmt::Display for ArtifactStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.badge())
    }
}

/// `index.json` entry of an installed artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedArtifactRecord {
    pub id: String,
    pub family: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub relative_path: String,
    pub source_url: String,
}

/// `index.json` at the cache root.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HubCacheIndex {
    pub entries: BTreeMap<String, CachedArtifactRecord>,
}

/// The local cache directory. See the module docs.
#[derive(Debug, Clone)]
pub struct HubCache {
    root_dir: PathBuf,
}

impl HubCache {
    /// `$DSP_KITCHEN_HUB_DIR`, else `<user cache>/dsp-kitchen/hub`.
    pub fn from_env_or_default() -> Self {
        if let Ok(dir) = std::env::var(HUB_DIR_ENV)
            && !dir.trim().is_empty()
        {
            return Self::new(PathBuf::from(dir.trim()));
        }
        let base = dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".cache")))
            .unwrap_or_else(|| PathBuf::from(".cache"));
        Self::new(base.join("dsp-kitchen").join("hub"))
    }

    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self { root_dir: root_dir.into() }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn index_path(&self) -> PathBuf {
        self.root_dir.join("index.json")
    }

    /// `<root>/models/<family>/<slug>/`.
    pub fn artifact_dir(&self, artifact: &Artifact) -> PathBuf {
        self.root_dir.join("models").join(&artifact.family).join(artifact.slug())
    }

    /// Local path of the artifact's file.
    pub fn path(&self, artifact: &Artifact) -> PathBuf {
        self.artifact_dir(artifact).join(&artifact.file_name)
    }

    /// `index.json`, or an empty index when there is none.
    pub fn load_index(&self) -> Result<HubCacheIndex> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(HubCacheIndex::default());
        }
        let content = std::fs::read_to_string(&path).with_context(|| format!("failed to read hub cache index {}", path.display()))?;
        serde_json::from_str(&content).with_context(|| format!("failed to parse hub cache index {}", path.display()))
    }

    pub fn save_index(&self, index: &HubCacheIndex) -> Result<()> {
        std::fs::create_dir_all(&self.root_dir).with_context(|| format!("failed to create hub cache directory {}", self.root_dir.display()))?;
        let path = self.index_path();
        std::fs::write(&path, serde_json::to_string_pretty(index)?).with_context(|| format!("failed to write hub cache index {}", path.display()))
    }

    /// Records a verified download in `index.json`.
    pub fn record_installed(&self, artifact: &Artifact, sha256: &str, size_bytes: u64) -> Result<()> {
        let path = self.path(artifact);
        let relative_path = path.strip_prefix(&self.root_dir).unwrap_or(&path).to_string_lossy().into_owned();
        let mut index = self.load_index().unwrap_or_default();
        index.entries.insert(
            artifact.id.clone(),
            CachedArtifactRecord {
                id: artifact.id.clone(),
                family: artifact.family.clone(),
                sha256: sha256.to_ascii_lowercase(),
                size_bytes,
                relative_path,
                source_url: artifact.url.clone(),
            },
        );
        self.save_index(&index)
    }

    /// Status from `index.json` and the file's size (no hashing; see [`Self::verify`]).
    pub fn status(&self, artifact: &Artifact) -> ArtifactStatus {
        let path = self.path(artifact);
        let Ok(meta) = path.metadata() else { return ArtifactStatus::Available };
        let recorded = self.load_index().ok().and_then(|i| i.entries.get(&artifact.id).cloned());
        match recorded {
            Some(r) if r.sha256.eq_ignore_ascii_case(&artifact.sha256) && r.size_bytes == artifact.size_bytes && meta.len() == artifact.size_bytes => {
                ArtifactStatus::Installed
            }
            _ => ArtifactStatus::Corrupted,
        }
    }

    /// Re-hashes the cached file: `Ok(status)` with `Installed` only when hash and size match.
    pub fn verify(&self, artifact: &Artifact) -> Result<(ArtifactStatus, Option<String>)> {
        let path = self.path(artifact);
        if !path.exists() {
            return Ok((ArtifactStatus::Available, None));
        }
        let (digest, size) = compute_file_sha256(&path)?;
        let ok = digest.eq_ignore_ascii_case(&artifact.sha256) && size == artifact.size_bytes;
        Ok((if ok { ArtifactStatus::Installed } else { ArtifactStatus::Corrupted }, Some(digest)))
    }

    /// Removes an artifact's directory and index entry; `true` when anything was removed.
    pub fn remove(&self, artifact: &Artifact) -> Result<bool> {
        let mut removed = false;
        let dir = self.artifact_dir(artifact);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).with_context(|| format!("failed to remove cached artifact directory {}", dir.display()))?;
            removed = true;
        }
        if let Ok(mut index) = self.load_index()
            && index.entries.remove(&artifact.id).is_some()
        {
            self.save_index(&index)?;
            removed = true;
        }
        Ok(removed)
    }

    /// Removes every cached artifact and the index; returns the bytes freed.
    pub fn clean_all(&self) -> Result<u64> {
        let models_dir = self.root_dir.join("models");
        let freed = dir_size_bytes(&models_dir);
        if models_dir.exists() {
            std::fs::remove_dir_all(&models_dir).with_context(|| format!("failed to clean hub models directory {}", models_dir.display()))?;
        }
        let _ = std::fs::remove_file(self.index_path());
        Ok(freed)
    }
}

/// Lowercase hex SHA-256 and byte length of a file.
pub fn compute_file_sha256(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("failed to open file for SHA-256 check: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK_BYTES];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex::encode(hasher.finalize()), total))
}

fn dir_size_bytes(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    entries
        .flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() { dir_size_bytes(&p) } else { e.metadata().map_or(0, |m| m.len()) }
        })
        .sum()
}
