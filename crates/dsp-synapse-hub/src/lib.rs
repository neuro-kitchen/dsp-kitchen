//! Verified download and local cache of published artifacts (model weights, sorter arrays).
//!
//! - [`Artifact`]: what to fetch — id, family, file name, URI (`hf://`, `gh://`, `zenodo://`,
//!   `https://`, `file://`, see [`providers`]), expected SHA-256 and size.
//! - [`Hub`]: [`Hub::pull`] downloads into the [`HubCache`] and verifies hash and size before the
//!   file is committed (an artifact without a hash is refused); [`Hub::status`] reads the cache
//!   index; [`Hub::verify`] re-hashes.
//!
//! Catalogs (which artifacts exist and whom to credit) belong to the crates that publish them,
//! e.g. `dsp-synapse-ml`. All network access of the workspace's model code lives here.

pub mod cache;
pub mod download;
pub mod providers;

use std::path::PathBuf;

use anyhow::{Result, bail};

pub use cache::{ArtifactStatus, CachedArtifactRecord, HubCache, HubCacheIndex, compute_file_sha256};
pub use download::{DownloadOutcome, RemoteLinkCheck, check_remote_link, download_and_verify};
pub use providers::{ProviderKind, resolve_gh_uri, resolve_hf_uri, resolve_weights_uri, resolve_zenodo_uri};

/// A published file to fetch and verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    /// Unique id `"<family>/<slug>"`.
    pub id: String,
    pub family: String,
    /// File name in the cache (e.g. `wTEMP.npz`).
    pub file_name: String,
    /// Download URI.
    pub url: String,
    /// Lowercase hex SHA-256.
    pub sha256: String,
    pub size_bytes: u64,
}

impl Artifact {
    /// Last segment of the id (the cache sub-directory).
    pub fn slug(&self) -> &str {
        self.id.rsplit('/').next().unwrap_or(&self.id)
    }
}

/// Local state and integrity of one artifact.
#[derive(Debug, Clone)]
pub struct ArtifactReport {
    pub id: String,
    pub status: ArtifactStatus,
    pub local_path: PathBuf,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub remote_check: Option<RemoteLinkCheck>,
}

/// Downloads into and reads from a [`HubCache`].
#[derive(Debug, Clone)]
pub struct Hub {
    cache: HubCache,
}

impl Hub {
    /// Cache at `$DSP_KITCHEN_HUB_DIR` or the user cache directory.
    pub fn from_env_or_default() -> Self {
        Self { cache: HubCache::from_env_or_default() }
    }

    pub fn with_cache(cache: HubCache) -> Self {
        Self { cache }
    }

    pub fn cache(&self) -> &HubCache {
        &self.cache
    }

    /// Local path of `artifact` (whether or not it is installed).
    pub fn path(&self, artifact: &Artifact) -> PathBuf {
        self.cache.path(artifact)
    }

    pub fn status(&self, artifact: &Artifact) -> ArtifactStatus {
        self.cache.status(artifact)
    }

    /// Downloads and verifies `artifact` unless it is already installed (or `force`), returning
    /// its local path.
    pub fn pull(&self, artifact: &Artifact, force: bool) -> Result<PathBuf> {
        let dest = self.cache.path(artifact);
        if !force && self.cache.status(artifact) == ArtifactStatus::Installed {
            return Ok(dest);
        }
        let outcome = download_and_verify(&artifact.url, &dest, &artifact.sha256, artifact.size_bytes)?;
        self.cache.record_installed(artifact, &outcome.sha256, outcome.bytes_written)?;
        Ok(dest)
    }

    /// Re-hashes the cached file (and, with `check_link`, probes the remote URL). A corrupted
    /// file is an error.
    pub fn verify(&self, artifact: &Artifact, check_link: bool) -> Result<ArtifactReport> {
        let (status, actual_sha256) = self.cache.verify(artifact)?;
        if status == ArtifactStatus::Corrupted {
            bail!(
                "corrupted cached file for '{}': expected sha256 {} and {} bytes, got {}",
                artifact.id,
                artifact.sha256,
                artifact.size_bytes,
                actual_sha256.as_deref().unwrap_or("<unreadable>")
            );
        }
        let remote_check = if check_link { Some(check_remote_link(&artifact.url)?) } else { None };
        Ok(ArtifactReport {
            id: artifact.id.clone(),
            status,
            local_path: self.cache.path(artifact),
            expected_sha256: artifact.sha256.clone(),
            actual_sha256,
            remote_check,
        })
    }

    pub fn remove(&self, artifact: &Artifact) -> Result<bool> {
        self.cache.remove(artifact)
    }

    pub fn clean(&self) -> Result<u64> {
        self.cache.clean_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn pull_verify_status_and_remove_a_local_artifact() {
        let root = std::env::temp_dir().join(format!("dsp_synapse_hub_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let bytes = b"LOCAL_TEST_ARTIFACT_BYTES";
        let src = root.join("source.bin");
        std::fs::write(&src, bytes).unwrap();
        let artifact = Artifact {
            id: "test/local-v1".into(),
            family: "test".into(),
            file_name: "local.bin".into(),
            url: format!("file://{}", src.display()),
            sha256: hex::encode(Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
        };
        let hub = Hub::with_cache(HubCache::new(root.join("cache")));
        assert_eq!(hub.status(&artifact), ArtifactStatus::Available);
        let path = hub.pull(&artifact, false).unwrap();
        assert!(path.exists());
        assert_eq!(hub.status(&artifact), ArtifactStatus::Installed);
        assert!(hub.verify(&artifact, true).unwrap().remote_check.unwrap().reachable);

        // A wrong size or an empty hash is refused
        let wrong_size = Artifact { size_bytes: 1, id: "test/wrong-size".into(), ..artifact.clone() };
        assert!(hub.pull(&wrong_size, false).is_err());
        let no_hash = Artifact { sha256: String::new(), id: "test/no-hash".into(), ..artifact.clone() };
        assert!(hub.pull(&no_hash, false).is_err());

        assert!(hub.remove(&artifact).unwrap());
        assert_eq!(hub.status(&artifact), ArtifactStatus::Available);
        let _ = std::fs::remove_dir_all(&root);
    }
}
