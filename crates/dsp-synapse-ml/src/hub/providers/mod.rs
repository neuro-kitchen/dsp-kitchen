//! URI resolvers converting shorthand provider URIs (`hf://`, `gh://`, `zenodo://`, `https://`, `file://`)
//! into direct downloadable URLs.

use anyhow::{Result, bail};

/// Supported remote or local artifact hosting providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderKind {
    /// HuggingFace Hub (`hf://<org>/<repo>[@<rev>]/<path>`).
    HuggingFace,
    /// GitHub Raw / Release (`gh://<org>/<repo>[@<rev>]/<path>`).
    GitHub,
    /// Zenodo Record (`zenodo://<record_id>/<filename>`).
    Zenodo,
    /// Direct HTTPS / HTTP URL (`https://...` or `http://...`).
    Https,
    /// Local filesystem path (`file:///path/to/weights`).
    LocalFile,
}

/// Splits a `<org>/<repo>[@<rev>]/<path>` body into `(org_repo, revision, file_path)`.
fn parse_repo_rev_path<'a>(rest: &'a str, default_rev: &'a str) -> Result<(&'a str, &'a str, &'a str)> {
    let mut parts = rest.splitn(3, '/');
    let org = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing organization/owner in URI '{rest}'"))?;
    let repo_and_rev = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing repository name in URI '{rest}'"))?;
    let path = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing artifact file path in URI '{rest}'"))?;

    let (repo, rev) = match repo_and_rev.split_once('@') {
        Some((r, rev)) if !r.is_empty() && !rev.is_empty() => (r, rev),
        Some(_) => bail!("malformed '@revision' in URI '{rest}'"),
        None => (repo_and_rev, default_rev),
    };

    let org_repo_len = org.len() + 1 + repo.len();
    let org_repo = &rest[..org_repo_len];
    Ok((org_repo, rev, path))
}

/// Resolves a `hf://<org>/<repo>[@<revision>]/<path>` URI to a direct HuggingFace `resolve` HTTPS URL.
pub fn resolve_hf_uri(uri: &str) -> Result<String> {
    let rest = uri
        .strip_prefix("hf://")
        .ok_or_else(|| anyhow::anyhow!("URI '{uri}' does not start with 'hf://'"))?;
    let (org_repo, rev, path) = parse_repo_rev_path(rest, "main")?;
    Ok(format!(
        "https://huggingface.co/{org_repo}/resolve/{rev}/{path}"
    ))
}

/// Resolves a `gh://<org>/<repo>[@<revision>]/<path>` URI to a direct `raw.githubusercontent.com` HTTPS URL.
pub fn resolve_gh_uri(uri: &str) -> Result<String> {
    let rest = uri
        .strip_prefix("gh://")
        .ok_or_else(|| anyhow::anyhow!("URI '{uri}' does not start with 'gh://'"))?;
    let (org_repo, rev, path) = parse_repo_rev_path(rest, "main")?;
    Ok(format!(
        "https://raw.githubusercontent.com/{org_repo}/{rev}/{path}"
    ))
}

/// Resolves a `zenodo://<record_id>/<filename>` URI to a direct Zenodo record file download HTTPS URL.
pub fn resolve_zenodo_uri(uri: &str) -> Result<String> {
    let rest = uri
        .strip_prefix("zenodo://")
        .ok_or_else(|| anyhow::anyhow!("URI '{uri}' does not start with 'zenodo://'"))?;
    let (record_id, filename) = rest.split_once('/').ok_or_else(|| {
        anyhow::anyhow!("expected 'zenodo://<record_id>/<filename>', got '{uri}'")
    })?;
    if record_id.is_empty() || filename.is_empty() {
        bail!("record_id and filename must be non-empty in '{uri}'");
    }
    Ok(format!(
        "https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    ))
}

/// Classifies and resolves any supported `weights_url` (`hf://`, `gh://`, `zenodo://`, `https://`, `http://`, `file://`)
/// into a concrete URL string and its [`ProviderKind`].
pub fn resolve_weights_uri(uri: &str) -> Result<(ProviderKind, String)> {
    let trimmed = uri.trim();
    if trimmed.starts_with("hf://") {
        Ok((ProviderKind::HuggingFace, resolve_hf_uri(trimmed)?))
    } else if trimmed.starts_with("gh://") {
        Ok((ProviderKind::GitHub, resolve_gh_uri(trimmed)?))
    } else if trimmed.starts_with("zenodo://") {
        Ok((ProviderKind::Zenodo, resolve_zenodo_uri(trimmed)?))
    } else if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        Ok((ProviderKind::Https, trimmed.to_string()))
    } else if trimmed.starts_with("file://") {
        Ok((ProviderKind::LocalFile, trimmed.to_string()))
    } else {
        bail!(
            "unsupported weights_url scheme in '{trimmed}' (expected hf://, gh://, zenodo://, https://, or file://)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_uri_resolvers() {
        let (kind, hf) =
            resolve_weights_uri("hf://SpikeInterface/UnitRefine@main/models/unitrefine.onnx")
                .unwrap();
        assert_eq!(kind, ProviderKind::HuggingFace);
        assert_eq!(
            hf,
            "https://huggingface.co/SpikeInterface/UnitRefine/resolve/main/models/unitrefine.onnx"
        );

        let (_, gh) = resolve_weights_uri("gh://MouseLand/Kilosort@v4.0/kilosort/models/wPCA.npy")
            .unwrap();
        assert_eq!(
            gh,
            "https://raw.githubusercontent.com/MouseLand/Kilosort/v4.0/kilosort/models/wPCA.npy"
        );

        let (zkind, zenodo) =
            resolve_weights_uri("zenodo://5163414/unet_ephys.onnx").unwrap();
        assert_eq!(zkind, ProviderKind::Zenodo);
        assert_eq!(
            zenodo,
            "https://zenodo.org/records/5163414/files/unet_ephys.onnx?download=1"
        );
    }
}
