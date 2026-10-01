//! Model weight downloader with streaming SHA-256 verification and remote link inspection.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use super::providers::{ProviderKind, resolve_weights_uri};

/// Summary returned after downloading and verifying a model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadOutcome {
    pub resolved_url: String,
    pub sha256: String,
    pub bytes_written: u64,
}

/// Summary returned when inspecting a remote or local weight URL via `HEAD` (or range probe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteLinkCheck {
    pub resolved_url: String,
    pub reachable: bool,
    pub status_code: Option<u16>,
    pub content_length: Option<u64>,
    pub detail: String,
}

/// Downloads a model artifact from `weights_url` into `dest_path`, computing its SHA-256 digest
/// on the fly and verifying it against `expected_sha256` before committing the file.
pub fn download_and_verify(
    weights_url: &str,
    dest_path: &Path,
    expected_sha256: &str,
) -> Result<DownloadOutcome> {
    let (provider, resolved_url) = resolve_weights_uri(weights_url)?;
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create parent directory for {}",
                dest_path.display()
            )
        })?;
    }

    let tmp_path = dest_path.with_extension("part");
    let mut out_file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create temporary file {}", tmp_path.display()))?;

    let mut hasher = Sha256::new();
    let mut bytes_written = 0u64;

    match provider {
        ProviderKind::LocalFile => {
            let local_src = resolved_url
                .strip_prefix("file://")
                .unwrap_or(&resolved_url);
            let mut src_file = std::fs::File::open(local_src)
                .with_context(|| format!("failed to open local weight source {local_src}"))?;
            copy_and_hash(&mut src_file, &mut out_file, &mut hasher, &mut bytes_written)?;
        }
        _ => {
            let client = build_http_client(Duration::from_secs(120))?;
            let mut resp = client
                .get(&resolved_url)
                .send()
                .with_context(|| format!("HTTP GET failed for {resolved_url}"))?
                .error_for_status()
                .with_context(|| format!("HTTP server returned error status for {resolved_url}"))?;
            copy_and_hash(&mut resp, &mut out_file, &mut hasher, &mut bytes_written)?;
        }
    }

    out_file.flush()?;
    drop(out_file);

    let actual_sha256 = hex::encode(hasher.finalize());
    if !expected_sha256.is_empty() && !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        let _ = std::fs::remove_file(&tmp_path);
        bail!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            resolved_url,
            expected_sha256.to_ascii_lowercase(),
            actual_sha256
        );
    }

    std::fs::rename(&tmp_path, dest_path).with_context(|| {
        format!(
            "failed to move verified download {} to {}",
            tmp_path.display(),
            dest_path.display()
        )
    })?;

    Ok(DownloadOutcome {
        resolved_url,
        sha256: actual_sha256,
        bytes_written,
    })
}

/// Checks whether a model's `weights_url` is reachable via HTTP `HEAD` (or local file stat).
pub fn check_remote_link(weights_url: &str) -> Result<RemoteLinkCheck> {
    let (provider, resolved_url) = resolve_weights_uri(weights_url)?;
    if provider == ProviderKind::LocalFile {
        let local_src = resolved_url
            .strip_prefix("file://")
            .unwrap_or(&resolved_url);
        let p = Path::new(local_src);
        return if let Ok(meta) = p.metadata() {
            Ok(RemoteLinkCheck {
                resolved_url,
                reachable: true,
                status_code: Some(200),
                content_length: Some(meta.len()),
                detail: "local file exists".to_string(),
            })
        } else {
            Ok(RemoteLinkCheck {
                resolved_url,
                reachable: false,
                status_code: Some(404),
                content_length: None,
                detail: "local file not found".to_string(),
            })
        };
    }

    let client = build_http_client(Duration::from_secs(10))?;
    match client.head(&resolved_url).send() {
        Ok(resp) => {
            let code = resp.status().as_u16();
            let reachable = resp.status().is_success() || resp.status().is_redirection();
            let content_length = resp
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            Ok(RemoteLinkCheck {
                resolved_url,
                reachable,
                status_code: Some(code),
                content_length,
                detail: format!("HTTP {code}"),
            })
        }
        Err(err) => Ok(RemoteLinkCheck {
            resolved_url,
            reachable: false,
            status_code: err.status().map(|s| s.as_u16()),
            content_length: None,
            detail: err.to_string(),
        }),
    }
}

fn build_http_client(timeout: Duration) -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("dsp-kitchen-hub/", env!("CARGO_PKG_VERSION")))
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")
}

fn copy_and_hash<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    hasher: &mut Sha256,
    bytes_written: &mut u64,
) -> Result<()> {
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        *bytes_written += n as u64;
    }
    Ok(())
}
