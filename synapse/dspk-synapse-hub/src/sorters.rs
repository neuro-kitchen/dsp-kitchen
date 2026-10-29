//! The `dspk-sorters` program: every spike sorter (Kilosort4 and EMUsort under GPL-3.0,
//! MountainSort 5, SpyKING CIRCUS 2, Tridesclous 2) in one executable, released as a file of the
//! GitHub release of each dspk version. dspk itself never links it: front ends run it as a separate
//! program, so their own licence stays MIT OR Apache-2.0.
//!
//! [`install`] downloads this platform's build into `<data dir>/sorters/<version>/` and verifies it
//! against the release's manifest (`dspk-sorters-manifest.json`: file, SHA-256 and size per
//! platform) before it is committed; [`locate`] finds an installed or bundled one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::download::{build_http_client, download_and_verify};

/// The program's name (plus [`std::env::consts::EXE_SUFFIX`]).
pub const SORTERS_PROGRAM: &str = "dspk-sorters";
/// Environment variable naming a `dspk-sorters` executable to use instead of looking for one.
pub const SORTERS_BIN_ENV: &str = "DSPK_SORTERS_BIN";
/// Environment variable replacing the user data folder dspk installs programs into.
pub const DATA_DIR_ENV: &str = "DSPK_DATA_DIR";
/// Where releases are downloaded from: `<base>/v<version>/<file>`.
pub const RELEASE_BASE_URL: &str = "https://github.com/neuro-kitchen/dsp-kitchen/releases/download";
/// The manifest file of a release.
pub const MANIFEST_FILE: &str = "dspk-sorters-manifest.json";

/// One platform's build in a release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortersAsset {
    /// File name in the release.
    pub file: String,
    pub sha256: String,
    pub size: u64,
}

/// `dspk-sorters-manifest.json`: the builds of one release, keyed by target triple.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SortersManifest {
    pub version: String,
    pub assets: BTreeMap<String, SortersAsset>,
}

/// The target triple of the builds this platform runs, if releases include one.
pub fn host_target() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        _ => None,
    }
}

/// The program's file name on this platform.
pub fn program_file_name() -> String {
    format!("{SORTERS_PROGRAM}{}", std::env::consts::EXE_SUFFIX)
}

/// `$DSPK_DATA_DIR`, else `<user data dir>/dspk` (`~/.local/share/dspk` on Linux,
/// `~/Library/Application Support/dspk` on macOS, `%LOCALAPPDATA%\dspk` on Windows).
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(DATA_DIR_ENV)
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir.trim());
    }
    let base = dirs::data_local_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from(".local/share"));
    base.join("dspk")
}

/// Where [`install`] puts `version`.
pub fn install_dir(version: &str) -> PathBuf {
    data_dir().join("sorters").join(version)
}

/// The installed program of `version`, if present.
pub fn installed(version: &str) -> Option<PathBuf> {
    let path = install_dir(version).join(program_file_name());
    path.is_file().then_some(path)
}

/// The `dspk-sorters` to run for `version`: `$DSPK_SORTERS_BIN`, else one next to `beside` (the
/// calling executable: a source build puts both in the same folder), else the installed one.
pub fn locate(version: &str, beside: Option<&Path>) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(SORTERS_BIN_ENV)
        && !path.trim().is_empty()
    {
        return Some(PathBuf::from(path.trim()));
    }
    if let Some(dir) = beside.and_then(Path::parent) {
        let path = dir.join(program_file_name());
        if path.is_file() {
            return Some(path);
        }
    }
    installed(version)
}

/// The release folder URL of `version`.
pub fn release_url(version: &str) -> String {
    format!("{RELEASE_BASE_URL}/v{version}")
}

/// Downloads and parses the manifest of `version`'s release.
pub fn fetch_manifest(version: &str) -> Result<SortersManifest> {
    let url = format!("{}/{MANIFEST_FILE}", release_url(version));
    let client = build_http_client(Duration::from_secs(30))?;
    let manifest: SortersManifest = client
        .get(&url)
        .send()
        .with_context(|| format!("HTTP GET failed for {url}"))?
        .error_for_status()
        .with_context(|| format!("no sorter manifest at {url} (is v{version} released with dspk-sorters?)"))?
        .json()
        .with_context(|| format!("{url} is not a sorter manifest"))?;
    if manifest.version != version {
        bail!("{url} describes version {}, not {version}", manifest.version);
    }
    Ok(manifest)
}

/// Installs `version` for this platform: fetches the release manifest, downloads this platform's
/// build, verifies its SHA-256 and size, and makes it executable. Returns the installed path.
pub fn install(version: &str) -> Result<PathBuf> {
    let Some(target) = host_target() else {
        bail!("no dspk-sorters build is released for {} {}; build it from source (`cargo build -p dspk-sorters`)", std::env::consts::OS, std::env::consts::ARCH);
    };
    let manifest = fetch_manifest(version)?;
    let Some(asset) = manifest.assets.get(target) else {
        bail!("release v{version} has no dspk-sorters build for {target}");
    };
    let dest = install_dir(version).join(program_file_name());
    let url = format!("{}/{}", release_url(version), asset.file);
    download_and_verify(&url, &dest, &asset.sha256, asset.size)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755)).with_context(|| format!("failed to make {} executable", dest.display()))?;
    }
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips() {
        let mut manifest = SortersManifest { version: "0.0.4".into(), ..Default::default() };
        manifest.assets.insert(
            "x86_64-unknown-linux-gnu".into(),
            SortersAsset { file: "dspk-sorters-x86_64-unknown-linux-gnu".into(), sha256: "ab".repeat(32), size: 42 },
        );
        let text = serde_json::to_string(&manifest).unwrap();
        assert_eq!(serde_json::from_str::<SortersManifest>(&text).unwrap(), manifest);
    }

    #[test]
    fn locate_prefers_the_program_beside_the_caller() {
        let dir = std::env::temp_dir().join(format!("dspk-sorters-locate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join(program_file_name());
        std::fs::write(&program, b"").unwrap();
        let caller = dir.join("dspk");
        // Only when the environment does not name one
        if std::env::var(SORTERS_BIN_ENV).is_err() {
            assert_eq!(locate("0.0.0-test", Some(&caller)), Some(program.clone()));
        }
        std::fs::remove_file(&program).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
