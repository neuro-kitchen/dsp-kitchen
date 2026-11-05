//! Catalogs of published artifacts, as data: what each file is, where it is downloaded from, how it
//! is verified (SHA-256, size) and whom to credit.
//!
//! One JSON document per domain, compiled in by its feature (`synapse`: [`synapse`]).
//! [`Catalog::load_default`] merges the compiled-in documents with an optional user file named by
//! [`CATALOG_ENV`] (entries with the same id replace the built-in ones, others are added).

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::Artifact;

#[cfg(feature = "synapse")]
pub mod synapse;

/// Environment variable naming a JSON catalog merged over the built-in ones.
pub const CATALOG_ENV: &str = "DSPK_HUB_CATALOG";

/// Version of the catalog documents this crate reads.
pub const CATALOG_SCHEMA_VERSION: u32 = 2;

/// A paper to cite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditPaper {
    pub title: String,
    pub authors: Vec<String>,
    pub venue: String,
    pub year: u32,
    pub doi: String,
}

/// The code the artifact comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditCode {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default)]
    pub version: String,
}

/// Whom to credit for an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Credit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paper: Option<CreditPaper>,
    pub code: CreditCode,
    /// Licence of the file itself, when its publisher states one (else `None`: never re-hosted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_license: Option<String>,
    #[serde(default)]
    pub notes: String,
}

impl Credit {
    /// One-line citation: authors (year), title, venue, DOI, then the code.
    pub fn citation(&self) -> String {
        let code = match &self.code.license {
            Some(l) => format!("Code: {} ({}, {l})", self.code.url, self.code.version),
            None => format!("Code: {} ({})", self.code.url, self.code.version),
        };
        match &self.paper {
            Some(p) => {
                let authors = match p.authors.as_slice() {
                    [] => String::new(),
                    [one] => one.clone(),
                    [first, second] => format!("{first} & {second}"),
                    [first, .., last] => format!("{first} … {last}"),
                };
                format!("{authors} ({}). {}. {}. https://doi.org/{}. {code}", p.year, p.title, p.venue, p.doi)
            }
            None => code,
        }
    }
}

/// One published artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Unique id `"<family>/<slug>"` (e.g. `"kilosort4/wtemp-v1"`).
    pub id: String,
    /// Human-readable title.
    pub name: String,
    /// Family (sorter or model) the artifact belongs to.
    pub family: String,
    /// What the file is (`"templates"`, `"weights"`, …), shown in listings.
    pub kind: String,
    #[serde(default)]
    pub description: String,
    /// File name in the cache (the publisher's name, e.g. `wTEMP.npz`).
    pub file_name: String,
    /// Download URI (`hf://`, `gh://`, `zenodo://`, `https://`, `file://`).
    pub url: String,
    /// Lowercase hex SHA-256 (required: an entry without one cannot be pulled).
    pub sha256: String,
    pub size_bytes: u64,
    pub credit: Credit,
}

impl CatalogEntry {
    /// What [`crate::Hub`] downloads and verifies for this entry.
    pub fn artifact(&self) -> Artifact {
        Artifact {
            id: self.id.clone(),
            family: self.family.clone(),
            file_name: self.file_name.clone(),
            url: self.url.clone(),
            sha256: self.sha256.clone(),
            size_bytes: self.size_bytes,
        }
    }
}

/// A catalog document: `{"schema_version": 2, "artifacts": [...]}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogDocument {
    schema_version: u32,
    artifacts: Vec<CatalogEntry>,
}

/// Entries of one or more catalog documents.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
}

impl Catalog {
    /// The compiled-in catalogs (one per enabled feature), plus the file named by [`CATALOG_ENV`].
    pub fn load_default() -> Result<Self> {
        let mut catalog = Self::default();
        #[cfg(feature = "synapse")]
        catalog.merge(Self::from_json_str(synapse::SYNAPSE_CATALOG_JSON).context("built-in synapse catalog")?);
        if let Some(path) = std::env::var_os(CATALOG_ENV).filter(|p| !p.is_empty()) {
            catalog.merge_file(Path::new(&path))?;
        }
        Ok(catalog)
    }

    /// Parses one catalog document.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let doc: CatalogDocument = serde_json::from_str(json).context("invalid catalog JSON")?;
        anyhow::ensure!(
            doc.schema_version == CATALOG_SCHEMA_VERSION,
            "catalog schema_version {} is not supported (this dspk-hub reads {CATALOG_SCHEMA_VERSION})",
            doc.schema_version
        );
        Ok(Self { entries: doc.artifacts })
    }

    /// Merges the catalog file at `path` (its entries replace those with the same id).
    pub fn merge_file(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path).with_context(|| format!("catalog {}", path.display()))?;
        self.merge(Self::from_json_str(&text).with_context(|| format!("catalog {}", path.display()))?);
        Ok(())
    }

    /// Merges `other`: its entries replace those with the same id, the others are appended.
    pub fn merge(&mut self, other: Catalog) {
        for incoming in other.entries {
            match self.entries.iter_mut().find(|e| e.id == incoming.id) {
                Some(existing) => *existing = incoming,
                None => self.entries.push(incoming),
            }
        }
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_replaces_by_id_and_appends_new_ids() {
        let doc = |id: &str, url: &str| {
            format!(
                r#"{{"schema_version": 2, "artifacts": [{{"id": "{id}", "name": "n", "family": "f", "kind": "k", "file_name": "a.bin",
                "url": "{url}", "sha256": "{}", "size_bytes": 1, "credit": {{"code": {{"url": "https://example.org"}}}}}}]}}"#,
                "0".repeat(64)
            )
        };
        let mut catalog = Catalog::from_json_str(&doc("f/a", "https://one")).unwrap();
        catalog.merge(Catalog::from_json_str(&doc("f/a", "https://two")).unwrap());
        catalog.merge(Catalog::from_json_str(&doc("f/b", "https://three")).unwrap());
        assert_eq!(catalog.entries().len(), 2);
        assert_eq!(catalog.get("f/a").unwrap().url, "https://two");
        assert!(Catalog::from_json_str(r#"{"schema_version": 1, "artifacts": []}"#).is_err(), "other schema versions are refused");
    }
}
