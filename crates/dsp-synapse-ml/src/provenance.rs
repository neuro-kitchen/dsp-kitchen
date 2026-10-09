//! Where every sorter and model comes from: the paper to cite, the upstream code and its license,
//! and the files downloaded to run it. Exposed by every sorter and model through [`Attributed`],
//! and stored in the catalog, so a result can always be traced to its authors.

use serde::{Deserialize, Serialize};

/// How a sorter or model relates to its upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    /// Algorithm written here from the paper (upstream code not ported).
    ReimplementedFromPaper,
    /// Published weights / arrays run as released.
    UpstreamWeights,
    /// Algorithm ported from the upstream source (permissive license), stage by stage.
    PortedFromCode,
}

/// The publication to cite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paper {
    pub title: String,
    /// Family names in publication order (first author first).
    pub authors: Vec<String>,
    /// Journal or preprint server.
    pub venue: String,
    pub year: u16,
    /// DOI without the `https://doi.org/` prefix (e.g. `10.1038/s41592-024-02232-7`).
    pub doi: String,
    /// License of the paper itself, when stated (e.g. `CC-BY-4.0`).
    #[serde(default)]
    pub license: Option<String>,
}

impl Paper {
    /// `https://doi.org/<doi>`.
    pub fn doi_url(&self) -> String {
        format!("https://doi.org/{}", self.doi)
    }
}

/// The upstream source code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamCode {
    pub url: String,
    /// SPDX identifier, or `None` when the repository states none.
    pub license: Option<String>,
    /// Release tag or commit the work follows.
    pub version: String,
}

/// A file downloaded to run the sorter or model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactSource {
    /// File name as published (e.g. `wTEMP.npz`).
    pub name: String,
    /// Where it was downloaded from.
    pub url: String,
    /// Lowercase hex SHA-256 of the file, as checked when it was added.
    pub sha256: String,
    pub size_bytes: u64,
    /// License of the file, when stated separately from the code.
    #[serde(default)]
    pub license: Option<String>,
}

/// Provenance of one sorter or model. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Name as its authors publish it (e.g. `Kilosort4`, `EMUsort`).
    pub name: String,
    pub kind: ProvenanceKind,
    /// `None` when no paper could be verified.
    pub paper: Option<Paper>,
    pub code: UpstreamCode,
    /// Files it needs (empty when it learns everything from the recording).
    #[serde(default)]
    pub artifacts: Vec<ArtifactSource>,
    /// What differs from upstream, what is unverified.
    #[serde(default)]
    pub notes: String,
}

impl Provenance {
    /// One-line reference: `Authors (year). Title. Venue. https://doi.org/…`, then the code.
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
                    [first, .., last] if p.authors.len() > 2 => format!("{first} … {last}"),
                    [first, second] => format!("{first} & {second}"),
                    _ => unreachable!(),
                };
                format!("{authors} ({}). {}. {}. {}. {code}", p.year, p.title, p.venue, p.doi_url())
            }
            None => format!("{} (no verified paper). {code}", self.name),
        }
    }
}

/// Anything whose origin can be cited.
pub trait Attributed {
    fn provenance(&self) -> Provenance;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn citation_names_paper_doi_and_code() {
        let p = Provenance {
            name: "Kilosort4".into(),
            kind: ProvenanceKind::ReimplementedFromPaper,
            paper: Some(Paper {
                title: "Spike sorting with Kilosort4".into(),
                authors: vec!["Pachitariu".into(), "Sridhar".into(), "Pennington".into(), "Stringer".into()],
                venue: "Nature Methods".into(),
                year: 2024,
                doi: "10.1038/s41592-024-02232-7".into(),
                license: None,
            }),
            code: UpstreamCode { url: "https://github.com/MouseLand/Kilosort".into(), license: Some("GPL-3.0".into()), version: "v4.1.3".into() },
            artifacts: Vec::new(),
            notes: String::new(),
        };
        let c = p.citation();
        assert!(c.starts_with("Pachitariu … Stringer (2024). Spike sorting with Kilosort4. Nature Methods. https://doi.org/10.1038/s41592-024-02232-7."), "{c}");
        assert!(c.ends_with("Code: https://github.com/MouseLand/Kilosort (v4.1.3, GPL-3.0)"), "{c}");
    }
}
