//! Which TDT software wrote a block.
//!
//! - **Synapse**: writes `Notes.txt`, `StoresListing.txt` and a zipped `.tin` whose
//!   `Summary.txt` carries the build number (`Versions.Synapse`).
//! - **OpenEx**: writes a `.tnt` note file and no Synapse sidecars.
//!
//! SEV header versions (0–4) are per file and handled with the SEV reader.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Software {
    Synapse { build: Option<String> },
    OpenEx,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtVersion {
    pub software: Software,
    /// `NOTEFILE_VERSION` from the `.tnt`, when present.
    pub notefile: Option<String>,
}

impl TdtVersion {
    pub fn detect(has_synapse_sidecars: bool, synapse_build: Option<String>, notefile: Option<String>) -> Self {
        let software = if has_synapse_sidecars || synapse_build.is_some() {
            Software::Synapse { build: synapse_build }
        } else if notefile.is_some() {
            Software::OpenEx
        } else {
            Software::Unknown
        };
        Self { software, notefile }
    }
}

impl fmt::Display for TdtVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.software {
            Software::Synapse { build: Some(b) } => write!(f, "Synapse {b}"),
            Software::Synapse { build: None } => write!(f, "Synapse"),
            Software::OpenEx => write!(f, "OpenEx"),
            Software::Unknown => write!(f, "unknown TDT software"),
        }
    }
}
