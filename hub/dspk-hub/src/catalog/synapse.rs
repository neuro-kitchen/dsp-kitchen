//! Electrophysiology artifacts (feature `synapse`): files that dspk's sorters and models read but
//! do not ship. `synapse.json` next to this file is the catalog.

/// The catalog document, compiled in.
pub const SYNAPSE_CATALOG_JSON: &str = include_str!("synapse.json");

/// Kilosort4's predefined temporal basis and universal templates (`wTEMP.npz`: `wPCA`, `wTEMP`,
/// 6 × 61 float32 each), used when templates are not learned from the recording.
pub const KILOSORT4_WTEMP: &str = "kilosort4/wtemp-v1";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;

    #[test]
    fn every_entry_is_verifiable_and_credited() {
        let catalog = Catalog::from_json_str(SYNAPSE_CATALOG_JSON).unwrap();
        assert!(!catalog.entries().is_empty());
        for e in catalog.entries() {
            assert_eq!(e.sha256.len(), 64, "{}: SHA-256 required", e.id);
            assert!(e.size_bytes > 0, "{}: size required", e.id);
            assert!(e.credit.paper.as_ref().is_some_and(|p| !p.doi.is_empty()), "{}: paper DOI required", e.id);
        }
        let ks = catalog.get(KILOSORT4_WTEMP).unwrap();
        assert_eq!(ks.file_name, "wTEMP.npz");
        assert!(ks.credit.citation().starts_with("Pachitariu … Stringer (2024). Spike sorting with Kilosort4."), "{}", ks.credit.citation());
    }
}
