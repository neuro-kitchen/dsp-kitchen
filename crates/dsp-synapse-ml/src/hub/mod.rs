//! The model catalog of `dsp-synapse-ml` (`catalog/models.json`: what each published artifact is,
//! how to verify it, whom to credit) and, with feature `hub`, [`ModelHub`]: catalog lookups backed
//! by `dsp-synapse-hub` downloads.

pub mod catalog;
pub mod manifest;
#[cfg(feature = "hub")]
mod model_hub;
pub mod safetensors;

pub use catalog::{CatalogDocument, EMBEDDED_MODELS_JSON, ModelCatalog};
pub use manifest::{ArraySpec, ModelFormat, ModelManifest, TensorIoSpec, TensorPortSpec};
#[cfg(feature = "hub")]
pub use model_hub::{ModelHub, ModelHubEntry};
pub use safetensors::{SafetensorEntryHeader, SafetensorsMap};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_catalog_entries_are_verifiable() {
        let catalog = ModelCatalog::load_default().unwrap();
        assert!(!catalog.models().is_empty());
        for m in catalog.models() {
            assert_eq!(m.sha256.len(), 64, "{}: SHA-256 required", m.id);
            assert!(m.size_bytes > 0, "{}: size required", m.id);
            assert!(m.provenance.paper.as_ref().is_some_and(|p| !p.doi.is_empty()), "{}: paper DOI required", m.id);
        }
        let ks = catalog.get("kilosort4/wtemp-v1").unwrap();
        assert_eq!(ks.format, ModelFormat::Npz);
        assert_eq!(ks.arrays.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["wPCA", "wTEMP"]);
        assert_eq!(ks.file_name(), "wTEMP.npz");
    }
}
