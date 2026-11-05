//! Model manifests of `dspk-synapse-ml`: the catalog (`catalog/models.json`: what each published
//! model is, its tensors, whom to credit), tensor specifications and the safetensors reader.
//! Downloading a model's file (with feature `hub`) goes through the `dspk-hub` catalog entry of
//! the same id ([`crate::runtime::pull_model`]).

pub mod catalog;
pub mod safetensors;
pub mod spec;

pub use catalog::{CatalogDocument, EMBEDDED_MODELS_JSON, ModelCatalog};
pub use safetensors::{SafetensorEntryHeader, SafetensorsMap};
pub use spec::{ArraySpec, ModelFormat, ModelManifest, TensorIoSpec, TensorPortSpec};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every manifest credits a paper; with `hub`, every one has a download in the dspk-hub
    /// catalog. (None yet: the models are parked until their artifacts are verified.)
    #[test]
    fn test_embedded_catalog_entries_are_credited() {
        let catalog = ModelCatalog::load_default().unwrap();
        for m in catalog.models() {
            assert!(m.provenance.paper.as_ref().is_some_and(|p| !p.doi.is_empty()), "{}: paper DOI required", m.id);
            #[cfg(feature = "hub")]
            assert!(dspk_hub::Catalog::load_default().unwrap().get(&m.id).is_some(), "{}: no dspk-hub catalog entry", m.id);
        }
    }
}
