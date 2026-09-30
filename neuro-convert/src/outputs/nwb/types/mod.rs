//! One writer per NWB type. Each writes through the [`Backend`](super::backend::Backend) trait.

pub mod devices;
pub mod electrodes;
pub mod intervals;
pub mod nwbfile;
pub mod series;
pub mod subject;
pub mod tables;

use serde_json::{json, Value};

use super::backend::Attrs;

/// Attributes every typed object carries: namespace, type and a fresh object id.
pub fn typed(namespace: &str, neurodata_type: &str) -> Attrs {
    let mut a = Attrs::new();
    a.insert("namespace".into(), json!(namespace));
    a.insert("neurodata_type".into(), json!(neurodata_type));
    a.insert("object_id".into(), json!(uuid::Uuid::new_v4().to_string()));
    a
}

/// `typed` plus extra attributes.
pub fn typed_with(namespace: &str, neurodata_type: &str, extra: &[(&str, Value)]) -> Attrs {
    let mut a = typed(namespace, neurodata_type);
    for (k, v) in extra {
        a.insert((*k).into(), v.clone());
    }
    a
}

pub fn attrs(pairs: &[(&str, Value)]) -> Attrs {
    pairs.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect()
}

/// A `VectorData` column's attributes.
pub fn column(description: &str) -> Attrs {
    typed_with("hdmf-common", "VectorData", &[("description", json!(description))])
}
