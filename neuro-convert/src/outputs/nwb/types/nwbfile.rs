//! `NWBFile`: root attributes, required datasets, top-level groups, `/general` fields and the
//! cached specifications.

use serde_json::json;

use super::{attrs, typed_with};
use crate::common::time::format_iso;
use crate::error::Result;
use crate::outputs::nwb::backend::{Attrs, Backend};
use crate::outputs::nwb::mapping::NwbPlan;
use crate::outputs::nwb::schema::{NWB_VERSION, SPECS};

pub fn write_root(b: &dyn Backend, plan: &NwbPlan) -> Result<()> {
    let root = typed_with("core", "NWBFile", &[("nwb_version", json!(NWB_VERSION)), (".specloc", json!("specifications"))]);
    b.group("/", root)?;

    let f = &plan.file;
    b.string("/session_description", &f.description, str_attrs())?;
    b.string("/identifier", &f.identifier, str_attrs())?;
    b.string("/session_start_time", &f.start_time, str_attrs())?;
    b.string("/timestamps_reference_time", &f.start_time, str_attrs())?;
    b.strings("/file_create_date", &[now_utc()], "num_modifications", Attrs::new())?;

    for g in ["/acquisition", "/analysis", "/processing", "/stimulus", "/stimulus/presentation", "/stimulus/templates", "/general"] {
        b.group(g, Attrs::new())?;
    }

    // /general
    if let Some(v) = &f.experiment_description {
        b.string("/general/experiment_description", v, str_attrs())?;
    }
    if !f.experimenters.is_empty() {
        b.strings("/general/experimenter", &f.experimenters, "num_experimenters", Attrs::new())?;
    }
    if let Some(v) = &f.lab {
        b.string("/general/lab", v, str_attrs())?;
    }
    if let Some(v) = &f.institution {
        b.string("/general/institution", v, str_attrs())?;
    }
    if let Some(v) = &f.notes {
        b.string("/general/notes", v, str_attrs())?;
    }
    if !f.keywords.is_empty() {
        b.strings("/general/keywords", &f.keywords, "num_keywords", Attrs::new())?;
    }
    Ok(())
}

/// Caches the schema so readers need not have the same version installed.
pub fn write_specifications(b: &dyn Backend) -> Result<()> {
    b.group("/specifications", Attrs::new())?;
    for (ns, version, sources) in SPECS {
        b.group(&format!("/specifications/{ns}"), Attrs::new())?;
        b.group(&format!("/specifications/{ns}/{version}"), Attrs::new())?;
        for (name, text) in *sources {
            b.string(&format!("/specifications/{ns}/{version}/{name}"), text, Attrs::new())?;
        }
    }
    Ok(())
}

fn str_attrs() -> Attrs {
    attrs(&[])
}

fn now_utc() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
    format!("{}+00:00", format_iso(secs))
}
