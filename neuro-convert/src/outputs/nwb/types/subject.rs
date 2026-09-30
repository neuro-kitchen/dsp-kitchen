//! `/general/subject` (`Subject`).

use serde_json::json;

use super::{attrs, typed};
use crate::error::Result;
use crate::outputs::nwb::backend::Backend;
use crate::outputs::nwb::mapping::SubjectFields;

pub fn write(b: &dyn Backend, s: &SubjectFields) -> Result<()> {
    b.group("/general/subject", typed("core", "Subject"))?;
    let fields = [("subject_id", &s.id), ("species", &s.species), ("sex", &s.sex), ("strain", &s.strain), ("description", &s.description)];
    for (name, value) in fields {
        if let Some(v) = value {
            b.string(&format!("/general/subject/{name}"), v, attrs(&[]))?;
        }
    }
    if let Some(age) = &s.age {
        b.string("/general/subject/age", age, attrs(&[("reference", json!("birth"))]))?;
    }
    Ok(())
}
