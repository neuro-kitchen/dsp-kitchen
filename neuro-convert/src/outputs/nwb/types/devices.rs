//! `/general/devices/<name>` (`Device`).

use serde_json::json;

use super::typed_with;
use crate::error::Result;
use crate::model::Device;
use crate::outputs::nwb::backend::{Attrs, Backend};

pub fn write(b: &dyn Backend, devices: &[Device]) -> Result<()> {
    if devices.is_empty() {
        return Ok(());
    }
    b.group("/general/devices", Attrs::new())?;
    for d in devices {
        let mut extra = vec![("description", json!(d.description))];
        if let Some(m) = &d.manufacturer {
            extra.push(("manufacturer", json!(m)));
        }
        b.group(&format!("/general/devices/{}", d.name), typed_with("core", "Device", &extra))?;
    }
    Ok(())
}
