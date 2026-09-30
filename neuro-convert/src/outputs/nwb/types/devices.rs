//! `/general/devices/<name>` (`Device`) and `/general/devices/models/<model>` (`DeviceModel`).
//!
//! Since NWB 2.9 the manufacturer and model belong to a `DeviceModel` that the device links to
//! (`Device.manufacturer` is deprecated). A device whose model is unknown keeps its manufacturer in
//! its description.

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
    let mut models = Vec::new();
    for d in devices {
        match (&d.model, &d.manufacturer) {
            (Some(model), Some(manufacturer)) => {
                if models.is_empty() {
                    b.group("/general/devices/models", Attrs::new())?;
                }
                if !models.contains(model) {
                    let a = typed_with("core", "DeviceModel", &[("manufacturer", json!(manufacturer))]);
                    b.group(&format!("/general/devices/models/{model}"), a)?;
                    models.push(model.clone());
                }
                let links = json!([{ "source": ".", "path": format!("/general/devices/models/{model}"), "name": "model" }]);
                let a = typed_with("core", "Device", &[("description", json!(d.description)), ("_LINKS", links)]);
                b.group(&format!("/general/devices/{}", d.name), a)?;
            }
            (_, manufacturer) => {
                let description = match manufacturer {
                    Some(m) => format!("{} (manufacturer: {m})", d.description),
                    None => d.description.clone(),
                };
                b.group(&format!("/general/devices/{}", d.name), typed_with("core", "Device", &[("description", json!(description))]))?;
            }
        }
    }
    Ok(())
}
