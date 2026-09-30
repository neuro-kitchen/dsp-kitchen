//! `/general/extracellular_ephys`: electrode groups and the electrodes table (one row per
//! channel of every electrical series, in series order, then snippet channels without a series).

use serde_json::json;

use super::{column, typed, typed_with};
use crate::error::Result;
use crate::model::Session;
use crate::outputs::nwb::backend::{Attrs, Backend};
use crate::outputs::nwb::mapping::NwbPlan;

pub const TABLE_PATH: &str = "/general/extracellular_ephys/electrodes";

pub fn write(b: &dyn Backend, plan: &NwbPlan, session: &Session) -> Result<()> {
    let electrical: Vec<_> = plan.series.iter().filter_map(|s| s.electrode_group.map(|g| (s, g))).collect();
    if plan.groups.is_empty() && electrical.is_empty() && plan.extra_electrodes.is_empty() {
        return Ok(());
    }
    b.group("/general/extracellular_ephys", Attrs::new())?;
    for g in &plan.groups {
        let links = json!([{ "source": ".", "path": format!("/general/devices/{}", g.device), "name": "device" }]);
        let a = typed_with("core", "ElectrodeGroup", &[("description", json!(g.description)), ("location", json!(g.location)), ("_LINKS", links)]);
        b.group(&format!("/general/extracellular_ephys/{}", g.name), a)?;
    }

    let (mut location, mut group, mut group_name, mut channel_name) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (s, gi) in &electrical {
        let g = &plan.groups[*gi];
        for c in &session.recordings[s.recording].info().channels {
            location.push(g.location.clone());
            group.push(format!("/general/extracellular_ephys/{}", g.name));
            group_name.push(g.name.clone());
            channel_name.push(c.name.clone());
        }
    }
    // Snippet channels whose group has no electrical series
    for (gi, name) in &plan.extra_electrodes {
        let g = &plan.groups[*gi];
        location.push(g.location.clone());
        group.push(format!("/general/extracellular_ephys/{}", g.name));
        group_name.push(g.name.clone());
        channel_name.push(name.clone());
    }
    let with_imp = plan.impedance_ohms.len() == location.len() && !location.is_empty();
    let mut colnames = vec!["location", "group", "group_name", "channel_name"];
    if with_imp {
        colnames.push("imp");
    }
    let colnames = json!(colnames);
    let table = {
        let mut a = typed("core", "ElectrodesTable");
        a.insert("description".into(), json!("metadata about extracellular electrodes"));
        a.insert("colnames".into(), colnames);
        a
    };
    b.group(TABLE_PATH, table)?;
    b.strings(&format!("{TABLE_PATH}/location"), &location, "dim0", column("Location of the electrode (channel)."))?;
    let mut group_attrs = column("Reference to the ElectrodeGroup.");
    group_attrs.insert("_DTYPE".into(), json!("object_reference"));
    b.strings(&format!("{TABLE_PATH}/group"), &group, "dim0", group_attrs)?;
    b.strings(&format!("{TABLE_PATH}/group_name"), &group_name, "dim0", column("Name of the ElectrodeGroup this electrode is a part of."))?;
    b.strings(&format!("{TABLE_PATH}/channel_name"), &channel_name, "dim0", column("Channel name in the source recording."))?;
    if with_imp {
        b.f32s(&format!("{TABLE_PATH}/imp"), &plan.impedance_ohms, "dim0", column("Impedance of the channel, in ohms."))?;
    }
    let ids: Vec<i64> = (0..location.len() as i64).collect();
    b.i64s(&format!("{TABLE_PATH}/id"), &ids, "num_rows", typed("hdmf-common", "ElementIdentifiers"))
}
