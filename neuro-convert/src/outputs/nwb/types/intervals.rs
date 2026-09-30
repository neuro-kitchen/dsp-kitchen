//! `/intervals/<name>` (`TimeIntervals`): events with onsets and offsets, plus their values.

use serde_json::json;

use super::{column, typed};
use crate::error::Result;
use crate::model::EventSeries;
use crate::outputs::nwb::backend::{Attrs, Backend};
use crate::outputs::nwb::mapping::EventPlan;

pub fn write_group(b: &dyn Backend) -> Result<()> {
    b.group("/intervals", Attrs::new())
}

pub fn write(b: &dyn Backend, plan: &EventPlan, e: &EventSeries) -> Result<()> {
    let path = format!("/intervals/{}", plan.name);
    let mut a = typed("core", "TimeIntervals");
    a.insert("description".into(), json!(plan.description));
    a.insert("colnames".into(), json!(["start_time", "stop_time", "value"]));
    b.group(&path, a)?;

    let n = e.len() as u64;
    let offsets = e.offsets.as_deref().unwrap_or(&e.onsets);
    b.f64s(&format!("{path}/start_time"), &e.onsets, &[n], &["dim0"], column("Start time of epoch, in seconds"))?;
    b.f64s(&format!("{path}/stop_time"), offsets, &[n], &["dim0"], column("Stop time of epoch, in seconds"))?;
    b.f64s(&format!("{path}/value"), &e.values, &[n], &["dim0"], column("Value recorded with the event"))?;
    let ids: Vec<i64> = (0..n as i64).collect();
    b.i64s(&format!("{path}/id"), &ids, "num_rows", typed("hdmf-common", "ElementIdentifiers"))
}
