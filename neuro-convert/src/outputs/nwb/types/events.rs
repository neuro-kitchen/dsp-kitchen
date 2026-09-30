//! `/events/<name>` (`EventsTable`, NWB 2.9+): one row per event with its timestamp, duration
//! (offset − onset, NaN when the event has no offset), recorded value and annotation (label).

use serde_json::json;

use super::{column, typed, typed_with};
use crate::error::Result;
use crate::model::EventSeries;
use crate::outputs::nwb::backend::{Attrs, Backend};
use crate::outputs::nwb::mapping::EventPlan;

pub fn write_group(b: &dyn Backend) -> Result<()> {
    b.group("/events", Attrs::new())
}

pub fn write(b: &dyn Backend, plan: &EventPlan, e: &EventSeries) -> Result<()> {
    let path = format!("/events/{}", plan.name);
    let n = e.len() as u64;
    let annotated = e.labels.iter().any(|l| !l.is_empty());
    let mut colnames = vec!["timestamp"];
    if e.offsets.is_some() {
        colnames.push("duration");
    }
    colnames.push("value");
    if annotated {
        colnames.push("annotation");
    }
    let mut a = typed("core", "EventsTable");
    a.insert("description".into(), json!(plan.description));
    a.insert("colnames".into(), json!(colnames));
    b.group(&path, a)?;

    let seconds = |kind: &str, desc: &str| typed_with("core", kind, &[("description", json!(desc)), ("unit", json!("seconds"))]);
    b.f64s(&format!("{path}/timestamp"), &e.onsets, &[n], &["num_times"], seconds("TimestampVectorData", "The time that each event occurred, in seconds, from the session start time."))?;
    if let Some(off) = &e.offsets {
        let durations: Vec<f64> = e.onsets.iter().zip(off).map(|(a, b)| b - a).collect();
        b.f64s(&format!("{path}/duration"), &durations, &[n], &["num_times"], seconds("DurationVectorData", "The duration of each event, in seconds."))?;
    }
    b.f64s(&format!("{path}/value"), &e.values, &[n], &["dim0"], column("Value recorded with the event"))?;
    if annotated {
        b.strings(&format!("{path}/annotation"), &e.labels, "num_times", column("User annotations about events."))?;
    }
    let ids: Vec<i64> = (0..n as i64).collect();
    b.i64s(&format!("{path}/id"), &ids, "num_rows", typed("hdmf-common", "ElementIdentifiers"))
}
