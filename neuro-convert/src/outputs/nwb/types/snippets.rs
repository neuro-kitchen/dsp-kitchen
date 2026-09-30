//! Spike snippets (NWB 2.11): `/acquisition/<name>_ch<c>` (`SpikeEventSeries`, one per channel,
//! since all events of a series span the same electrodes) and `/units` (`Units`) for sorted
//! snippets (non-zero sort codes), one unit per channel × sort code.

use serde_json::json;

use super::electrodes::TABLE_PATH;
use super::{attrs, column, typed, typed_with};
use crate::error::Result;
use crate::model::{SampleType, SnippetSeries};
use crate::outputs::nwb::backend::Backend;
use crate::outputs::nwb::mapping::SnippetPlan;

fn reference(path: &str) -> serde_json::Value {
    json!({ "_REFERENCE": { "source": ".", "path": path } })
}

/// A 2-D float32 dataset written in one piece.
fn f32_matrix(b: &dyn Backend, path: &str, rows: usize, cols: usize, values: &[f32], dims: &[&str], a: super::super::backend::Attrs) -> Result<()> {
    let sink = b.stream(path, &[rows as u64, cols as u64], rows.max(1) as u64, SampleType::F32, dims, a)?;
    if rows > 0 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        sink.write_rows(0, rows as u64, &bytes)?;
    }
    Ok(())
}

/// One `SpikeEventSeries` per channel of `plan`.
pub fn write(b: &dyn Backend, plan: &SnippetPlan, sn: &SnippetSeries) -> Result<()> {
    let w = sn.samples_per_snippet;
    for &(channel, row) in &plan.rows {
        let events: Vec<usize> = (0..sn.len()).filter(|&i| sn.channels[i] == channel).collect();
        let path = format!("/acquisition/{}_ch{channel}", plan.name);
        let description = format!("{} (channel {channel})", plan.description);
        b.group(&path, typed_with("core", "SpikeEventSeries", &[("description", json!(description)), ("comments", json!("no comments"))]))?;

        // [num_events, num_channels = 1, num_samples]: the channel axis matches `electrodes`
        let data: Vec<f32> = events.iter().flat_map(|&i| sn.data[i * w..(i + 1) * w].iter().copied()).collect();
        let data_attrs = attrs(&[("unit", json!("volts")), ("conversion", json!(plan.conversion)), ("offset", json!(0.0)), ("resolution", json!(-1.0))]);
        let sink = b.stream(&format!("{path}/data"), &[events.len() as u64, 1, w as u64], events.len().max(1) as u64, SampleType::F32, &["num_events", "num_channels", "num_samples"], data_attrs)?;
        if !events.is_empty() {
            sink.write_rows(0, events.len() as u64, &data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }

        let times: Vec<f64> = events.iter().map(|&i| sn.timestamps[i]).collect();
        b.f64s(&format!("{path}/timestamps"), &times, &[times.len() as u64], &["num_times"], attrs(&[("interval", json!(1)), ("unit", json!("seconds"))]))?;

        let region = typed_with("hdmf-common", "DynamicTableRegion", &[("description", json!("electrode of this channel")), ("table", reference(TABLE_PATH))]);
        b.i64s(&format!("{path}/electrodes"), &[row as i64], "num_rows", region)?;
    }
    Ok(())
}

/// A sorted unit: the snippets of one channel with one non-zero sort code.
struct Unit<'a> {
    plan: &'a SnippetPlan,
    sn: &'a SnippetSeries,
    channel: u16,
    row: usize,
    sort_code: u16,
    events: Vec<usize>,
}

/// `/units` for every non-zero sort code of `stores`; nothing when no snippet is sorted.
pub fn write_units(b: &dyn Backend, stores: &[(&SnippetPlan, &SnippetSeries)]) -> Result<()> {
    let mut units: Vec<Unit> = Vec::new();
    for &(plan, sn) in stores {
        for &(channel, row) in &plan.rows {
            let mut codes: Vec<u16> =
                (0..sn.len()).filter(|&i| sn.channels[i] == channel && sn.sort_codes.get(i).is_some_and(|&c| c != 0)).map(|i| sn.sort_codes[i]).collect();
            codes.sort_unstable();
            codes.dedup();
            for sort_code in codes {
                let events = (0..sn.len()).filter(|&i| sn.channels[i] == channel && sn.sort_codes[i] == sort_code).collect();
                units.push(Unit { plan, sn, channel, row, sort_code, events });
            }
        }
    }
    if units.is_empty() {
        return Ok(());
    }
    // One waveform length per waveform_mean column
    let w = units[0].sn.samples_per_snippet;
    let with_waveforms = units.iter().all(|u| u.sn.samples_per_snippet == w);

    let mut colnames = vec!["spike_times", "electrodes"];
    if with_waveforms {
        colnames.push("waveform_mean");
    }
    colnames.extend(["source_store", "source_channel", "sort_code"]);
    let mut a = typed("core", "Units");
    a.insert("description".into(), json!("units from online / offline sorted spike snippets (one per channel and sort code)"));
    a.insert("colnames".into(), json!(colnames));
    b.group("/units", a)?;

    let n = units.len();
    let mut spike_times = Vec::new();
    let mut spike_index = Vec::with_capacity(n);
    for u in &units {
        let mut t: Vec<f64> = u.events.iter().map(|&i| u.sn.timestamps[i]).collect();
        t.sort_by(f64::total_cmp);
        spike_times.extend(t);
        spike_index.push(spike_times.len() as i64);
    }
    b.f64s("/units/spike_times", &spike_times, &[spike_times.len() as u64], &["num_spikes"], column("the spike times for each unit in seconds"))?;
    let index = |target: &str, desc: &str| typed_with("hdmf-common", "VectorIndex", &[("description", json!(desc)), ("target", reference(target))]);
    b.i64s("/units/spike_times_index", &spike_index, "num_rows", index("/units/spike_times", "Index for VectorData 'spike_times'"))?;

    let rows: Vec<i64> = units.iter().map(|u| u.row as i64).collect();
    let region = typed_with("hdmf-common", "DynamicTableRegion", &[("description", json!("electrode of each unit")), ("table", reference(TABLE_PATH))]);
    b.i64s("/units/electrodes", &rows, "num_electrodes", region)?;
    let electrode_index: Vec<i64> = (1..=n as i64).collect();
    b.i64s("/units/electrodes_index", &electrode_index, "num_rows", index("/units/electrodes", "Index for VectorData 'electrodes'"))?;

    if with_waveforms {
        let mut mean = vec![0.0f32; n * w];
        for (k, u) in units.iter().enumerate() {
            let mut acc = vec![0.0f64; w];
            for &i in &u.events {
                for (a, &v) in acc.iter_mut().zip(&u.sn.data[i * w..(i + 1) * w]) {
                    *a += v as f64;
                }
            }
            // Mean in volts (the Units waveform columns carry no conversion)
            let scale = u.plan.conversion / u.events.len().max(1) as f64;
            for (m, a) in mean[k * w..(k + 1) * w].iter_mut().zip(acc) {
                *m = (a * scale) as f32;
            }
        }
        let mut a = column("the spike waveform mean for each spike unit");
        a.insert("sampling_rate".into(), json!(units[0].sn.sample_rate as f32));
        a.insert("unit".into(), json!("volts"));
        f32_matrix(b, "/units/waveform_mean", n, w, &mean, &["num_units", "num_samples"], a)?;
    }

    let stores: Vec<String> = units.iter().map(|u| u.plan.source.clone()).collect();
    b.strings("/units/source_store", &stores, "num_rows", column("snippet store the unit was sorted from"))?;
    let channels: Vec<i64> = units.iter().map(|u| u.channel as i64).collect();
    b.i64s("/units/source_channel", &channels, "num_rows", column("channel of the snippet store (as numbered by the source)"))?;
    let codes: Vec<i64> = units.iter().map(|u| u.sort_code as i64).collect();
    b.i64s("/units/sort_code", &codes, "num_rows", column("sort code of the unit in the source"))?;
    let ids: Vec<i64> = (0..n as i64).collect();
    b.i64s("/units/id", &ids, "num_rows", typed("hdmf-common", "ElementIdentifiers"))
}
