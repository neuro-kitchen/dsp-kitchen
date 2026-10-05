//! NWB `/units` tables. Holds the helpers shared with the sorting readers until they move here
//! (phase 3 of the dsp-io restructure).

use std::path::Path;

use serde_json::Value;

use crate::container::zarr::read_node_attributes;

/// Infers the recording sample rate (Hz) from an NWB Zarr store without hardcoding 30 kHz:
/// 1. `/units` `sample_rate_hz` or `sampling_rate` attribute
/// 2. `/units/waveform_mean` `sampling_rate` attribute
/// 3. `1.0 / resolution` on `/units/spike_times` (written by `neuro-convert`)
/// 4. Any `/acquisition/<series>/starting_time` `rate` attribute in the parent NWB store
pub fn infer_nwb_sample_rate(nwb_or_units_dir: &Path) -> Option<f64> {
    let (root_dir, units_prefix) = if nwb_or_units_dir.join("units").is_dir() {
        (nwb_or_units_dir, "/units")
    } else {
        (nwb_or_units_dir.parent().unwrap_or(nwb_or_units_dir), "")
    };
    let base = if units_prefix.is_empty() { nwb_or_units_dir } else { root_dir };

    if let Some(attrs) = read_node_attributes(base, units_prefix) {
        if let Some(sr) = attrs.get("sample_rate_hz").or_else(|| attrs.get("sampling_rate")).and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some(sr);
        }
    }
    let wm_path = format!("{units_prefix}/waveform_mean");
    if let Some(attrs) = read_node_attributes(base, &wm_path) {
        if let Some(sr) = attrs.get("sampling_rate").and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some(sr);
        }
    }
    let st_path = format!("{units_prefix}/spike_times");
    if let Some(attrs) = read_node_attributes(base, &st_path) {
        if let Some(res) = attrs.get("resolution").and_then(Value::as_f64).filter(|&r| r > 0.0) {
            return Some((1.0 / res).round());
        }
    }
    if let Ok(entries) = std::fs::read_dir(root_dir.join("acquisition")) {
        for entry in entries.flatten() {
            let st = format!("/acquisition/{}/starting_time", entry.file_name().to_string_lossy());
            if let Some(attrs) = read_node_attributes(root_dir, &st) {
                if let Some(rate) = attrs.get("rate").and_then(Value::as_f64).filter(|&r| r > 0.0) {
                    return Some(rate);
                }
            }
        }
    }
    None
}
