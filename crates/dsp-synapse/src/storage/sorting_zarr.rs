//! `.sorting.zarr` stores ↔ [`SortingOutput`]. The store itself is read and written by
//! [`dsp_io::neuro::sorting_zarr::SortingZarr`]; it keeps every unit metric, so nothing is
//! recomputed on load.

use std::path::Path;

use dsp_core::DspResult;
use dsp_io::container::npy::NpyArray;
use dsp_io::neuro::sorting_zarr::{SortingZarr, SortingZarrManifest, SortingZarrUnit, SORTING_ZARR_FORMAT};
use dsp_io::neuro::templates::TemplateAxisOrder;
use serde_json::Value;

use crate::core::{pack_templates, unpack_template, SortedUnit, SortingOutput, UnitQualityLabel};

/// A label as the store writes it (its serde name, e.g. `SingleUnit`).
fn label_text(label: UnitQualityLabel) -> String {
    match serde_json::to_value(label) {
        Ok(Value::String(s)) => s,
        _ => label.as_str().to_string(),
    }
}

/// A stored label: serde names and their aliases, else Phy names.
fn parse_label(text: &str) -> UnitQualityLabel {
    serde_json::from_value(Value::String(text.to_string())).unwrap_or_else(|_| UnitQualityLabel::parse(text))
}

/// The store form of a [`SortingOutput`] (templates `[units, channels, samples]` on the probe's
/// channels, unit row order).
pub fn from_sorting_output(so: &SortingOutput) -> SortingZarr {
    let units = so
        .units
        .iter()
        .map(|u| SortingZarrUnit {
            unit_id: u.unit_id,
            primary_channel: u.primary_channel,
            quality_label: label_text(u.quality_label),
            snr: u.snr,
            firing_rate_hz: u.firing_rate_hz,
            isi_violation_ratio: u.isi_violation_ratio,
            presence_ratio: u.presence_ratio,
            amplitude_cutoff: u.amplitude_cutoff,
            num_spikes: u.spike_samples.len(),
        })
        .collect();
    let manifest = SortingZarrManifest {
        dsp_format: SORTING_ZARR_FORMAT.into(),
        sorter_name: so.sorter_name.clone(),
        sample_rate_hz: so.sample_rate_hz,
        total_samples: so.total_samples,
        recording_meta: serde_json::to_value(&so.recording_meta).unwrap_or(Value::Null),
        probe: so.probe.clone(),
        drift: so.drift.as_ref().and_then(|d| serde_json::to_value(d).ok()),
        units,
    };

    let (spike_times, spike_clusters, amplitudes, locations) = so.flattened_spikes();
    let probe_channels = so.probe.as_ref().map_or(0, |p| p.total_channels());
    let packed = pack_templates(
        so.units.iter().enumerate().map(|(row, u)| (row, u.template.as_ref())),
        probe_channels,
        TemplateAxisOrder::ChannelsSamples,
    );
    let (templates_mean, templates_std, templates_se) = match packed {
        Some((shape, mean, sd, se)) => (Some(NpyArray { data: mean, shape: shape.to_vec() }), Some(sd), Some(se)),
        None => (None, None, None),
    };
    SortingZarr { manifest, spike_times, spike_clusters, amplitudes, locations, templates_mean, templates_std, templates_se }
}

/// A store as a [`SortingOutput`] (stored metrics and labels kept as written).
pub fn to_sorting_output(store: &SortingZarr) -> SortingOutput {
    let m = &store.manifest;
    let mut groups =
        SortingOutput::group_spikes_by_cluster(&store.spike_times, &store.spike_clusters, &store.amplitudes, &store.locations);
    let units = m
        .units
        .iter()
        .enumerate()
        .map(|(row, meta)| {
            let (spike_samples, amplitudes_uv, locations_um) = groups.remove(&meta.unit_id).unwrap_or_default();
            let template = store.templates_mean.as_ref().and_then(|t| {
                let shape: [usize; 3] = t.shape.as_slice().try_into().ok()?;
                // Row order; older stores indexed templates by unit id
                let slot = if row < shape[0] && m.units.len() == shape[0] { row } else { meta.unit_id };
                unpack_template(
                    slot,
                    shape,
                    TemplateAxisOrder::ChannelsSamples,
                    &t.data,
                    store.templates_std.as_deref(),
                    store.templates_se.as_deref(),
                    meta.num_spikes.max(spike_samples.len()),
                )
            });
            SortedUnit {
                unit_id: meta.unit_id,
                primary_channel: meta.primary_channel,
                spike_samples,
                amplitudes_uv,
                locations_um,
                template,
                quality_label: parse_label(&meta.quality_label),
                snr: meta.snr,
                firing_rate_hz: meta.firing_rate_hz,
                isi_violation_ratio: meta.isi_violation_ratio,
                presence_ratio: meta.presence_ratio,
                amplitude_cutoff: meta.amplitude_cutoff,
            }
        })
        .collect();

    let drift = m.drift.clone().and_then(|d| serde_json::from_value(d).ok());
    SortingOutput::new(m.sorter_name.clone(), m.sample_rate_hz, m.total_samples, m.probe.clone(), units, drift)
        .with_recording_meta(serde_json::from_value(m.recording_meta.clone()).unwrap_or_default())
}

/// Saves a [`SortingOutput`] as a `.sorting.zarr` store.
pub fn save_sorting_zarr(sorting: &SortingOutput, dir: &Path) -> DspResult<()> {
    from_sorting_output(sorting).write(dir)
}

/// Loads a `.sorting.zarr` store as a [`SortingOutput`].
pub fn load_sorting_zarr(dir: &Path) -> DspResult<SortingOutput> {
    Ok(to_sorting_output(&SortingZarr::read(dir)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WaveformTemplate;
    use dsp_io::neuro::probe::{Position3D, SensorLayout, SensorSite};

    #[test]
    fn test_sorting_zarr_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dsp_zarr_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let times = vec![500u64, 1200, 2900];
        let amps = vec![95.0f32, 105.0, 100.0];
        let locs = vec![[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]];
        let t_mean = vec![0.0f32; 2 * 20];
        let template = WaveformTemplate::with_count(vec![0, 1], 20, 3, t_mean, vec![1.0; 40]);

        let unit0 = SortedUnit::from_spikes(5, 1, times, amps, locs, Some(template), 30_000.0, 5000, 10.0);
        let contacts = vec![
            SensorSite::new(0, Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, Position3D::new(0.0, 20.0, 0.0), 0),
        ];
        let probe = SensorLayout::new("2ch", contacts);
        let orig = SortingOutput::new("zarr_test_sorter", 30_000.0, 5000, Some(probe), vec![unit0], None);

        save_sorting_zarr(&orig, &dir).unwrap();
        assert!(dir.join("zarr.json").exists());
        assert!(dir.join("spikes").join("times").join("zarr.json").exists());
        assert!(dir.join("templates").join("mean").join("zarr.json").exists());

        let loaded = load_sorting_zarr(&dir).unwrap();
        assert_eq!(loaded.sorter_name, "zarr_test_sorter");
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].unit_id, 5);
        assert_eq!(loaded.units[0].primary_channel, 1);
        assert_eq!(loaded.units[0].quality_label, orig.units[0].quality_label);
        assert_eq!(loaded.units[0].spike_samples, orig.units[0].spike_samples);
        assert_eq!(loaded.units[0].locations_um, orig.units[0].locations_um);
        assert!(loaded.units[0].template.is_some());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
