//! NWB `/units` tables ↔ [`SortingOutput`]. The table itself is read and written by
//! [`dsp_io::neuro::nwb::NwbUnitsTable`]; this module converts and computes the unit metrics the
//! table lacks.

use std::path::Path;

use dsp_core::DspResult;
use dsp_io::container::npy::NpyArray;
use dsp_io::neuro::nwb::NwbUnitsTable;
use dsp_io::neuro::templates::TemplateAxisOrder;

use crate::core::{
    pack_templates, unpack_template, RecordingMeta, SortedUnit, SortingOutput, UnitQualityLabel,
    WaveformTemplate,
};
use crate::metrics::QualityCriteria;

/// Sorter name of a table that does not record one.
const UNNAMED_SORTER: &str = "nwb_units";

/// The table form of a [`SortingOutput`] (templates `[K, C, T]` on the probe's channels, row
/// order; per-spike amplitudes / locations only when every unit has them).
pub fn from_sorting_output(so: &SortingOutput) -> NwbUnitsTable {
    let (ids, spike_times_sec, spike_times_index) = so.to_ragged_spikes();
    let has_spikes = so.total_spikes() > 0;
    let all = |f: fn(&SortedUnit) -> bool| has_spikes && so.units.iter().all(f);
    let spike_amplitudes = all(|u| u.amplitudes_uv.len() == u.spike_samples.len())
        .then(|| so.units.iter().flat_map(|u| u.amplitudes_uv.iter().copied()).collect());
    let spike_locations = all(|u| u.locations_um.len() == u.spike_samples.len())
        .then(|| so.units.iter().flat_map(|u| u.locations_um.iter().copied()).collect());

    let probe_channels = so.probe.as_ref().map_or(0, |p| p.total_channels());
    let packed = pack_templates(
        so.units.iter().enumerate().map(|(row, u)| (row, u.template.as_ref())),
        probe_channels,
        TemplateAxisOrder::ChannelsSamples,
    );
    let (waveform_mean, waveform_sd, waveform_se) = match packed {
        Some((shape, mean, sd, se)) => (Some(NpyArray { data: mean, shape: shape.to_vec() }), Some(sd), Some(se)),
        None => (None, None, None),
    };

    NwbUnitsTable {
        nwb_root: None,
        sample_rate_hz: so.sample_rate_hz.max(1.0),
        sorter_name: Some(so.sorter_name.clone()),
        total_samples: Some(so.total_samples),
        ids,
        spike_times_sec,
        spike_times_index,
        spike_amplitudes,
        spike_locations,
        snr: Some(so.units.iter().map(|u| u.snr).collect()),
        firing_rate: Some(so.units.iter().map(|u| u.firing_rate_hz as f32).collect()),
        primary_channel: Some(so.units.iter().map(|u| u.primary_channel).collect()),
        waveform_mean,
        waveform_sd,
        waveform_se,
        quality_labels: so.units.iter().map(|u| u.quality_label.as_str().to_string()).collect(),
        probe: so.probe.clone(),
    }
}

/// A table as a [`SortingOutput`]: metrics computed from the spikes, then stored SNR and labels
/// applied.
pub fn to_sorting_output(table: &NwbUnitsTable) -> SortingOutput {
    let fs = table.sample_rate_hz;
    let unpacked = SortingOutput::unpack_ragged_spikes(&table.ids, &table.spike_times_sec, &table.spike_times_index, fs);
    let total_samples = table
        .total_samples
        .unwrap_or_else(|| unpacked.iter().filter_map(|(_, s)| s.last().copied()).max().unwrap_or(0));

    let criteria = QualityCriteria::default();
    let units = unpacked
        .into_iter()
        .enumerate()
        .map(|(row, (uid, spike_samples))| {
            let spikes = table.spikes_of(row);
            let amps = table.spike_amplitudes.as_ref().and_then(|a| a.get(spikes.clone())).map(<[f32]>::to_vec).unwrap_or_default();
            let locs = table.spike_locations.as_ref().and_then(|l| l.get(spikes.clone())).map(<[_]>::to_vec).unwrap_or_default();
            let primary = table.primary_channel.as_ref().and_then(|c| c.get(row).copied());

            let template = table.waveform_mean.as_ref().and_then(|arr| match *arr.shape.as_slice() {
                [n_u, n_c, n_s] => unpack_template(
                    row,
                    [n_u, n_c, n_s],
                    TemplateAxisOrder::ChannelsSamples,
                    &arr.data,
                    table.waveform_sd.as_deref(),
                    table.waveform_se.as_deref(),
                    spike_samples.len(),
                ),
                // Single-channel snippet units (neuro-convert): [K, T] on the primary channel
                [n_u, n_s] if row < n_u && n_s > 0 => {
                    let off = row * n_s;
                    let mean = arr.data[off..off + n_s].to_vec();
                    let std = table.waveform_sd.as_ref().map(|s| s[off..off + n_s].to_vec()).unwrap_or_else(|| vec![1.0; n_s]);
                    Some(WaveformTemplate::with_count(vec![primary.unwrap_or(0)], n_s, spike_samples.len(), mean, std))
                }
                _ => None,
            });

            let mut unit = SortedUnit::from_spikes_with(uid, primary, spike_samples, amps, locs, template, fs, total_samples, None, criteria);
            if let Some(&snr) = table.snr.as_ref().and_then(|s| s.get(row)) {
                unit.snr = snr;
                unit.quality_label = criteria.classify(unit.num_spikes(), snr, unit.isi_violation_ratio);
            }
            if let Some(label) = table.quality_labels.get(row) {
                unit.quality_label = UnitQualityLabel::parse(label);
            }
            unit
        })
        .collect();

    let sorter = table.sorter_name.clone().unwrap_or_else(|| UNNAMED_SORTER.to_string());
    let mut out = SortingOutput::new(sorter, fs, total_samples, table.probe.clone(), units, None);
    if let Some(root) = &table.nwb_root {
        out = out.with_recording_meta(RecordingMeta { dat_path: Some(root.to_string_lossy().into_owned()), ..Default::default() });
    }
    out
}

/// Saves a [`SortingOutput`] as the `/units` group of the NWB store `nwb_zarr_dir`.
///
/// # Errors
///
/// Fails ([`DspError::Io`](dsp_core::DspError::Io)) when the path cannot be written.
pub fn save_nwb_units(sorting: &SortingOutput, nwb_zarr_dir: &Path) -> DspResult<()> {
    from_sorting_output(sorting).write(nwb_zarr_dir)
}

/// Loads the `/units` table at `nwb_zarr_dir` (an NWB root or the `units` group). The sample rate
/// is `sample_rate_hz` when given, else inferred from the store.
///
/// # Errors
///
/// Fails ([`DspError::Io`](dsp_core::DspError::Io) and others) when the path cannot be read or is not in that format.
pub fn load_nwb_units(nwb_zarr_dir: &Path, sample_rate_hz: impl Into<Option<f64>>) -> DspResult<SortingOutput> {
    Ok(to_sorting_output(&NwbUnitsTable::read(nwb_zarr_dir, sample_rate_hz.into())?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nwb_units_roundtrip_and_inferred_rate() {
        let dir = std::env::temp_dir().join(format!("dsp_nwb_units_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Include a timestamp past 2^24 samples to verify f64 precision is preserved
        let times = vec![300u64, 1500, 20_000_003];
        let t_mean = vec![0.0f32; 2 * 25];
        let template = WaveformTemplate::with_count(vec![0, 1], 25, 3, t_mean, vec![1.0; 50]);

        let unit0 = SortedUnit::from_spikes(0, 1, times.clone(), Vec::new(), Vec::new(), Some(template), 20_000.0, 20_001_000, 8.0);
        let orig = SortingOutput::new("nwb_sorter", 20_000.0, 20_001_000, None, vec![unit0], None);

        save_nwb_units(&orig, &dir).unwrap();
        for node in ["", "id", "spike_times", "spike_times_index"] {
            assert!(dir.join("units").join(node).join("zarr.json").exists(), "{node}");
        }

        // Load with None: sample rate (20_000.0 Hz) is inferred from Zarr metadata
        let loaded = load_nwb_units(&dir, None).unwrap();
        assert_eq!(loaded.sample_rate_hz, 20_000.0);
        assert_eq!(loaded.units.len(), 1);
        assert_eq!(loaded.units[0].primary_channel, 1);
        assert_eq!(loaded.units[0].spike_samples, times);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
