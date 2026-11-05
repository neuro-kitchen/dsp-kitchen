//! The unit table as a Polars data frame: one row per unit, its metrics as columns; sorting and
//! filtering are frame operations, the views read the resulting unit order.

use polars::prelude::*;

use crate::sorting::SortingData;

/// Columns, in display order: (key in the frame, header, width in logical px, decimals).
pub const COLUMNS: [(&str, &str, f32, usize); 9] = [
    ("unit", "Unit", 64., 0),
    ("label", "Label", 72., 0),
    ("spikes", "Spikes", 64., 0),
    ("rate", "Rate", 56., 2),
    ("snr", "SNR", 48., 1),
    ("isi", "ISI viol.", 64., 3),
    ("presence", "Presence", 64., 2),
    ("amp_cut", "Amp. cut", 64., 3),
    ("score", "Score", 48., 2),
];

pub struct UnitTable {
    df: DataFrame,
    /// Distinct labels (filter choices), sorted.
    pub labels: Vec<String>,
}

impl UnitTable {
    pub fn new(sorting: &SortingData) -> PolarsResult<Self> {
        let u = &sorting.units;
        let f = |g: fn(&dspk_synapse::io::sorting_zarr::SortingZarrUnit) -> f64| u.iter().map(|x| g(&x.info)).collect::<Vec<f64>>();
        let df = df!(
            "index" => (0..u.len() as u32).collect::<Vec<u32>>(),
            "unit" => u.iter().map(|x| x.info.unit_id as u32).collect::<Vec<u32>>(),
            "label" => u.iter().map(|x| x.info.quality_label.clone()).collect::<Vec<String>>(),
            "spikes" => u.iter().map(|x| x.info.num_spikes as u32).collect::<Vec<u32>>(),
            "rate" => f(|i| i.firing_rate_hz),
            "snr" => f(|i| i.snr as f64),
            "isi" => f(|i| i.isi_violation_ratio),
            "presence" => f(|i| i.presence_ratio),
            "amp_cut" => f(|i| i.amplitude_cutoff),
            "score" => f(|i| i.composite_score),
        )?;
        let mut labels: Vec<String> = u.iter().map(|x| x.info.quality_label.clone()).collect();
        labels.sort();
        labels.dedup();
        Ok(Self { df, labels })
    }

    /// Unit indices sorted by `column` (`descending`), only `label` when given.
    pub fn order(&self, column: &str, descending: bool, label: Option<&str>) -> PolarsResult<Vec<usize>> {
        let mut df = self.df.clone();
        if let Some(l) = label {
            let mask = df.column("label")?.str()?.equal(l);
            df = df.filter(&mask)?;
        }
        let df = df.sort([column], SortMultipleOptions::default().with_order_descending(descending).with_nulls_last(true))?;
        Ok(df.column("index")?.u32()?.into_no_null_iter().map(|i| i as usize).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorting::{Extent, Unit};
    use dspk_synapse::io::sorting_zarr::SortingZarrUnit;

    fn unit(id: usize, label: &str, spikes: usize) -> Unit {
        let info = SortingZarrUnit {
            unit_id: id,
            primary_channel: 0,
            quality_label: label.into(),
            snr: 1.0,
            firing_rate_hz: spikes as f64 / 10.0,
            isi_violation_ratio: 0.0,
            presence_ratio: 1.0,
            amplitude_cutoff: 0.0,
            composite_score: f64::NAN,
            num_spikes: spikes,
        };
        Unit { info, extent: Extent { before: 1, after: 1, channels: vec![0] }, template: Vec::new() }
    }

    #[test]
    fn test_sort_and_filter() {
        let sorting = SortingData {
            path: "/x".into(),
            sorter: "t".into(),
            sample_rate: 1000.0,
            times: Vec::new(),
            unit_of: Vec::new(),
            amplitudes: Vec::new(),
            units: vec![unit(5, "good", 10), unit(6, "mua", 30), unit(7, "good", 20)],
            by_id: Default::default(),
            channels: None,
        };
        let t = UnitTable::new(&sorting).unwrap();
        assert_eq!(t.labels, ["good", "mua"]);
        assert_eq!(t.order("spikes", true, None).unwrap(), [1, 2, 0]);
        assert_eq!(t.order("spikes", false, Some("good")).unwrap(), [0, 2]);
    }
}
