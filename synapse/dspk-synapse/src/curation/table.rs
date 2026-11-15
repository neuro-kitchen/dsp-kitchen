//! The unit table as a Polars data frame (through `dspk-table`): one row per unit, its metrics
//! as columns `index` (the unit's position in the sorting), `unit`, `label`, `spikes`, `rate`,
//! `snr`, `isi`, `presence`, `amp_cut`, `score`. Sorting and filtering are frame operations; a
//! view reads the resulting unit order. How a view shows the columns (headers, widths) is the
//! view's.

use dspk_table::prelude::*;
use dspk_table::{Only, TableResult};

use super::sorting::SortingData;

pub struct UnitTable {
    df: DataFrame,
    /// Distinct labels (filter choices), sorted.
    pub labels: Vec<String>,
}

impl UnitTable {
    /// The units not merged away.
    pub fn new(sorting: &SortingData) -> PolarsResult<Self> {
        let index: Vec<u32> = sorting.alive().map(|i| i as u32).collect();
        let u: Vec<&super::sorting::Unit> = index.iter().map(|&i| &sorting.units[i as usize]).collect();
        let f = |g: fn(&crate::io::sorting_zarr::SortingZarrUnit) -> f64| u.iter().map(|x| g(&x.info)).collect::<Vec<f64>>();
        let df = df!(
            "index" => index.clone(),
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
    pub fn order(&self, column: &str, descending: bool, label: Option<&str>) -> TableResult<Vec<usize>> {
        dspk_table::order(&self.df, "index", column, descending, label.map(|value| Only { column: "label", value }))
    }

    /// The frame itself, for any other table operation.
    pub fn frame(&self) -> &DataFrame {
        &self.df
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sorting::{Extent, Unit};
    use crate::io::sorting_zarr::SortingZarrUnit;

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
        Unit { info, extent: Extent { before: 1, after: 1, channels: vec![0] }, template: Vec::new(), template_row: None, merged_into: None }
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
            by_unit: Vec::new(),
            positions: Vec::new(),
            templates: None,
        };
        let t = UnitTable::new(&sorting).unwrap();
        assert_eq!(t.labels, ["good", "mua"]);
        assert_eq!(t.order("spikes", true, None).unwrap(), [1, 2, 0]);
        assert_eq!(t.order("spikes", false, Some("good")).unwrap(), [0, 2]);
    }
}
