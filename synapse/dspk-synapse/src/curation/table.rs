//! The unit table: one row per unit, its metrics as columns `index` (the unit's position in the
//! sorting), `unit`, `label`, `spikes`, `rate`, `snr`, `isi`, `presence`, `amp_cut`, `score`.
//! Sorting and filtering give a view the unit order; how it shows the columns (headers, widths)
//! is the view's.
//!
//! With the `table` feature (desktop, server) it is a Polars data frame (through `dspk-table`),
//! for any other frame operation. Without it (a browser page: Polars 0.54+ does not build for
//! wasm32-unknown-unknown, pola-rs/polars#28423) the same rows are sorted and filtered as plain
//! values ([`plain_order`]), in the order Polars gives (NaN largest).

use super::sorting::SortingData;

/// The columns, as (name, value of a unit's row): every one but `index` and `label`.
// With Polars (desktop) the plain order is the test's reference only
#[cfg_attr(feature = "table", allow(dead_code))]
const NUMBERS: [&str; 8] = ["unit", "spikes", "rate", "snr", "isi", "presence", "amp_cut", "score"];

/// One unit's row: its index, label and [`NUMBERS`].
#[derive(Debug, Clone)]
struct Row {
    index: usize,
    label: String,
    numbers: [f64; 8],
}

fn rows_of(sorting: &SortingData) -> Vec<Row> {
    sorting
        .alive()
        .map(|i| {
            let x = &sorting.units[i].info;
            Row {
                index: i,
                label: x.quality_label.clone(),
                numbers: [x.unit_id as f64, x.num_spikes as f64, x.firing_rate_hz, x.snr as f64, x.isi_violation_ratio, x.presence_ratio, x.amplitude_cutoff, x.composite_score],
            }
        })
        .collect()
}

fn labels_of(rows: &[Row]) -> Vec<String> {
    let mut labels: Vec<String> = rows.iter().map(|r| r.label.clone()).collect();
    labels.sort();
    labels.dedup();
    labels
}

/// Unit indices of `rows` sorted by `column` (`descending`; NaN is the largest value, as in
/// Polars), only those labelled `label` when given; `None` for a column the table does not have.
#[cfg_attr(feature = "table", allow(dead_code))]
fn plain_order(rows: &[Row], column: &str, descending: bool, label: Option<&str>) -> Option<Vec<usize>> {
    let mut kept: Vec<&Row> = rows.iter().filter(|r| label.is_none_or(|l| r.label == l)).collect();
    // A stable sort either way: equal values keep the units' order, as Polars does
    let flip = |o: std::cmp::Ordering| if descending { o.reverse() } else { o };
    if column == "label" {
        kept.sort_by(|a, b| flip(a.label.cmp(&b.label)));
    } else {
        let c = NUMBERS.iter().position(|n| *n == column)?;
        kept.sort_by(|a, b| flip(a.numbers[c].total_cmp(&b.numbers[c])));
    }
    Some(kept.into_iter().map(|r| r.index).collect())
}

#[cfg(feature = "table")]
pub use frame::UnitTable;
#[cfg(not(feature = "table"))]
pub use plain::UnitTable;

#[cfg(feature = "table")]
mod frame {
    use dspk_table::prelude::*;
    use dspk_table::{Only, TableResult};

    use super::{labels_of, rows_of, SortingData};

    pub struct UnitTable {
        df: DataFrame,
        /// Distinct labels (filter choices), sorted.
        pub labels: Vec<String>,
    }

    impl UnitTable {
        /// The units not merged away.
        pub fn new(sorting: &SortingData) -> PolarsResult<Self> {
            let rows = rows_of(sorting);
            let col = |c: usize| rows.iter().map(|r| r.numbers[c]).collect::<Vec<f64>>();
            let int = |c: usize| rows.iter().map(|r| r.numbers[c] as u32).collect::<Vec<u32>>();
            let df = df!(
                "index" => rows.iter().map(|r| r.index as u32).collect::<Vec<u32>>(),
                "unit" => int(0),
                "label" => rows.iter().map(|r| r.label.clone()).collect::<Vec<String>>(),
                "spikes" => int(1),
                "rate" => col(2),
                "snr" => col(3),
                "isi" => col(4),
                "presence" => col(5),
                "amp_cut" => col(6),
                "score" => col(7),
            )?;
            Ok(Self { df, labels: labels_of(&rows) })
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
}

#[cfg(not(feature = "table"))]
mod plain {
    use super::{labels_of, plain_order, rows_of, Row, SortingData};

    pub struct UnitTable {
        rows: Vec<Row>,
        /// Distinct labels (filter choices), sorted.
        pub labels: Vec<String>,
    }

    impl UnitTable {
        /// The units not merged away.
        pub fn new(sorting: &SortingData) -> Result<Self, std::convert::Infallible> {
            let rows = rows_of(sorting);
            Ok(Self { labels: labels_of(&rows), rows })
        }

        /// Unit indices sorted by `column` (`descending`), only `label` when given.
        pub fn order(&self, column: &str, descending: bool, label: Option<&str>) -> Result<Vec<usize>, String> {
            plain_order(&self.rows, column, descending, label).ok_or_else(|| format!("no column {column}"))
        }
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
        // The browser's table (no Polars) orders as the frame does, NaN as the largest value
        let rows = rows_of(&sorting);
        for (column, descending, label) in [("spikes", true, None), ("spikes", false, Some("good")), ("unit", false, None), ("rate", true, Some("mua"))] {
            assert_eq!(plain_order(&rows, column, descending, label).unwrap(), t.order(column, descending, label).unwrap(), "{column} {descending} {label:?}");
        }
        let mut with_nan = sorting.clone();
        with_nan.units[1].info.snr = f32::NAN;
        let rows = rows_of(&with_nan);
        let frame = UnitTable::new(&with_nan).unwrap();
        for descending in [false, true] {
            assert_eq!(plain_order(&rows, "snr", descending, None).unwrap(), frame.order("snr", descending, None).unwrap(), "NaN, descending {descending}");
        }
        assert!(plain_order(&rows, "missing", false, None).is_none());
    }
}
