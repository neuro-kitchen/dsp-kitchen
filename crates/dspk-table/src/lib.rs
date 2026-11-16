//! Tables for dspk, as [Polars](https://pola.rs) data frames.
//!
//! This crate is the one place that depends on Polars: the other crates use the re-export
//! ([`polars`], [`prelude`]), so the whole workspace builds one Polars with one feature set.
//! It holds what is true of any table, whatever the domain:
//!
//! - [`write_parquet`] / [`read_parquet`]: Parquet, the table format of dspk's outputs;
//! - [`order`]: the row order a view asks for (sorted by a column, optionally only the rows with
//!   one value of a text column), as indices into the caller's own records.
//!
//! Domain tables are built on it in their own crates: the events of a `crossings` chain in
//! `dspk-pipeline`, the units of a sorting in `dspk-synapse`.

use std::fs::File;
use std::path::Path;

pub use polars;
pub use polars::prelude;

use polars::prelude::*;

/// What can go wrong with a table.
#[derive(Debug, thiserror::Error)]
pub enum TableError {
    /// The file could not be created or opened.
    #[error("{0}: {1}")]
    Io(String, std::io::Error),
    /// Polars refused the operation (a missing column, a wrong type, a damaged file).
    #[error("table: {0}")]
    Polars(#[from] PolarsError),
}

/// Result of a table operation.
pub type TableResult<T> = Result<T, TableError>;

/// Writes `table` as a Parquet file at `path` (replacing it). Returns the bytes written.
pub fn write_parquet(path: &Path, table: &mut DataFrame) -> TableResult<u64> {
    let file = File::create(path).map_err(|e| TableError::Io(path.display().to_string(), e))?;
    Ok(ParquetWriter::new(file).finish(table)?)
}

/// Reads the Parquet file at `path`.
pub fn read_parquet(path: &Path) -> TableResult<DataFrame> {
    let file = File::open(path).map_err(|e| TableError::Io(path.display().to_string(), e))?;
    Ok(ParquetReader::new(file).finish()?)
}

/// Only the rows whose text column `column` equals `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Only<'a> {
    pub column: &'a str,
    pub value: &'a str,
}

/// The values of the u32 column `index` in the order a view shows the rows: sorted by `by`
/// (`descending`; nulls last), keeping only the rows `only` selects when given.
///
/// `index` is the caller's own row key (e.g. a unit's position in its sorting), so the result
/// indexes the caller's records, not this frame.
pub fn order(table: &DataFrame, index: &str, by: &str, descending: bool, only: Option<Only<'_>>) -> TableResult<Vec<usize>> {
    let mut table = table.clone();
    if let Some(only) = only {
        let mask = table.column(only.column)?.str()?.equal(only.value);
        table = table.filter(&mask)?;
    }
    let sorted = table.sort([by], SortMultipleOptions::default().with_order_descending(descending).with_nulls_last(true))?;
    Ok(sorted.column(index)?.u32()?.into_no_null_iter().map(|i| i as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units() -> DataFrame {
        df!(
            "index" => [0u32, 1, 2, 3],
            "label" => ["good", "mua", "good", "noise"],
            "rate" => [Some(4.0f64), Some(9.5), None, Some(1.25)],
        )
        .unwrap()
    }

    #[test]
    fn test_order_sorts_filters_and_puts_nulls_last() {
        let t = units();
        assert_eq!(order(&t, "index", "rate", false, None).unwrap(), vec![3, 0, 1, 2]);
        assert_eq!(order(&t, "index", "rate", true, None).unwrap(), vec![1, 0, 3, 2]);
        let good = Only { column: "label", value: "good" };
        assert_eq!(order(&t, "index", "rate", true, Some(good)).unwrap(), vec![0, 2]);
        assert!(order(&t, "index", "missing", true, None).is_err());
    }

    #[test]
    fn test_parquet_round_trip() {
        let dir = std::env::temp_dir().join(format!("dspk-table-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("units.parquet");
        let mut t = units();
        assert!(write_parquet(&path, &mut t).unwrap() > 0);
        assert!(read_parquet(&path).unwrap().equals_missing(&t));
        assert!(matches!(read_parquet(&dir.join("absent.parquet")), Err(TableError::Io(..))));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
