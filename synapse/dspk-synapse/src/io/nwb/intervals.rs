//! NWB `/intervals` tables in a `.nwb.zarr` store (hdmf-zarr layout): `epochs`, `trials` and any
//! other `TimeIntervals` table, each with `start_time` and `stop_time` columns in seconds of the
//! session. Text columns (`tags`, ragged) are not read: an interval is named after its table and
//! row.

use std::path::Path;

use dspk_core::DspResult;
use dspk_io::container::zarr::{has_array, read_array};

/// One row of an intervals table.
#[derive(Debug, Clone, PartialEq)]
pub struct NwbInterval {
    /// The table it comes from (`epochs`, `trials`, …).
    pub table: String,
    /// Its row in the table.
    pub row: usize,
    /// Seconds of the session.
    pub start_sec: f64,
    pub stop_sec: f64,
}

/// Every interval of the store's `/intervals` tables (tables in name order, rows in order);
/// none when it has no `/intervals`. Rows whose stop is not after their start are left out.
pub fn read_intervals(nwb: &Path) -> DspResult<Vec<NwbInterval>> {
    let dir = nwb.join("intervals");
    let Ok(entries) = std::fs::read_dir(&dir) else { return Ok(Vec::new()) };
    let mut tables: Vec<String> = entries.filter_map(Result::ok).filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    tables.sort();
    let mut out = Vec::new();
    for table in tables {
        let (start, stop) = (format!("/intervals/{table}/start_time"), format!("/intervals/{table}/stop_time"));
        if !has_array(nwb, &start) || !has_array(nwb, &stop) {
            continue;
        }
        let (a, b) = (read_array::<f64>(nwb, &start)?.data, read_array::<f64>(nwb, &stop)?.data);
        out.extend(a.iter().zip(&b).enumerate().filter(|(_, (s, e))| e > s).map(|(row, (&start_sec, &stop_sec))| NwbInterval { table: table.clone(), row, start_sec, stop_sec }));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_io::container::zarr::{open_rw_store, write_array_f64, write_group};
    use serde_json::{json, Map};

    #[test]
    fn epochs_and_trials_tables_are_read() {
        let dir = std::env::temp_dir().join(format!("dspk_nwb_intervals_{}.nwb.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = open_rw_store(&dir).unwrap();
        let mut root = Map::new();
        root.insert("neurodata_type".into(), json!("NWBFile"));
        write_group(&store, &dir, "/", root).unwrap();
        write_group(&store, &dir, "/intervals", Map::new()).unwrap();
        for (table, start, stop) in [("trials", vec![1.0, 3.0], vec![2.0, 3.0]), ("epochs", vec![0.0], vec![10.0])] {
            write_group(&store, &dir, &format!("/intervals/{table}"), Map::new()).unwrap();
            let n = [start.len()];
            write_array_f64(&store, &dir, &format!("/intervals/{table}/start_time"), &start, &n, &["rows"], Map::new()).unwrap();
            write_array_f64(&store, &dir, &format!("/intervals/{table}/stop_time"), &stop, &n, &["rows"], Map::new()).unwrap();
        }
        let got = read_intervals(&dir).unwrap();
        let rows: Vec<_> = got.iter().map(|i| (i.table.as_str(), i.row, i.start_sec, i.stop_sec)).collect();
        // The second trial ends where it starts: left out
        assert_eq!(rows, vec![("epochs", 0, 0.0, 10.0), ("trials", 0, 1.0, 2.0)]);
        assert!(read_intervals(&dir.join("missing")).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
