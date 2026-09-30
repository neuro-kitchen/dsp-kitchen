//! Session tables (e.g. impedance checks) as `DynamicTable`s in `/analysis`. Columns whose
//! cells all parse as numbers are written as float64, others as text.

use serde_json::json;

use super::{column, typed};
use crate::error::Result;
use crate::model::Table;
use crate::outputs::nwb::backend::Backend;
use crate::outputs::nwb::mapping::{safe_name, TablePlan};

pub fn write(b: &dyn Backend, plan: &TablePlan, t: &Table) -> Result<()> {
    let path = format!("/analysis/{}", plan.name);
    let names: Vec<String> = t.columns.iter().map(|c| safe_name(c)).collect();
    let mut a = typed("hdmf-common", "DynamicTable");
    a.insert("description".into(), json!(plan.description));
    a.insert("colnames".into(), json!(names));
    b.group(&path, a)?;

    let n = t.rows.len() as u64;
    for (ci, name) in names.iter().enumerate() {
        let cells: Vec<String> = t.rows.iter().map(|r| r.get(ci).cloned().unwrap_or_default()).collect();
        let numbers: Option<Vec<f64>> = cells.iter().map(|c| c.parse::<f64>().ok()).collect();
        let desc = column(&format!("Column {:?} of the source table", t.columns[ci]));
        match numbers {
            Some(v) => b.f64s(&format!("{path}/{name}"), &v, &[n], &["dim0"], desc)?,
            None => b.strings(&format!("{path}/{name}"), &cells, "dim0", desc)?,
        }
    }
    let ids: Vec<i64> = (0..n as i64).collect();
    b.i64s(&format!("{path}/id"), &ids, "num_rows", typed("hdmf-common", "ElementIdentifiers"))
}
