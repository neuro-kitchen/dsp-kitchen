//! CSV tables saved in the block folder (Synapse impedance checks: `Z_<gizmo>.csv`,
//! `<device>p<n>.csv` with `TIME (S), FREQUENCY (Hz), R1 (kOhm), …`; `-1.00` = not measured).

use std::path::Path;

use crate::common::text::read_text;
use crate::model::Table;

pub fn read_csv_table(path: &Path) -> Option<Table> {
    let text = read_text(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let columns: Vec<String> = lines.next()?.split(',').map(|c| c.trim().to_string()).collect();
    let rows = lines.map(|l| l.split(',').map(|c| c.trim().to_string()).collect()).collect();
    let name = path.file_stem()?.to_string_lossy().into_owned();
    let description = if columns.iter().any(|c| c.contains("(kOhm)")) {
        "Electrode impedance measurements in kOhm (-1 = not measured), exported by Synapse".to_string()
    } else {
        "CSV table saved with the block".to_string()
    };
    Some(Table { name, description, columns, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impedance_csv() {
        let dir = std::env::temp_dir().join(format!("nc_csv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("Z_EMG.csv");
        std::fs::write(&p, "TIME (S),FREQUENCY (Hz),R1 (kOhm),REF (kOhm)\n59,1120,0.96,-1.00\n").unwrap();
        let t = read_csv_table(&p).unwrap();
        assert_eq!(t.name, "Z_EMG");
        assert_eq!(t.column("R1 (kOhm)"), Some(2));
        assert_eq!(t.rows[0][2], "0.96");
        assert!(t.description.starts_with("Electrode impedance"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
