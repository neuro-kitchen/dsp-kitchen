//! Synapse run archive (`.tin`): a zip holding `Summary.txt` (experiment, subject, user,
//! start time, software versions), the experiment file and rig settings.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// `Summary.txt` flattened to `Section.Key` → value (e.g. `Versions.Synapse` → `53575`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TinSummary {
    pub fields: BTreeMap<String, String>,
    /// Every file in the archive.
    pub entries: Vec<String>,
}

impl TinSummary {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }
}

pub fn read_tin(path: &Path) -> Option<TinSummary> {
    let file = std::fs::File::open(path).ok()?;
    let mut zip = zip::ZipArchive::new(file).ok()?;
    let entries = zip.file_names().map(str::to_string).collect();
    let mut text = String::new();
    zip.by_name("Summary.txt").ok()?.read_to_string(&mut text).ok()?;
    Some(TinSummary { fields: parse_summary(&text), entries })
}

/// Indented `Key: value` text: unindented `Section:` lines open a section.
pub fn parse_summary(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut section = String::new();
    for line in text.lines() {
        let indented = line.starts_with(' ') || line.starts_with('\t');
        let Some((k, v)) = line.trim().split_once(':') else { continue };
        let (k, v) = (k.trim(), v.trim());
        if !indented && v.is_empty() {
            section = k.to_string();
        } else if !v.is_empty() && v != "NONE" {
            let key = if section.is_empty() { k.to_string() } else { format!("{section}.{k}") };
            out.insert(key, v.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_sections() {
        let s = parse_summary("Experiment:\n  Name: HD_MEPs_v2\n\nRecording:\n  StartTime: 2025-02-26T15:25:56\nVersions:\n  Qt: 5.15.0\n  GLBase: NONE\n  Synapse: 53575\n");
        assert_eq!(s["Experiment.Name"], "HD_MEPs_v2");
        // The first colon splits, so ISO times keep theirs
        assert_eq!(s["Recording.StartTime"], "2025-02-26T15:25:56");
        assert_eq!(s["Versions.Synapse"], "53575");
        assert!(!s.contains_key("Versions.GLBase"));
    }
}
