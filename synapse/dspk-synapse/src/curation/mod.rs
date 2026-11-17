//! Curation of a sorting, Phy style: units labelled good / mua / noise (or back to unsorted) and
//! merged. Every edit is an entry of a log, so undo / redo are moves along it and nothing of the
//! sorter's output is rewritten: the log is saved inside the `.sorting.zarr` folder as
//! `dspk-curation.json` (readers that do not know it see the sorter's result), and applied to the
//! loaded sorting on read.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult};
use serde::{Deserialize, Serialize};

pub mod amplitudes;
pub mod charts;
pub mod phy;
pub mod service;
pub mod similarity;
pub mod sorting;
pub mod split;
/// The unit table (a Polars frame with feature `table`, on by default; plain rows without it).
pub mod table;
pub mod unit_stats;
pub mod views;
pub mod waveforms;

pub use similarity::{similar, SIMILARITY_LAG};
pub use sorting::SortingData;
pub use unit_stats::UnitPoint;
pub use waveforms::{Snippets, Windows};

/// The log's file inside the store.
pub const CURATION_FILE: &str = "dspk-curation.json";
/// The log's name before v0.0.11. Phy and SpikeInterface keep merge proposals in a file of that
/// name, in another format, so the log moved: one found there is still read when it is ours (it
/// has the `dspk_curation` key) and renamed at the next save; anyone else's is left alone.
pub const LEGACY_CURATION_FILE: &str = "curation.json";
/// Format version of the log.
pub const CURATION_VERSION: u32 = 1;
/// Phy's labels, in menu order.
pub const LABELS: [&str; 4] = ["good", "mua", "noise", "unsorted"];

/// One curation edit; units by their unit id (stable across edits, unlike row indices).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum CurationOp {
    Label { units: Vec<usize>, label: String },
    /// `units` become one unit, id `into` (labelled unsorted, as Phy does).
    Merge { units: Vec<usize>, into: usize },
    /// The spikes of `unit` at sample times `spikes` become a new unit, id `into`.
    Split { unit: usize, spikes: Vec<u64>, into: usize },
    /// The merged unit `unit` is taken apart: the units merged into it come back, each with the
    /// spikes it brought (and its label then); `unit` is gone. Its spikes split off since stay
    /// where they went.
    Unmerge { unit: usize },
}

/// The units merged into `unit` by `ops` (`None`: `unit` is not a merge, or was taken apart).
pub fn merged_parts(ops: &[CurationOp], unit: usize) -> Option<Vec<usize>> {
    let mut parts = None;
    for op in ops {
        match op {
            CurationOp::Merge { units, into } if *into == unit => parts = Some(units.clone()),
            CurationOp::Unmerge { unit: u } if *u == unit => parts = None,
            _ => {}
        }
    }
    parts
}

/// When, where and by whom an edit was made: each optional (a log written before they were kept
/// has none, and reads as before).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditMeta {
    /// UTC, RFC 3339 (`2026-11-17T10:52:03Z`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    /// Where: `desktop`, `page` (the Pages viewer), `server` (for a page it serves).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
    /// Who, when known (the server's name for the client that sent it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
}

impl EditMeta {
    /// An edit made `unix_secs` after 1970 (UTC), `via` there.
    pub fn at(unix_secs: u64, via: &str) -> Self {
        Self { at: Some(rfc3339_utc(unix_secs)), via: Some(via.into()), by: None }
    }

    /// An edit made now, `via` there.
    pub fn now(via: &str) -> Self {
        Self::at(unix_now(), via)
    }
}

pub use dspk_core::time::wall::{rfc3339_utc, unix_now};

/// One edit of the log as written: the edit and its [`EditMeta`], side by side
/// (`{"op": "merge", "units": [5, 6], "into": 8, "at": "…", "via": "desktop"}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggedEdit {
    #[serde(flatten)]
    pub op: CurationOp,
    #[serde(flatten)]
    pub meta: EditMeta,
}

/// The log as written (`dspk-curation.json`). A log written before the edits' [`EditMeta`], the
/// save time and the undone edits were kept reads the same (they are optional).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CurationFile {
    pub dspk_curation: u32,
    /// When it was written (UTC, RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_at: Option<String>,
    /// What wrote it (`dspk-synapse 0.0.12`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written_by: Option<String>,
    /// The edits applied, in order.
    pub ops: Vec<LoggedEdit>,
    /// Edits undone and not replaced (redo brings them back, also after a restart), last first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub undone: Vec<LoggedEdit>,
}

/// A sorting's log: the edits applied (and when, where, by whom: `meta`, one per edit), the ones
/// undone (redo), and how many were saved.
#[derive(Debug, Clone, Default)]
pub struct Curation {
    pub ops: Vec<CurationOp>,
    pub meta: Vec<EditMeta>,
    pub undone: Vec<(CurationOp, EditMeta)>,
    pub saved: usize,
}


impl Curation {
    /// The log saved in `store` (none: an empty one).
    pub fn load(store: &Path) -> DspResult<Self> {
        let path = store.join(CURATION_FILE);
        let text = if path.is_file() {
            std::fs::read_to_string(&path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?
        } else {
            match legacy_log(store) {
                Some(text) => text,
                None => return Ok(Self::default()),
            }
        };
        let file: CurationFile = serde_json::from_str(&text).map_err(|e| DspError::UnsupportedFormat(format!("{}: {e}", path.display())))?;
        if file.dspk_curation != CURATION_VERSION {
            return Err(DspError::UnsupportedFormat(format!("{}: curation version {} (this version reads {CURATION_VERSION})", path.display(), file.dspk_curation)));
        }
        Ok(Self::from_file(file))
    }

    /// The log a file holds, saved as it is.
    pub fn from_file(file: CurationFile) -> Self {
        let (ops, meta): (Vec<_>, Vec<_>) = file.ops.into_iter().map(|e| (e.op, e.meta)).unzip();
        let undone = file.undone.into_iter().rev().map(|e| (e.op, e.meta)).collect();
        Self { saved: ops.len(), ops, meta, undone }
    }

    /// The log as written: each edit with its time, place and author, stamped `saved_at` now.
    pub fn file(&self) -> CurationFile {
        let logged = |(op, meta): (&CurationOp, &EditMeta)| LoggedEdit { op: op.clone(), meta: meta.clone() };
        CurationFile {
            dspk_curation: CURATION_VERSION,
            saved_at: Some(rfc3339_utc(unix_now())),
            written_by: Some(concat!("dspk-synapse ", env!("CARGO_PKG_VERSION")).into()),
            ops: self.ops.iter().zip(self.metas()).map(logged).collect(),
            undone: self.undone.iter().rev().map(|(op, meta)| logged((op, meta))).collect(),
        }
    }

    /// One [`EditMeta`] per edit (an edit pushed without one: empty).
    fn metas(&self) -> impl Iterator<Item = &EditMeta> {
        static NONE: EditMeta = EditMeta { at: None, via: None, by: None };
        (0..self.ops.len()).map(|i| self.meta.get(i).unwrap_or(&NONE))
    }

    /// Writes the log into `store` (through a temporary file). A log under the earlier name
    /// ([`LEGACY_CURATION_FILE`]) is removed once the new file is in place, so a store holds one.
    pub fn save(&mut self, store: &Path) -> DspResult<PathBuf> {
        let path = store.join(CURATION_FILE);
        let file = self.file();
        let tmp = store.join(format!("{CURATION_FILE}.tmp"));
        let io = |p: &Path, e: std::io::Error| DspError::Io(format!("{}: {e}", p.display()));
        let text = serde_json::to_string_pretty(&file).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        let ours_before = legacy_log(store).is_some();
        std::fs::write(&tmp, text).map_err(|e| io(&tmp, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| io(&path, e))?;
        if ours_before {
            let old = store.join(LEGACY_CURATION_FILE);
            std::fs::remove_file(&old).map_err(|e| io(&old, e))?;
        }
        self.saved = self.ops.len();
        Ok(path)
    }

    pub fn dirty(&self) -> bool {
        self.saved != self.ops.len()
    }

    /// Adds `op`, made now (no place).
    pub fn push(&mut self, op: CurationOp) {
        self.push_with(op, EditMeta { at: Some(rfc3339_utc(unix_now())), ..Default::default() });
    }

    /// Adds `op`, made when, where and by whom `meta` says.
    pub fn push_with(&mut self, op: CurationOp, meta: EditMeta) {
        // Undone past the saved state and edited: that state can no longer be reached
        if self.saved > self.ops.len() {
            self.saved = usize::MAX;
        }
        self.meta.resize(self.ops.len(), EditMeta::default());
        self.ops.push(op);
        self.meta.push(meta);
        self.undone.clear();
    }

    pub fn undo(&mut self) -> bool {
        self.meta.resize(self.ops.len(), EditMeta::default());
        match self.ops.pop() {
            Some(op) => {
                let meta = self.meta.pop().unwrap_or_default();
                self.undone.push((op, meta));
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.undone.pop() {
            Some((op, meta)) => {
                self.meta.resize(self.ops.len(), EditMeta::default());
                self.ops.push(op);
                self.meta.push(meta);
                true
            }
            None => false,
        }
    }
}

/// The text of `store`'s [`LEGACY_CURATION_FILE`] when that file is a dspk log (a JSON object with
/// the `dspk_curation` key); `None` when it is absent, unreadable or someone else's.
fn legacy_log(store: &Path) -> Option<String> {
    let text = std::fs::read_to_string(store.join(LEGACY_CURATION_FILE)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value.get("dspk_curation").is_some().then_some(text)
}

/// A unit id no unit of `sorting` has.
pub fn next_unit_id(sorting: &SortingData) -> usize {
    sorting.units.iter().map(|u| u.info.unit_id + 1).max().unwrap_or(0)
}

/// `base` with `ops` applied: labels set; merged units emptied (kept as rows, so unit indices and
/// colours stay put) and their spikes given to the new unit appended at the end.
pub fn apply(base: &SortingData, ops: &[CurationOp]) -> SortingData {
    let mut s = base.clone();
    // Per merged unit (its index): the units merged into it (indices); each spike's own unit is
    // what it was before the merge, kept to take it apart
    let mut merges: std::collections::HashMap<usize, (Vec<usize>, Vec<(usize, u32)>)> = Default::default();
    for op in ops {
        match op {
            CurationOp::Label { units, label } => {
                for id in units {
                    if let Some(&u) = s.by_id.get(&(*id as i32)) {
                        if s.units[u].merged_into.is_none() {
                            s.units[u].info.quality_label = label.clone();
                        }
                    }
                }
            }
            CurationOp::Merge { units, into } => {
                let idx: Vec<usize> = units.iter().filter_map(|id| s.by_id.get(&(*id as i32)).copied()).filter(|&u| s.units[u].merged_into.is_none()).collect();
                if idx.len() < 2 || s.by_id.contains_key(&(*into as i32)) {
                    continue;
                }
                let new = s.units.len();
                // The largest unit gives the merged one its channel, extent and template
                let lead = *idx.iter().max_by_key(|&&u| s.units[u].info.num_spikes).expect("two units");
                let mut unit = s.units[lead].clone();
                unit.info.unit_id = *into;
                unit.info.quality_label = "unsorted".into();
                unit.info.num_spikes = idx.iter().map(|&u| s.units[u].info.num_spikes).sum();
                unit.merged_into = None;
                let set: HashSet<u32> = idx.iter().map(|&u| u as u32).collect();
                let mut before = Vec::new();
                for (k, slot) in s.unit_of.iter_mut().enumerate() {
                    if let Some(u) = slot.filter(|u| set.contains(u)) {
                        before.push((k, u));
                        *slot = Some(new as u32);
                    }
                }
                merges.insert(new, (idx.clone(), before));
                for &u in &idx {
                    s.units[u].merged_into = Some(new);
                    s.units[u].info.num_spikes = 0;
                }
                s.units.push(unit);
                s.by_id.insert(*into as i32, new);
            }
            CurationOp::Split { unit, spikes, into } => {
                let Some(&u) = s.by_id.get(&(*unit as i32)) else { continue };
                if s.units[u].merged_into.is_some() || s.by_id.contains_key(&(*into as i32)) || spikes.is_empty() {
                    continue;
                }
                let take: HashSet<u64> = spikes.iter().copied().collect();
                let new = s.units.len();
                let mut moved = 0;
                for k in 0..s.times.len() {
                    if s.unit_of[k] == Some(u as u32) && take.contains(&s.times[k]) {
                        s.unit_of[k] = Some(new as u32);
                        moved += 1;
                    }
                }
                let mut part = s.units[u].clone();
                part.info.unit_id = *into;
                part.info.quality_label = "unsorted".into();
                part.info.num_spikes = moved;
                s.units[u].info.num_spikes -= moved.min(s.units[u].info.num_spikes);
                s.units.push(part);
                s.by_id.insert(*into as i32, new);
            }
            CurationOp::Unmerge { unit } => {
                let Some(&m) = s.by_id.get(&(*unit as i32)) else { continue };
                let Some((parts, before)) = merges.remove(&m) else { continue };
                if s.units[m].merged_into.is_some() {
                    continue;
                }
                // The spikes still in the merged unit go back to the unit each came from
                for (k, u) in before {
                    if s.unit_of[k] == Some(m as u32) {
                        s.unit_of[k] = Some(u);
                    }
                }
                for &u in &parts {
                    s.units[u].merged_into = None;
                    s.units[u].info.num_spikes = s.unit_of.iter().filter(|&&x| x == Some(u as u32)).count();
                }
                // The merged unit is gone (its row stays, empty, as a merged unit's does)
                s.units[m].merged_into = Some(m);
                s.units[m].info.num_spikes = 0;
                s.by_id.remove(&(*unit as i32));
            }
        }
    }
    s.index_units();
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::sorting::testing::sorting_of_units;

    #[test]
    fn test_label_merge_undo() {
        let base = sorting_of_units(&[(5, 0, 3), (6, 0, 2), (7, 1, 4)]);
        let mut log = Curation::default();
        log.push(CurationOp::Label { units: vec![7], label: "noise".into() });
        log.push(CurationOp::Merge { units: vec![5, 6], into: next_unit_id(&base) });
        let s = apply(&base, &log.ops);
        assert_eq!(s.units[2].info.quality_label, "noise");
        assert_eq!(s.units.len(), 4);
        let merged = &s.units[3];
        assert_eq!((merged.info.unit_id, merged.info.num_spikes), (8, 5));
        assert_eq!(s.unit_of.iter().filter(|u| **u == Some(3)).count(), 5, "every spike of 5 and 6 moved");
        assert!(s.units[0].merged_into == Some(3) && s.units[0].info.num_spikes == 0);
        assert!(log.undo());
        assert_eq!(apply(&base, &log.ops).units.len(), 3);
        assert!(log.redo() && log.dirty());
        // Split: two of unit 7's spikes (times 50 and 60) become unit 9
        let split = apply(&base, &[CurationOp::Split { unit: 7, spikes: vec![50, 60], into: 9 }]);
        assert_eq!((split.units[2].info.num_spikes, split.units[3].info.num_spikes), (2, 2));
        assert_eq!(split.unit_times(3), vec![50, 60]);
    }

    #[test]
    fn test_saved_log_round_trips() {
        let dir = std::env::temp_dir().join(format!("dspk_curation_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut log = Curation::default();
        log.push(CurationOp::Label { units: vec![1, 2], label: "good".into() });
        log.save(&dir).unwrap();
        assert!(!log.dirty());
        let back = Curation::load(&dir).unwrap();
        assert_eq!(back.ops, log.ops);
        assert!(dir.join("dspk-curation.json").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A store of a test, empty.
    fn store(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dspk_curation_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_a_log_under_the_earlier_name_is_read_and_renamed_at_the_next_save() {
        let dir = store("legacy");
        let ops = vec![CurationOp::Label { units: vec![3], label: "mua".into() }];
        let old = serde_json::json!({"dspk_curation": CURATION_VERSION, "ops": ops}).to_string();
        std::fs::write(dir.join(LEGACY_CURATION_FILE), old).unwrap();
        let mut log = Curation::load(&dir).unwrap();
        assert_eq!(log.ops, ops);
        assert!(dir.join(LEGACY_CURATION_FILE).is_file(), "reading changes nothing");
        log.push(CurationOp::Label { units: vec![4], label: "good".into() });
        log.save(&dir).unwrap();
        assert!(dir.join(CURATION_FILE).is_file() && !dir.join(LEGACY_CURATION_FILE).exists(), "one log, under the new name");
        assert_eq!(Curation::load(&dir).unwrap().ops.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_another_programs_curation_json_is_left_alone() {
        let dir = store("foreign");
        // The shape of a SpikeInterface / phy proposals file: not ours (no `dspk_curation`)
        let theirs = r#"{"format_version": "2", "unit_ids": [1, 2], "merges": [{"unit_ids": [1, 2]}]}"#;
        std::fs::write(dir.join(LEGACY_CURATION_FILE), theirs).unwrap();
        let mut log = Curation::load(&dir).unwrap();
        assert!(log.ops.is_empty(), "nothing of ours to read");
        log.push(CurationOp::Label { units: vec![1], label: "good".into() });
        log.save(&dir).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join(LEGACY_CURATION_FILE)).unwrap(), theirs, "untouched");
        assert_eq!(Curation::load(&dir).unwrap().ops.len(), 1, "ours is read from its own file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_the_new_name_wins_over_a_log_under_the_earlier_one() {
        let dir = store("both");
        let file = |label: &str| serde_json::json!({"dspk_curation": CURATION_VERSION, "ops": [CurationOp::Label { units: vec![1], label: label.into() }]}).to_string();
        std::fs::write(dir.join(LEGACY_CURATION_FILE), file("noise")).unwrap();
        std::fs::write(dir.join(CURATION_FILE), file("good")).unwrap();
        assert_eq!(Curation::load(&dir).unwrap().ops, vec![CurationOp::Label { units: vec![1], label: "good".into() }]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Each edit is written with when, where and by whom it was made, the save's time and what
    /// wrote it, and the edits undone (redo after a restart); a log without them still reads.
    #[test]
    fn test_edits_keep_when_where_and_who_and_the_undone_ones() {
        let dir = store("meta");
        let mut log = Curation::default();
        let by = EditMeta { by: Some("browser-de22bc".into()), ..EditMeta::at(1_763_377_923, "page") };
        log.push_with(CurationOp::Merge { units: vec![5, 6], into: 9 }, by.clone());
        log.push_with(CurationOp::Label { units: vec![9], label: "good".into() }, EditMeta::at(1_763_377_930, "desktop"));
        log.undo();
        log.save(&dir).unwrap();
        let text = std::fs::read_to_string(dir.join(CURATION_FILE)).unwrap();
        assert!(text.contains(r#""at": "2025-11-17T11:12:03Z""#) && text.contains(r#""by": "browser-de22bc""#) && text.contains("saved_at") && text.contains("written_by"), "{text}");
        let mut back = Curation::load(&dir).unwrap();
        assert_eq!((back.ops.len(), back.meta[0].clone()), (1, by));
        assert!(back.redo(), "the undone edit is kept");
        assert_eq!(back.meta[1].via.as_deref(), Some("desktop"));
        assert_eq!(back.ops, vec![CurationOp::Merge { units: vec![5, 6], into: 9 }, CurationOp::Label { units: vec![9], label: "good".into() }]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A merge taken apart gives back its units with the spikes and labels they had; spikes split
    /// off the merged unit since stay in their new unit; it is told apart from other units.
    #[test]
    fn test_an_unmerge_gives_back_the_units_merged() {
        let base = sorting::testing::sorting_of_units(&[(5, 0, 4), (6, 100, 4), (7, 200, 4)]);
        let n_spikes = |s: &SortingData, id: usize| s.by_id.get(&(id as i32)).map(|&u| s.unit_of.iter().filter(|&&x| x == Some(u as u32)).count());
        let into = next_unit_id(&base);
        let mut ops = vec![CurationOp::Label { units: vec![5], label: "good".into() }, CurationOp::Merge { units: vec![5, 6], into }];
        assert_eq!(merged_parts(&ops, into), Some(vec![5, 6]));
        let merged = apply(&base, &ops);
        assert_eq!(n_spikes(&merged, into), Some(8));
        // One spike of the merged unit split off, then the merge taken apart
        let moved = merged.unit_times(merged.by_id[&(into as i32)])[0];
        ops.push(CurationOp::Split { unit: into, spikes: vec![moved], into: into + 1 });
        ops.push(CurationOp::Unmerge { unit: into });
        assert_eq!(merged_parts(&ops, into), None, "taken apart");
        let back = apply(&base, &ops);
        assert_eq!(n_spikes(&back, into), None, "the merged unit is gone");
        assert_eq!(n_spikes(&back, 5).unwrap() + n_spikes(&back, 6).unwrap(), 7);
        assert_eq!(n_spikes(&back, into + 1), Some(1), "the split-off spike stays where it went");
        assert_eq!(back.units[back.by_id[&5]].info.quality_label, "good", "its label as it was");
        assert!(back.units[back.by_id[&5]].merged_into.is_none() && back.units[back.by_id[&6]].merged_into.is_none());
        let json = serde_json::to_string(&CurationOp::Unmerge { unit: 3 }).unwrap();
        assert_eq!(json, r#"{"op":"unmerge","unit":3}"#);
    }
}
