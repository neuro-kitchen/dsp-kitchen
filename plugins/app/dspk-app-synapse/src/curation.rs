//! Curation of a sorting, Phy style: units labelled good / mua / noise (or back to unsorted) and
//! merged. Every edit is an entry of a log, so undo / redo are moves along it and nothing of the
//! sorter's output is rewritten: the log is saved inside the `.sorting.zarr` folder as
//! `curation.json` (readers that do not know it see the sorter's result), and applied to the
//! loaded sorting on read.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::sorting::SortingData;

/// The log's file inside the store.
pub const CURATION_FILE: &str = "curation.json";
/// Format version of `curation.json`.
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
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CurationFile {
    pub dspk_curation: u32,
    pub ops: Vec<CurationOp>,
}

/// A sorting's log: the edits applied, the ones undone (redo), and how many were saved.
#[derive(Debug, Clone, Default)]
pub struct Curation {
    pub ops: Vec<CurationOp>,
    pub undone: Vec<CurationOp>,
    pub saved: usize,
}

impl Curation {
    /// The log saved in `store` (none: an empty one).
    pub fn load(store: &Path) -> Result<Self> {
        let path = store.join(CURATION_FILE);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path).with_context(|| format!("Failed to read {}", dspk_app_core::domain::paths::short_path(&path)))?;
        let file: CurationFile = serde_json::from_str(&text).with_context(|| format!("{}", dspk_app_core::domain::paths::short_path(&path)))?;
        anyhow::ensure!(file.dspk_curation == CURATION_VERSION, "{}: curation version {} (this app reads {CURATION_VERSION})", dspk_app_core::domain::paths::short_path(&path), file.dspk_curation);
        Ok(Self { saved: file.ops.len(), ops: file.ops, undone: Vec::new() })
    }

    /// Writes the log into `store` (through a temporary file).
    pub fn save(&mut self, store: &Path) -> Result<PathBuf> {
        let path = store.join(CURATION_FILE);
        let file = CurationFile { dspk_curation: CURATION_VERSION, ops: self.ops.clone() };
        let tmp = store.join(format!("{CURATION_FILE}.tmp"));
        std::fs::write(&tmp, serde_json::to_string_pretty(&file)?).with_context(|| format!("Failed to write {}", dspk_app_core::domain::paths::short_path(&tmp)))?;
        std::fs::rename(&tmp, &path)?;
        self.saved = self.ops.len();
        Ok(path)
    }

    pub fn dirty(&self) -> bool {
        self.saved != self.ops.len()
    }

    pub fn push(&mut self, op: CurationOp) {
        // Undone past the saved state and edited: that state can no longer be reached
        if self.saved > self.ops.len() {
            self.saved = usize::MAX;
        }
        self.ops.push(op);
        self.undone.clear();
    }

    pub fn undo(&mut self) -> bool {
        match self.ops.pop() {
            Some(op) => {
                self.undone.push(op);
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.undone.pop() {
            Some(op) => {
                self.ops.push(op);
                true
            }
            None => false,
        }
    }
}

/// A unit id no unit of `sorting` has.
pub fn next_unit_id(sorting: &SortingData) -> usize {
    sorting.units.iter().map(|u| u.info.unit_id + 1).max().unwrap_or(0)
}

/// `base` with `ops` applied: labels set; merged units emptied (kept as rows, so unit indices and
/// colours stay put) and their spikes given to the new unit appended at the end.
pub fn apply(base: &SortingData, ops: &[CurationOp]) -> SortingData {
    let mut s = base.clone();
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
                for slot in s.unit_of.iter_mut() {
                    if slot.is_some_and(|u| set.contains(&u)) {
                        *slot = Some(new as u32);
                    }
                }
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
        }
    }
    s.index_units();
    s
}

/// Samples of shift tried when comparing templates.
pub const SIMILARITY_LAG: usize = 5;

/// A unit's mean template as dspk-synapse compares templates: every channel its stored template
/// has signal on (else its extent channels).
fn waveform_template(sorting: &SortingData, u: usize) -> Option<dspk_synapse::WaveformTemplate> {
    let unit = &sorting.units[u];
    if let (Some(t), Some(row)) = (&sorting.templates, unit.template_row) {
        let r = t.row(row);
        let channels: Vec<usize> = (0..t.channels).filter(|&c| r[c * t.samples..(c + 1) * t.samples].iter().any(|v| *v != 0.0)).collect();
        if !channels.is_empty() {
            let mean: Vec<f32> = channels.iter().flat_map(|&c| r[c * t.samples..(c + 1) * t.samples].iter().copied()).collect();
            let n = mean.len();
            return Some(dspk_synapse::WaveformTemplate::new(channels, t.samples, mean, vec![0.0; n]));
        }
    }
    let samples = unit.template.first()?.len();
    let mean: Vec<f32> = unit.template.iter().flatten().copied().collect();
    Some(dspk_synapse::WaveformTemplate::new(unit.extent.channels.clone(), samples, mean, vec![0.0; unit.template.len() * samples]))
}

/// Units most like `lead` by template: dspk-synapse's lag-maximised cosine similarity (rows
/// paired by channel, best shift within ±[`SIMILARITY_LAG`] samples), best first, at most `max`.
pub fn similar(sorting: &SortingData, lead: usize, max: usize) -> Vec<(usize, f32)> {
    let Some(a) = waveform_template(sorting, lead) else { return Vec::new() };
    let mut out: Vec<(usize, f32)> = (0..sorting.units.len())
        .filter(|&u| u != lead && sorting.units[u].merged_into.is_none())
        .filter_map(|u| {
            let b = waveform_template(sorting, u)?;
            let (sim, _) = dspk_synapse::sorting::similarity::template_max_cosine_similarity(&a, &b, SIMILARITY_LAG);
            (sim != 0.0).then_some((u, sim))
        })
        .collect();
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    out.truncate(max);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorting::tests::sorting_of_units;

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
        let _ = std::fs::remove_dir_all(&dir);
    }
}
