//! Session + user metadata → an NWB plan: what is written where, and what is missing.

use crate::common::time::{format_iso, parse_iso};
use crate::metadata::{ImpedanceSpec, Issue, Level, MetadataFile, StreamType};
use crate::model::{Device, Session, Table};

/// NWB file-level fields, resolved.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FileFields {
    pub description: String,
    pub identifier: String,
    /// ISO 8601 with a time zone.
    pub start_time: String,
    pub experiment_description: Option<String>,
    pub experimenters: Vec<String>,
    pub lab: Option<String>,
    pub institution: Option<String>,
    pub keywords: Vec<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SubjectFields {
    pub id: Option<String>,
    pub species: Option<String>,
    pub sex: Option<String>,
    pub age: Option<String>,
    pub strain: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GroupPlan {
    pub name: String,
    pub description: String,
    pub location: String,
    pub device: String,
    #[serde(skip)]
    pub impedance: Option<ImpedanceSpec>,
}

/// One continuous signal to write.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SeriesPlan {
    /// Index into `Session::recordings`.
    pub recording: usize,
    pub source: String,
    pub name: String,
    pub description: String,
    /// `Some(group)` for an `ElectricalSeries`.
    pub electrode_group: Option<usize>,
    /// First row of this series in the electrodes table (electrical only).
    pub first_electrode: usize,
    pub unit: String,
    pub conversion: f64,
}

/// One event series to write: intervals (with offsets) or a timestamped `TimeSeries`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EventPlan {
    /// Index into `Session::events`.
    pub event: usize,
    pub source: String,
    pub name: String,
    pub description: String,
    pub intervals: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TablePlan {
    pub table: usize,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct NwbPlan {
    pub file: FileFields,
    pub subject: SubjectFields,
    pub devices: Vec<Device>,
    pub groups: Vec<GroupPlan>,
    pub series: Vec<SeriesPlan>,
    pub events: Vec<EventPlan>,
    pub tables: Vec<TablePlan>,
    /// Impedance in ohms per electrodes-table row (NaN = not measured); empty = no `imp` column.
    pub impedance_ohms: Vec<f32>,
    /// Source items left out (by the metadata file).
    pub skipped: Vec<String>,
    pub issues: Vec<Issue>,
}

impl NwbPlan {
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.level == Level::Error)
    }
}

/// NWB object names cannot contain `/`; TDT store names often end in one (`MET/`).
pub fn safe_name(source: &str) -> String {
    let trimmed = source.trim_end_matches(['/', '\\']);
    let s: String = trimmed.chars().map(|c| if c == '/' || c == '\\' { '_' } else { c }).collect();
    if s.is_empty() { "unnamed".into() } else { s }
}

/// True for `...Z`, `...+hh:mm` or `...-hh:mm` after the time part.
fn has_zone(iso: &str) -> bool {
    let Some((_, time)) = iso.split_once('T') else { return false };
    time.ends_with('Z') || time.contains('+') || time.rfind('-').is_some()
}

fn valid_zone(z: &str) -> bool {
    z == "Z" || (z.len() == 6 && (z.starts_with('+') || z.starts_with('-')) && z.as_bytes()[3] == b':')
}

pub fn resolve(session: &Session, meta: &MetadataFile, new_identifier: impl FnOnce() -> String) -> NwbPlan {
    let mut plan = NwbPlan::default();
    let issues = &mut plan.issues;
    let sm = &session.metadata;
    let ms = &meta.session;

    // File fields
    plan.file.description = ms.description.clone().unwrap_or_default();
    if plan.file.description.trim().is_empty() {
        issues.push(Issue::error("session.description is required (a sentence describing the session)"));
    }
    plan.file.identifier = ms.identifier.clone().unwrap_or_else(new_identifier);
    plan.file.start_time = match (&ms.start_time, &sm.start_time) {
        (Some(t), _) if has_zone(t) && parse_iso(t).is_some() => t.clone(),
        (Some(t), _) => {
            issues.push(Issue::error(format!("session.start_time {t:?} needs a time zone, e.g. 2025-02-26T15:25:56-05:00")));
            t.clone()
        }
        (None, Some(rec)) if has_zone(rec) => rec.clone(),
        (None, Some(rec)) => match &ms.timezone {
            Some(z) if valid_zone(z) => format!("{}{z}", parse_iso(rec).map_or_else(|| rec.clone(), format_iso)),
            Some(z) => {
                issues.push(Issue::error(format!("session.timezone {z:?} must look like -05:00, +01:00 or Z")));
                rec.clone()
            }
            None => {
                issues.push(Issue::error(format!(
                    "the recorded start time {rec} has no time zone: set session.timezone (e.g. -05:00) or session.start_time"
                )));
                rec.clone()
            }
        },
        (None, None) => {
            issues.push(Issue::error("no start time recorded: set session.start_time"));
            String::new()
        }
    };
    plan.file.experiment_description = ms.experiment_description.clone().or_else(|| sm.experiment.clone());
    plan.file.experimenters = if ms.experimenters.is_empty() { sm.experimenters.clone() } else { ms.experimenters.clone() };
    plan.file.lab = ms.lab.clone().or_else(|| sm.lab.clone());
    plan.file.institution = ms.institution.clone().or_else(|| sm.institution.clone());
    plan.file.keywords = ms.keywords.clone();
    plan.file.notes = (!sm.notes.is_empty()).then(|| sm.notes.join("\n"));

    // Subject
    let s = &meta.subject;
    plan.subject = SubjectFields {
        id: s.id.clone().or_else(|| sm.subject.id.clone()),
        species: s.species.clone().or_else(|| sm.subject.species.clone()),
        sex: s.sex.clone().or_else(|| sm.subject.sex.clone()).or(Some("U".into())),
        age: s.age.clone().or_else(|| sm.subject.age.clone()),
        strain: s.strain.clone(),
        description: s.description.clone().or_else(|| sm.subject.description.clone()),
    };
    if plan.subject.species.is_none() {
        issues.push(Issue::warning("subject.species is missing (required by DANDI)"));
    }
    if plan.subject.age.is_none() {
        issues.push(Issue::warning("subject.age is missing (required by DANDI, ISO 8601 e.g. P90D)"));
    }
    if let Some(sex) = &plan.subject.sex {
        if !["M", "F", "U", "O"].contains(&sex.as_str()) {
            issues.push(Issue::warning(format!("subject.sex {sex:?} should be M, F, U or O")));
        }
    }

    // Devices and electrode groups
    plan.devices = sm.devices.clone();
    for g in &meta.electrode_groups {
        let device = match &g.device {
            Some(d) => {
                if !plan.devices.iter().any(|x| &x.name == d) {
                    plan.devices.push(Device { name: d.clone(), description: "declared in the metadata file".into(), manufacturer: None });
                }
                d.clone()
            }
            None => match plan.devices.first() {
                Some(d) => d.name.clone(),
                None => {
                    plan.devices.push(Device { name: "acquisition_system".into(), description: "recording hardware".into(), manufacturer: None });
                    "acquisition_system".into()
                }
            },
        };
        plan.groups.push(GroupPlan {
            name: g.name.clone(),
            description: g.description.clone(),
            location: g.location.clone(),
            device,
            impedance: g.impedance.clone(),
        });
    }

    // Streams
    let mut electrode_row = 0;
    for (i, rec) in session.recordings.iter().enumerate() {
        let info = rec.info();
        let spec = meta.stream(&info.name);
        if spec.include == Some(false) {
            plan.skipped.push(format!("stream {}", info.name));
            continue;
        }
        let kind = spec.kind.unwrap_or(StreamType::Timeseries);
        let electrode_group = match (kind, &spec.electrode_group) {
            (StreamType::Electrical, Some(g)) => match plan.groups.iter().position(|x| &x.name == g) {
                Some(gi) => Some(gi),
                None => {
                    issues.push(Issue::error(format!("stream {}: electrode_group {g:?} is not declared under electrode_groups", info.name)));
                    None
                }
            },
            (StreamType::Electrical, None) => {
                issues.push(Issue::error(format!("stream {}: electrical streams need an electrode_group", info.name)));
                None
            }
            (StreamType::Timeseries, _) => None,
        };
        let electrical = kind == StreamType::Electrical;
        if electrical && spec.unit.as_deref().is_some_and(|u| u != "volts" && u != "V" && u != "a.u.") {
            issues.push(Issue::warning(format!("stream {}: electrical series are stored in volts; use conversion to scale", info.name)));
        }
        let first_electrode = electrode_row;
        if electrical {
            electrode_row += info.channel_count();
        }
        plan.series.push(SeriesPlan {
            recording: i,
            source: info.name.clone(),
            name: spec.name.unwrap_or_else(|| safe_name(&info.name)),
            description: spec.description.unwrap_or_else(|| {
                if info.description.is_empty() {
                    format!("{} from the {} recording ({} channels)", info.name, session.provenance.format, info.channel_count())
                } else {
                    info.description.clone()
                }
            }),
            electrode_group,
            first_electrode,
            unit: if electrical { "volts".into() } else { spec.unit.unwrap_or_else(|| "a.u.".into()) },
            conversion: spec.conversion.unwrap_or(1.0),
        });
    }

    // Events and tables
    for (i, e) in session.events.iter().enumerate() {
        let spec = meta.event(&e.name);
        if spec.include == Some(false) {
            plan.skipped.push(format!("events {}", e.name));
            continue;
        }
        plan.events.push(EventPlan {
            event: i,
            source: e.name.clone(),
            name: spec.name.unwrap_or_else(|| safe_name(&e.name)),
            description: spec.description.unwrap_or_else(|| {
                if e.description.is_empty() { format!("{} events from the {} recording", e.name, session.provenance.format) } else { e.description.clone() }
            }),
            intervals: e.offsets.is_some() && e.channels == 1,
        });
    }
    for (i, t) in session.tables.iter().enumerate() {
        let spec = meta.table(&t.name);
        if spec.include == Some(false) {
            plan.skipped.push(format!("table {}", t.name));
            continue;
        }
        plan.tables.push(TablePlan { table: i, name: spec.name.unwrap_or_else(|| safe_name(&t.name)), description: spec.description.unwrap_or_else(|| t.description.clone()) });
    }

    // Electrode impedances, one per electrodes-table row
    if plan.groups.iter().any(|g| g.impedance.is_some()) {
        for s in &plan.series {
            let Some(gi) = s.electrode_group else { continue };
            let channels = session.recordings[s.recording].info().channel_count();
            let values = match &plan.groups[gi].impedance {
                None => vec![f32::NAN; channels],
                Some(spec) => match session.tables.iter().find(|t| t.name == spec.table) {
                    Some(t) => impedances(t, spec, channels),
                    None => {
                        plan.issues.push(Issue::warning(format!("electrode group {}: impedance table {:?} not found", plan.groups[gi].name, spec.table)));
                        vec![f32::NAN; channels]
                    }
                },
            };
            plan.impedance_ohms.extend(values);
        }
    }

    // Names must be unique within their NWB group
    let mut seen = std::collections::BTreeSet::new();
    for n in plan.series.iter().map(|s| &s.name).chain(plan.events.iter().filter(|e| !e.intervals).map(|e| &e.name)) {
        if !seen.insert(n.clone()) {
            plan.issues.push(Issue::error(format!("two acquisition items are named {n:?}: rename one in the metadata file")));
        }
    }
    for key in meta.streams.keys().filter(|k| *k != "*") {
        if session.recording(key).is_none() {
            plan.issues.push(Issue::warning(format!("metadata streams.{key}: no such stream in the recording")));
        }
    }
    plan
}

/// Impedance (ohms) of channels `1..=channels` from `table` columns `<prefix><n> (<unit>)`.
fn impedances(table: &Table, spec: &ImpedanceSpec, channels: usize) -> Vec<f32> {
    let prefix = spec.prefix.as_deref().unwrap_or("R");
    (1..=channels)
        .map(|n| {
            let head = format!("{prefix}{n} (");
            let Some((ci, col)) = table.columns.iter().enumerate().find(|(_, c)| c.starts_with(&head)) else { return f32::NAN };
            let scale = if col.ends_with("(kOhm)") { 1e3 } else if col.ends_with("(MOhm)") { 1e6 } else { 1.0 };
            let value = |r: &Vec<String>| r.get(ci).and_then(|v| v.parse::<f64>().ok()).filter(|v| *v >= 0.0);
            let v = match spec.row {
                Some(row) => table.rows.get(row).and_then(value),
                None => table.rows.iter().rev().find_map(value),
            };
            v.map_or(f32::NAN, |v| (v * scale) as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impedances_last_measured_row() {
        let t = Table {
            name: "Z".into(),
            description: String::new(),
            columns: vec!["TIME (S)".into(), "R1 (kOhm)".into(), "R2 (kOhm)".into()],
            rows: vec![vec!["60".into(), "1.5".into(), "0.4".into()], vec!["64".into(), "-1.00".into(), "0.5".into()]],
        };
        let spec = ImpedanceSpec { table: "Z".into(), ..Default::default() };
        let v = impedances(&t, &spec, 3);
        assert_eq!(&v[..2], &[1500.0, 500.0]);
        assert!(v[2].is_nan());
        let first = impedances(&t, &ImpedanceSpec { row: Some(0), ..spec }, 2);
        assert_eq!(first, vec![1500.0, 400.0]);
    }

    #[test]
    fn test_names_and_zones() {
        assert_eq!(safe_name("MET/"), "MET");
        assert_eq!(safe_name("CnA\\"), "CnA");
        assert_eq!(safe_name("a/b"), "a_b");
        assert!(has_zone("2025-02-26T15:25:56-05:00"));
        assert!(has_zone("2025-02-26T15:25:56Z"));
        assert!(!has_zone("2025-02-26T15:25:56"));
        assert!(valid_zone("-05:00") && valid_zone("Z") && !valid_zone("EST"));
    }
}
