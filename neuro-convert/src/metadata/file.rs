//! The user's metadata file (YAML): what the source files do not say, and how each source item
//! maps to the output. Keys under `streams` / `events` / `tables` are source names (e.g. TDT
//! store names); `"*"` sets the default for every item not listed.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetadataFile {
    pub session: SessionSection,
    pub subject: SubjectSection,
    pub electrode_groups: Vec<ElectrodeGroupSpec>,
    pub streams: BTreeMap<String, StreamSpec>,
    pub events: BTreeMap<String, ItemSpec>,
    pub tables: BTreeMap<String, ItemSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionSection {
    pub description: Option<String>,
    /// Unique id; a UUID is generated when absent.
    pub identifier: Option<String>,
    /// ISO 8601 with a time zone, e.g. `2025-02-26T15:25:56-05:00` (overrides the recorded time).
    pub start_time: Option<String>,
    /// Zone of the recorded local start time, e.g. `-05:00` (used when `start_time` is absent).
    pub timezone: Option<String>,
    pub experiment_description: Option<String>,
    pub experimenters: Vec<String>,
    pub lab: Option<String>,
    pub institution: Option<String>,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubjectSection {
    pub id: Option<String>,
    /// Latin binomial, e.g. `Rattus norvegicus`.
    pub species: Option<String>,
    /// `M`, `F`, `U` (unknown) or `O` (other).
    pub sex: Option<String>,
    /// ISO 8601 duration, e.g. `P90D`.
    pub age: Option<String>,
    pub strain: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElectrodeGroupSpec {
    pub name: String,
    pub description: String,
    /// Anatomical location (e.g. `diaphragm`).
    pub location: String,
    /// Device name; defaults to the first device of the session.
    pub device: Option<String>,
    /// Impedance table for this group's channels (fills the electrodes `imp` column).
    pub impedance: Option<ImpedanceSpec>,
}

/// Where a group's impedances are: a session table with one column per channel, named
/// `<prefix><n> (kOhm)` (or `(Ohm)` / `(MOhm)`), `n` counting channels from 1.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImpedanceSpec {
    pub table: String,
    /// Column prefix before the channel number (default `R`).
    pub prefix: Option<String>,
    /// Row to use; default: the last row with a measurement (>= 0) for each channel.
    pub row: Option<usize>,
}

/// How a continuous signal is exported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamType {
    /// Voltage from electrodes (NWB `ElectricalSeries`, needs an electrode group).
    Electrical,
    /// Any other signal (NWB `TimeSeries`).
    Timeseries,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreamSpec {
    pub include: Option<bool>,
    /// Output name (defaults to the source name, made safe).
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<StreamType>,
    pub electrode_group: Option<String>,
    pub description: Option<String>,
    /// Unit of `TimeSeries` data (electrical series are always volts).
    pub unit: Option<String>,
    /// Multiplier from stored values to `unit` (volts for electrical series).
    pub conversion: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ItemSpec {
    pub include: Option<bool>,
    pub name: Option<String>,
    pub description: Option<String>,
}

impl MetadataFile {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        Self::parse(&text).map_err(|e| Error::format("metadata", format!("{}: {e}", path.display())))
    }

    pub fn parse(text: &str) -> std::result::Result<Self, serde_yaml_ng::Error> {
        serde_yaml_ng::from_str(text)
    }

    /// The spec for source stream `name`: its own entry over the `"*"` default.
    pub fn stream(&self, name: &str) -> StreamSpec {
        let own = self.streams.get(name).cloned().unwrap_or_default();
        let def = self.streams.get("*").cloned().unwrap_or_default();
        StreamSpec {
            include: own.include.or(def.include),
            name: own.name,
            kind: own.kind.or(def.kind),
            electrode_group: own.electrode_group.or(def.electrode_group),
            description: own.description,
            unit: own.unit.or(def.unit),
            conversion: own.conversion.or(def.conversion),
        }
    }

    pub fn event(&self, name: &str) -> ItemSpec {
        item(&self.events, name)
    }

    pub fn table(&self, name: &str) -> ItemSpec {
        item(&self.tables, name)
    }
}

fn item(map: &BTreeMap<String, ItemSpec>, name: &str) -> ItemSpec {
    let own = map.get(name).cloned().unwrap_or_default();
    let def = map.get("*").cloned().unwrap_or_default();
    ItemSpec { include: own.include.or(def.include), name: own.name, description: own.description }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_defaults() {
        let m = MetadataFile::parse(
            "session:\n  description: test\n  timezone: '-05:00'\nsubject:\n  species: Rattus norvegicus\n\
electrode_groups:\n  - name: HDEMG\n    description: grid\n    location: diaphragm\n\
streams:\n  '*': { type: timeseries, unit: a.u. }\n  HDEG: { type: electrical, electrode_group: HDEMG }\n  SU_1: { include: false }\n",
        )
        .unwrap();
        assert_eq!(m.session.timezone.as_deref(), Some("-05:00"));
        assert_eq!(m.stream("HDEG").kind, Some(StreamType::Electrical));
        assert_eq!(m.stream("HDEG").unit.as_deref(), Some("a.u."));
        assert_eq!(m.stream("bpPe").kind, Some(StreamType::Timeseries));
        assert_eq!(m.stream("SU_1").include, Some(false));
        assert!(MetadataFile::parse("sesion: {}\n").is_err(), "typos are rejected");
    }
}
