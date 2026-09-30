//! Synapse sidecars: `Notes.txt` and `StoresListing.txt`.

use std::collections::BTreeMap;

/// `Notes.txt`: `Key: value` header lines (Experiment, Subject, User, Start, Stop) and any
/// runtime notes typed during the recording.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynapseNotes {
    pub fields: BTreeMap<String, String>,
    pub notes: Vec<String>,
}

const HEADER_KEYS: [&str; 5] = ["Experiment", "Subject", "User", "Start", "Stop"];

pub fn parse_notes(text: &str) -> SynapseNotes {
    let mut out = SynapseNotes::default();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match line.split_once(':') {
            Some((k, v)) if HEADER_KEYS.contains(&k.trim()) => {
                out.fields.insert(k.trim().to_string(), v.trim().to_string());
            }
            _ => out.notes.push(line.to_string()),
        }
    }
    out
}

/// One store as described by `StoresListing.txt`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoreDescription {
    /// Gizmo / hardware object that wrote the store (e.g. `HDEMG`).
    pub object: String,
    /// Object type (e.g. `Stream Data Storage`).
    pub object_type: String,
    /// `Format`, `Scale`, `Rate`, `Mode`, `Duration`, … as written.
    pub properties: BTreeMap<String, String>,
    /// Signal source from the flat listing (e.g. `Streaming: ~PZAn(1).HDEMG`).
    pub source: Option<String>,
}

/// `StoresListing.txt`: stores grouped by the object that wrote them, then a flat listing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoresListing {
    pub stores: BTreeMap<String, StoreDescription>,
    /// Hardware objects (`RZ2(1) - RZn Processor`, `IZV10(1) - IZV`), i.e. names with a unit index.
    pub hardware: Vec<(String, String)>,
}

pub fn parse_stores_listing(text: &str) -> StoresListing {
    let mut out = StoresListing::default();
    let (mut object, mut object_type) = (String::new(), String::new());
    let mut current: Option<String> = None;
    let mut in_flat = false;

    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with("Flat Listing") {
            in_flat = true;
            continue;
        }
        if in_flat {
            // StoreID  Gizmo  Description...
            let mut parts = line.split_whitespace();
            if let (Some(id), Some(_gizmo)) = (parts.next(), parts.next()) {
                let rest: Vec<&str> = parts.collect();
                if id != "StoreID" && !rest.is_empty() {
                    let d = out.stores.entry(id.to_string()).or_default();
                    d.source = Some(rest.join(" "));
                }
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else { continue };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "Object ID" => {
                let (name, ty) = value.split_once(" - ").unwrap_or((value, ""));
                object = name.trim().to_string();
                object_type = ty.trim().to_string();
                current = None;
                if object.ends_with(')') && object.contains('(') {
                    out.hardware.push((object.clone(), object_type.clone()));
                }
            }
            "Store ID" => {
                let d = out.stores.entry(value.to_string()).or_default();
                d.object = object.clone();
                d.object_type = object_type.clone();
                current = Some(value.to_string());
            }
            // Before the first object: session header (Experiment, Subject, …)
            _ => {
                if let Some(id) = &current {
                    out.stores.get_mut(id).unwrap().properties.insert(key.to_string(), value.to_string());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notes_header_and_free_text() {
        let n = parse_notes("Experiment: HD_MEPs_v2\r\nSubject: rat1\r\nStart: 3:25:56pm 02/26/2025\r\n\r\nStimulus threshold found\r\nStop: 4:13:09pm 02/26/2025\r\n");
        assert_eq!(n.fields["Subject"], "rat1");
        assert_eq!(n.fields["Stop"], "4:13:09pm 02/26/2025");
        assert_eq!(n.notes, vec!["Stimulus threshold found"]);
    }

    #[test]
    fn test_stores_listing() {
        let text = "Experiment: X\r\n\r\nObject ID : RZ2(1) - RZn Processor\r\n Rate     : 24414.1 Hz\r\n Store ID : Tick\r\n\r\n\
Object ID : HDEMG - Stream Data Storage\r\n Store ID : HDEG\r\n  Format  : Float-32\r\n  Rate    : 24414.1 Hz\r\n\r\n\
Flat Listing:\r\nStoreID  Gizmo/Hal      Description\r\nHDEG     HDEMG          Streaming: ~PZAn(1).HDEMG\r\n";
        let l = parse_stores_listing(text);
        assert_eq!(l.hardware, vec![("RZ2(1)".to_string(), "RZn Processor".to_string())]);
        let h = &l.stores["HDEG"];
        assert_eq!(h.object, "HDEMG");
        assert_eq!(h.properties["Format"], "Float-32");
        assert_eq!(h.source.as_deref(), Some("Streaming: ~PZAn(1).HDEMG"));
        assert!(l.stores["Tick"].properties.is_empty());
    }
}
