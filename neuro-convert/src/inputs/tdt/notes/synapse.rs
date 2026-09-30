//! Synapse sidecars: `Notes.txt` and `StoresListing.txt`.

use std::collections::BTreeMap;

/// `Notes.txt`: `Key: value` header lines (Experiment, Subject, User, Start, Stop), runtime
/// notes (`Note-<n>: <clock> [<button>] "<text>"`, text may span lines) and any other text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynapseNotes {
    pub fields: BTreeMap<String, String>,
    pub entries: Vec<NoteEntry>,
    /// Lines that are neither header fields nor notes.
    pub notes: Vec<String>,
}

/// One runtime note.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoteEntry {
    pub index: u32,
    /// Wall-clock time as written, e.g. `12:09:59pm`.
    pub clock: String,
    /// Note button pressed (`sleep`); `None` for typed notes (`[none]`) or no buttons.
    pub button: Option<String>,
    pub text: String,
}

impl NoteEntry {
    /// `button: text`, `button` or `text`.
    pub fn label(&self) -> String {
        match (&self.button, self.text.trim()) {
            (Some(b), "") => b.clone(),
            (Some(b), t) => format!("{b}: {t}"),
            (None, t) => t.to_string(),
        }
    }
}

const HEADER_KEYS: [&str; 5] = ["Experiment", "Subject", "User", "Start", "Stop"];

pub fn parse_notes(text: &str) -> SynapseNotes {
    let mut out = SynapseNotes::default();
    let mut open: Option<NoteEntry> = None; // a note whose quoted text continues on later lines
    for raw in text.lines() {
        if let Some(mut e) = open.take() {
            match raw.split_once('"') {
                Some((rest, _)) => {
                    e.text.push('\n');
                    e.text.push_str(rest);
                    out.entries.push(e);
                }
                None => {
                    e.text.push('\n');
                    e.text.push_str(raw);
                    open = Some(e);
                }
            }
            continue;
        }
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("Note-") {
            let (index, body) = rest.split_once(':').unwrap_or((rest, ""));
            let body = body.trim();
            let clock = body.split_whitespace().next().unwrap_or("").to_string();
            let button = body.split_once('[').and_then(|(_, r)| r.split_once(']')).map(|(b, _)| b.trim().to_string()).filter(|b| b != "none");
            let mut e = NoteEntry { index: index.trim().parse().unwrap_or(0), clock, button, text: String::new() };
            let mut quoted = body.splitn(3, '"');
            quoted.next();
            match (quoted.next(), quoted.next()) {
                (Some(t), Some(_)) => {
                    e.text = t.to_string();
                    out.entries.push(e);
                }
                (Some(t), None) => {
                    e.text = t.to_string();
                    open = Some(e);
                }
                _ => out.entries.push(e),
            }
            continue;
        }
        match line.split_once(':') {
            Some((k, v)) if HEADER_KEYS.contains(&k.trim()) => {
                out.fields.insert(k.trim().to_string(), v.trim().to_string());
            }
            _ => out.notes.push(line.to_string()),
        }
    }
    out.entries.extend(open);
    out
}

/// Seconds since midnight for `h:mm:ss[am|pm]` (12- or 24-hour).
pub fn clock_seconds(clock: &str) -> Option<f64> {
    let c = clock.trim().to_ascii_lowercase();
    let (time, pm) = match (c.strip_suffix("pm"), c.strip_suffix("am")) {
        (Some(t), _) => (t, Some(true)),
        (_, Some(t)) => (t, Some(false)),
        _ => (c.as_str(), None),
    };
    let mut parts = time.split(':');
    let mut h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let s: f64 = parts.next().unwrap_or("0").parse().ok()?;
    match pm {
        Some(true) if h < 12.0 => h += 12.0,
        Some(false) if h == 12.0 => h = 0.0,
        _ => {}
    }
    Some(h * 3600.0 + m * 60.0 + s)
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
    fn test_runtime_notes() {
        let n = parse_notes(
            "Start: 12:09:55pm 04/26/2018\r\n\r\nNote-1: 12:09:59pm [sleep] \"\"\r\nNote-2: 10:07:00am [none] \"Bottle In\"\r\n\
Note-3: 12:10:20pm \"two\r\nlines\"\r\nStop: 12:10:28pm 04/26/2018\r\n",
        );
        let labels: Vec<String> = n.entries.iter().map(NoteEntry::label).collect();
        assert_eq!(labels, vec!["sleep", "Bottle In", "two\nlines"]);
        assert_eq!((n.entries[0].index, n.entries[0].clock.as_str()), (1, "12:09:59pm"));
        assert!(n.notes.is_empty());
        assert_eq!(clock_seconds("12:09:59pm"), Some(12.0 * 3600.0 + 9.0 * 60.0 + 59.0));
        assert_eq!(clock_seconds("12:00:01am"), Some(1.0));
        assert_eq!(clock_seconds("10:07:00am"), Some(36420.0));
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
