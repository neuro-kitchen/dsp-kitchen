//! What a recording session was, in NWB's own terms: the file's identifier, description and start
//! time, who recorded it, where, and the subject. Read from the data where it says (an NWB file's
//! `/general` and `/general/subject`, a SpikeGLX `.meta`'s `fileCreateTime`); what it does not say
//! is filled in by the person and kept in the recording's derived store (`<rec>.dspk.zarr`, root
//! attribute `nwb_session`, [`crate::derived::session_meta`]), never in the recording itself. The
//! names are NWB's, so an NWB export writes them as they are.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The subject (`/general/subject`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Subject {
    pub subject_id: String,
    pub species: String,
    pub sex: String,
    pub age: String,
    pub genotype: String,
    pub strain: String,
    pub description: String,
}

/// A session's metadata; an empty field is unknown.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionMetadata {
    pub identifier: String,
    pub session_description: String,
    /// ISO 8601 (`2026-11-17T10:52:03-05:00`).
    pub session_start_time: String,
    pub experimenter: Vec<String>,
    pub lab: String,
    pub institution: String,
    pub experiment_description: String,
    pub keywords: Vec<String>,
    pub subject: Subject,
}

/// One field: its key (as [`SessionMetadata::get`] names it), its label, its path in an NWB file,
/// whether NWB requires it, and what it holds.
#[derive(Debug, Clone, Copy)]
pub struct Field {
    pub key: &'static str,
    pub label: &'static str,
    pub nwb: &'static str,
    pub required: bool,
    pub hint: &'static str,
}

/// Every field, in the order shown.
pub const FIELDS: &[Field] = &[
    Field { key: "identifier", label: "Identifier", nwb: "identifier", required: true, hint: "A unique id for this session's file" },
    Field { key: "session_description", label: "Description", nwb: "session_description", required: true, hint: "What the session was" },
    Field { key: "session_start_time", label: "Start time", nwb: "session_start_time", required: true, hint: "ISO 8601, e.g. 2026-11-17T10:52:03-05:00" },
    Field { key: "experimenter", label: "Experimenters", nwb: "general/experimenter", required: false, hint: "Names, separated by ;" },
    Field { key: "lab", label: "Lab", nwb: "general/lab", required: false, hint: "" },
    Field { key: "institution", label: "Institution", nwb: "general/institution", required: false, hint: "" },
    Field { key: "experiment_description", label: "Experiment", nwb: "general/experiment_description", required: false, hint: "The experiment the session belongs to" },
    Field { key: "keywords", label: "Keywords", nwb: "general/keywords", required: false, hint: "Separated by ;" },
    Field { key: "subject.subject_id", label: "Subject id", nwb: "general/subject/subject_id", required: false, hint: "" },
    Field { key: "subject.species", label: "Species", nwb: "general/subject/species", required: false, hint: "Latin binomial, e.g. Mus musculus" },
    Field { key: "subject.sex", label: "Sex", nwb: "general/subject/sex", required: false, hint: "F, M, U or O" },
    Field { key: "subject.age", label: "Age", nwb: "general/subject/age", required: false, hint: "ISO 8601 duration, e.g. P90D" },
    Field { key: "subject.genotype", label: "Genotype", nwb: "general/subject/genotype", required: false, hint: "" },
    Field { key: "subject.strain", label: "Strain", nwb: "general/subject/strain", required: false, hint: "" },
    Field { key: "subject.description", label: "Subject notes", nwb: "general/subject/description", required: false, hint: "" },
];

/// List fields are written as one line, items separated by `;`.
pub const LIST_SEPARATOR: &str = "; ";

impl SessionMetadata {
    fn slot(&mut self, key: &str) -> Option<&mut String> {
        Some(match key {
            "identifier" => &mut self.identifier,
            "session_description" => &mut self.session_description,
            "session_start_time" => &mut self.session_start_time,
            "lab" => &mut self.lab,
            "institution" => &mut self.institution,
            "experiment_description" => &mut self.experiment_description,
            "subject.subject_id" => &mut self.subject.subject_id,
            "subject.species" => &mut self.subject.species,
            "subject.sex" => &mut self.subject.sex,
            "subject.age" => &mut self.subject.age,
            "subject.genotype" => &mut self.subject.genotype,
            "subject.strain" => &mut self.subject.strain,
            "subject.description" => &mut self.subject.description,
            _ => return None,
        })
    }

    /// Field `key` as one line (lists joined by [`LIST_SEPARATOR`]); "" when unknown.
    pub fn get(&self, key: &str) -> String {
        match key {
            "experimenter" => self.experimenter.join(LIST_SEPARATOR),
            "keywords" => self.keywords.join(LIST_SEPARATOR),
            k => self.clone().slot(k).map(|s| s.clone()).unwrap_or_default(),
        }
    }

    /// Sets field `key` from one line (trimmed; lists split at `;`).
    pub fn set(&mut self, key: &str, value: &str) {
        let list = || value.split(';').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect::<Vec<_>>();
        match key {
            "experimenter" => self.experimenter = list(),
            "keywords" => self.keywords = list(),
            k => {
                if let Some(s) = self.slot(k) {
                    *s = value.trim().to_string();
                }
            }
        }
    }

    /// Each field of `self`, else `under`'s: what the data says over what was filled in.
    pub fn over(&self, under: &SessionMetadata) -> SessionMetadata {
        let mut out = under.clone();
        for f in FIELDS {
            let v = self.get(f.key);
            if !v.is_empty() {
                out.set(f.key, &v);
            }
        }
        out
    }

    /// The fields NWB requires that are still empty.
    pub fn missing_required(&self) -> Vec<&'static Field> {
        FIELDS.iter().filter(|f| f.required && self.get(f.key).is_empty()).collect()
    }

    /// What a recording's reader already knows (its info's metadata): SpikeGLX's
    /// `fileCreateTime` is the session's start; NWB field names are taken as they are.
    pub fn from_reader(metadata: &BTreeMap<String, String>) -> SessionMetadata {
        let mut out = SessionMetadata::default();
        for f in FIELDS {
            if let Some(v) = metadata.get(f.key).or_else(|| metadata.get(f.nwb)) {
                out.set(f.key, v);
            }
        }
        if out.session_start_time.is_empty()
            && let Some(t) = metadata.get("fileCreateTime")
        {
            out.session_start_time = t.trim().to_string();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fields are read and written by key, lists split at `;`; what the data says wins over
    /// what was filled in; the required ones are named while empty.
    #[test]
    fn fields_by_key_and_the_data_over_the_form() {
        let mut filled = SessionMetadata::default();
        filled.set("experimenter", "Ada; Grace ;");
        filled.set("subject.species", " Mus musculus ");
        filled.set("session_description", "filled in");
        assert_eq!(filled.experimenter, ["Ada", "Grace"]);
        assert_eq!(filled.get("experimenter"), "Ada; Grace");
        assert_eq!(filled.subject.species, "Mus musculus");
        let from_file = SessionMetadata::from_reader(&[("fileCreateTime".to_string(), "2019-07-04T13:20:40".to_string()), ("session_description".to_string(), "from the file".to_string())].into());
        let both = from_file.over(&filled);
        assert_eq!((both.session_description.as_str(), both.session_start_time.as_str(), both.subject.species.as_str()), ("from the file", "2019-07-04T13:20:40", "Mus musculus"));
        assert_eq!(both.missing_required().iter().map(|f| f.key).collect::<Vec<_>>(), ["identifier"]);
    }
}
