use serde::{Deserialize, Serialize};

/// A small named table carried with a session (e.g. electrode impedance measurements).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub description: String,
    pub columns: Vec<String>,
    /// Row-major cells as text; outputs parse the columns they understand.
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn column(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }
}
