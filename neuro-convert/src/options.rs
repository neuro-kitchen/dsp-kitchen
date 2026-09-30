/// Options shared by every input format.
#[derive(Debug, Clone, Default)]
pub struct OpenOptions {
    /// Only load these stores / streams (by name); `None` loads everything.
    pub only: Option<Vec<String>>,
    /// Container formats (e.g. a TDT tank): which block / session to open.
    pub block: Option<String>,
    /// Spike-sort result to apply to snippets (TDT: a folder name under `sort/`); `None`
    /// keeps the sort codes recorded online.
    pub sort: Option<String>,
}

impl OpenOptions {
    pub fn wants(&self, name: &str) -> bool {
        self.only.as_ref().is_none_or(|names| names.iter().any(|n| n == name))
    }
}
