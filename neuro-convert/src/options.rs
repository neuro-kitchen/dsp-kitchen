/// Options shared by every input format.
#[derive(Debug, Clone, Default)]
pub struct OpenOptions {
    /// Only load these stores / streams (by name); `None` loads everything.
    pub only: Option<Vec<String>>,
}

impl OpenOptions {
    pub fn wants(&self, name: &str) -> bool {
        self.only.as_ref().is_none_or(|names| names.iter().any(|n| n == name))
    }
}
