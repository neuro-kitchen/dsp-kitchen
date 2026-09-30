//! Persistence of the whole app: active module, each module's own session, recent files.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::modules::spikes::module::SpikeSession;
use crate::modules::time::module::TimeSession;

use super::model::Module;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub active: Module,
    pub time: TimeSession,
    pub spikes: SpikeSession,
    pub recent: Vec<PathBuf>,
}

impl Session {
    /// 3: split into Time / Spikes modules. Older files fail to parse and fall back to defaults.
    pub const VERSION: u32 = 3;

    /// `$XDG_CONFIG_HOME/croc-app/session.json` (or `~/.config/…`; `%APPDATA%` on Windows).
    pub fn path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("croc-app").join("session.json"))
    }

    pub fn load() -> Option<Session> {
        let text = std::fs::read_to_string(Self::path()?).ok()?;
        serde_json::from_str::<Session>(&text).ok().filter(|s| s.version == Self::VERSION)
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::path() else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}
