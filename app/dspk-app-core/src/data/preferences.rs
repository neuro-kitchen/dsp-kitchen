//! What the app remembers between runs: recent recordings (newest first), the workspace shown.
//!
//! Its own file, `$XDG_CONFIG_HOME/dspk-app/workbench.json`, so the Slint app (which keeps
//! `session.json` in the same folder) and this one never overwrite each other. On a first run the
//! recent list is taken from the Slint app's file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::trace::view::TimeView;
use crate::navigation::route::Workspace;

/// Recent recordings kept.
const RECENT: usize = 10;

/// Saved per-view settings in the Explore workspace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedTimeView {
    pub kind: TimeViewKind,
    #[serde(default)]
    pub source: String,
    #[serde(default = "default_lanes")]
    pub lanes: usize,
    #[serde(default = "default_gain")]
    pub gain: f32,
    #[serde(default = "yes")]
    pub auto_scale: bool,
    #[serde(default = "yes")]
    pub remove_dc: bool,
    #[serde(default)]
    pub pinned_selection: bool,
    #[serde(default = "yes")]
    pub spikes: bool,
    #[serde(default)]
    pub color_by: crate::screens::explore::trace::renderer::ColorBy,
}

impl SavedTimeView {
    pub fn from_view(v: &TimeView) -> Self {
        Self {
            kind: v.kind,
            source: v.source.clone(),
            lanes: v.lanes,
            gain: v.gain,
            auto_scale: v.auto_scale,
            remove_dc: v.remove_dc,
            pinned_selection: v.pinned_selection,
            spikes: v.spikes,
            color_by: v.color_by,
        }
    }

    pub fn apply(&self, v: &mut TimeView) {
        v.lanes = self.lanes.max(1);
        v.gain = self.gain.clamp(0.1, 20.0);
        v.auto_scale = self.auto_scale;
        v.remove_dc = self.remove_dc;
        v.pinned_selection = self.pinned_selection;
        v.spikes = self.spikes;
        v.color_by = self.color_by;
        v.needs_render = true;
    }
}

fn default_lanes() -> usize {
    8
}
fn default_gain() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}

/// Saved Explore workspace layout (side-dock states and open views).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploreSession {
    #[serde(default = "yes")]
    pub left_open: bool,
    #[serde(default = "yes")]
    pub right_open: bool,
    #[serde(default = "yes")]
    pub bottom_open: bool,
    #[serde(default)]
    pub views: Vec<SavedTimeView>,
    /// The panel shown on the right side bar (`processing`, `display`, `plugin0`, …).
    #[serde(default)]
    pub right_panel: String,
}

impl Default for ExploreSession {
    fn default() -> Self {
        Self { left_open: true, right_open: true, bottom_open: true, views: Vec::new(), right_panel: String::new() }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    #[serde(default)]
    pub recent: Vec<PathBuf>,
    #[serde(default)]
    pub recent_sortings: Vec<PathBuf>,
    #[serde(default)]
    pub workspace: Workspace,
    /// `Some(true)`: dark, `Some(false)`: light, `None`: follow the system.
    #[serde(default)]
    pub dark: Option<bool>,
    /// "Tune this device" was suggested (once, at the first launch).
    #[serde(default)]
    pub tune_suggested: bool,
    /// Keep plots dark in the light theme.
    #[serde(default)]
    pub dark_plots: bool,
    #[serde(default)]
    pub explore: ExploreSession,
    /// Widths (heights for bottom panels) of resizable panels, by key (`explore.left`, …).
    #[serde(default)]
    pub panel_widths: std::collections::BTreeMap<String, f32>,
    /// Side panels shown or hidden, by key (`sorting.left`, …).
    #[serde(default)]
    pub panel_open: std::collections::BTreeMap<String, bool>,
    /// Servers (`dspk serve`) the user works with (Settings).
    #[serde(default)]
    pub servers: Vec<crate::data::remote::ServerProfile>,
    /// Where results fetched from servers go (`None`: [`Session::default_results_dir`]).
    #[serde(default)]
    pub results_dir: Option<PathBuf>,
    /// The compute runtime (`wgpu`, `cuda`, `cpu`; `None`: the first that is compiled in). Takes
    /// effect at the next start.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Open the last recording when the app starts (default off: the start screen).
    #[serde(default)]
    pub reopen_last: bool,
}

impl Session {
    pub const VERSION: u32 = 1;

    pub fn new() -> Self {
        Self { version: Self::VERSION, ..Self::default() }
    }

    /// Results fetched from servers by default: `~/dspk/remote` (the home folder, or the current
    /// one when there is no home).
    pub fn default_results_dir() -> PathBuf {
        dirs::home_dir().unwrap_or_default().join("dspk").join("remote")
    }

    /// Where results fetched from servers go.
    pub fn results_dir(&self) -> PathBuf {
        self.results_dir.clone().unwrap_or_else(Self::default_results_dir)
    }

    /// The user's recipe library: `<config>/dspk-app/recipes/`.
    pub fn recipes_dir() -> Option<PathBuf> {
        Some(Self::config_dir()?.join("dspk-app").join("recipes"))
    }

    fn config_dir() -> Option<PathBuf> {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    }

    pub fn path() -> Option<PathBuf> {
        Some(Self::config_dir()?.join("dspk-app").join("workbench.json"))
    }

    /// The session document of the open work (views, op chains, linked sortings), next to the
    /// preferences: `current-session.json`. An agent edits it; the app picks the edits up.
    pub fn document_path() -> Option<PathBuf> {
        Some(Self::config_dir()?.join("dspk-app").join("current-session.json"))
    }

    /// The saved session; else the Slint app's recent files; else an empty one.
    pub fn load(path: Option<&Path>) -> Self {
        let read = |p: &Path| std::fs::read_to_string(p).ok();
        if let Some(s) = path.and_then(read).and_then(|t| serde_json::from_str::<Session>(&t).ok()).filter(|s| s.version == Self::VERSION) {
            return s;
        }
        let slint = path.and_then(|p| p.parent()).map(|d| d.join("session.json"));
        let recent = slint
            .as_deref()
            .and_then(read)
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| serde_json::from_value::<Vec<PathBuf>>(v.get("recent")?.clone()).ok())
            .unwrap_or_default();
        Self { recent, ..Self::new() }
    }

    pub fn save(&self, path: Option<&Path>) -> std::io::Result<()> {
        let Some(path) = path else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)
    }

    /// Forgets `path` from the recent recordings and sortings.
    pub fn forget_recent(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent_sortings.retain(|p| p != path);
    }

    /// Forgets every recent recording and sorting.
    pub fn clear_recent(&mut self) {
        self.recent.clear();
        self.recent_sortings.clear();
    }

    /// `path` becomes the newest recent recording.
    pub fn push_recent(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(RECENT);
    }

    /// `path` becomes the newest recent sorting folder.
    #[allow(dead_code)] // Curation (parked) opens sortings
    pub fn push_recent_sorting(&mut self, path: &Path) {
        self.recent_sortings.retain(|p| p != path);
        self.recent_sortings.insert(0, path.to_path_buf());
        self.recent_sortings.truncate(RECENT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dspk-app-session-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_round_trip_and_recent_order() {
        let dir = scratch("round");
        let path = dir.join("workbench.json");
        let mut s = Session::new();
        for p in ["/a", "/b", "/a"] {
            s.push_recent(Path::new(p));
        }
        assert_eq!(s.recent, vec![PathBuf::from("/a"), PathBuf::from("/b")]);
        s.workspace = Workspace::Curation;
        s.save(Some(&path)).unwrap();
        assert_eq!(Session::load(Some(&path)), s);
    }

    #[test]
    fn recent_entries_are_forgotten_and_the_last_file_is_not_reopened_by_default() {
        let mut s = Session::new();
        assert!(!s.reopen_last, "start on the start screen unless Settings says otherwise");
        // An older session file without the field: off too
        let old: Session = serde_json::from_str(r#"{"version": 1, "recent": ["/a"]}"#).unwrap();
        assert!(!old.reopen_last);
        s.push_recent(Path::new("/a"));
        s.push_recent(Path::new("/b"));
        s.push_recent_sorting(Path::new("/a"));
        s.forget_recent(Path::new("/a"));
        assert_eq!((s.recent.clone(), s.recent_sortings.len()), (vec![PathBuf::from("/b")], 0));
        s.push_recent_sorting(Path::new("/c"));
        s.clear_recent();
        assert!(s.recent.is_empty() && s.recent_sortings.is_empty());
    }

    #[test]
    fn test_first_run_takes_the_slint_apps_recent_files() {
        let dir = scratch("slint");
        std::fs::write(dir.join("session.json"), r#"{"version": 3, "recent": ["/data/x.bin"], "active": "Time"}"#).unwrap();
        let s = Session::load(Some(&dir.join("workbench.json")));
        assert_eq!(s.recent, vec![PathBuf::from("/data/x.bin")]);
        assert_eq!(s.workspace, Workspace::Explore);
    }
}
