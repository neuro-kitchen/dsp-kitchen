//! The two files a user edits by hand, in `$XDG_CONFIG_HOME/dspk-app/`:
//!
//! - `settings.toml`: the preferences (look, start, compute runtime, where fetched results go).
//!   The Settings screen writes the same file, keeping its comments and order; the app applies an
//!   edit made outside while it runs (the compute runtime at the next start).
//! - `keybindings.toml`: a key, or a list of keys, per command, in sections (`[plot]`,
//!   `[commands]`, and one per plugin). Read at start.
//!
//! A missing file means the defaults, and is written with every default so the user sees what can
//! be changed. A value that cannot be read keeps its default and is named in a notice.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use toml_edit::{DocumentMut, Item, Table, Value};

use crate::data::preferences::Session;

/// The preferences kept in `settings.toml`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Settings {
    /// `Some(true)`: dark, `Some(false)`: light, `None`: follow the system.
    pub theme: Option<bool>,
    /// `None`: the app's own accent.
    pub accent: Option<kitchen_ui::Rgb>,
    pub dark_plots: bool,
    pub reopen_last: bool,
    /// `None`: automatic (the first runtime compiled in).
    pub runtime: Option<String>,
    /// `None`: [`Session::default_results_dir`].
    pub results_dir: Option<PathBuf>,
    /// RAM the app may keep (decoded recording chunks, frozen processed views, spike windows), in
    /// bytes; `None`: automatic ([`automatic_memory`]).
    pub memory: Option<u64>,
}

/// The RAM limit when the setting is automatic: `DSPK_HOST_MEMORY` when set, else a quarter of the
/// computer's RAM (the app shares it with everything else); in a browser half of what it reports
/// (`navigator.deviceMemory`, at most 8 GB: 4 GiB), the page's tab being the program that uses it.
pub fn automatic_memory() -> u64 {
    if std::env::var_os("DSPK_HOST_MEMORY").is_some() {
        return dspk_core::memory::default_limit();
    }
    match total_memory() {
        0 => dspk_core::memory::default_limit(),
        total if cfg!(target_family = "wasm") => total / 2,
        total => total / 4,
    }
}

/// The computer's RAM in bytes (0: unknown). In a browser, what it reports (`navigator.deviceMemory`:
/// rounded, at most 8 GB; not every browser has it).
pub fn total_memory() -> u64 {
    #[cfg(target_family = "wasm")]
    {
        let navigator = js_sys::Reflect::get(&js_sys::global(), &"navigator".into()).ok();
        let gb = navigator.and_then(|n| js_sys::Reflect::get(&n, &"deviceMemory".into()).ok()).and_then(|v| v.as_f64()).unwrap_or(0.0);
        (gb * (1u64 << 30) as f64) as u64
    }
    #[cfg(not(target_family = "wasm"))]
    sysinfo::System::new().map_or(0, |mut sys| {
        sys.refresh_memory();
        sys.total_memory()
    })
}

/// Applies the RAM setting to the app's cache ([`dspk_core::memory::host_cache`]).
pub fn apply_memory(memory: Option<u64>) {
    dspk_core::memory::host_cache().set_limit(memory.unwrap_or_else(automatic_memory));
}

const TEMPLATE: &str = r##"# dspk-app settings. The Settings screen writes this file; edits made here apply while the app runs.
# Delete a line to go back to its default.

[appearance]
theme = "system"        # "system" | "dark" | "light"
accent = "app"          # "app" or a colour as "#rrggbb"
dark_plots = false      # keep plots dark in the light theme

[start]
reopen_last = false     # open the last recording when the app starts

[compute]
runtime = "automatic"   # "automatic" | "wgpu" | "cuda" | "cpu"; takes effect at the next start
memory = "automatic"    # RAM the app may keep to be fast, as "8GiB" or "512MiB"; "automatic" is a quarter of the RAM

[remote]
results_dir = ""        # where results fetched from servers go; "" is ~/dspk/remote
"##;

/// What `settings.toml` may hold: (section, key).
const KNOWN: &[(&str, &[&str])] = &[("appearance", &["theme", "accent", "dark_plots"]), ("start", &["reopen_last"]), ("compute", &["runtime", "memory"]), ("remote", &["results_dir"])];

impl Settings {
    pub fn from_session(s: &Session) -> Self {
        Self {
            theme: s.dark,
            accent: s.accent,
            dark_plots: s.dark_plots,
            reopen_last: s.reopen_last,
            runtime: s.runtime.clone().filter(|r| !r.is_empty()),
            results_dir: s.results_dir.clone(),
            memory: s.memory,
        }
    }

    pub fn apply_to(&self, s: &mut Session) {
        s.dark = self.theme;
        s.accent = self.accent;
        s.dark_plots = self.dark_plots;
        s.reopen_last = self.reopen_last;
        s.runtime = self.runtime.clone();
        s.results_dir = self.results_dir.clone();
        s.memory = self.memory;
        apply_memory(self.memory);
    }

    /// The settings in `text` and a notice per line that was not understood (which keeps its
    /// default). `Err`: the file is not TOML, nothing is applied.
    pub fn parse(text: &str) -> Result<(Self, Vec<String>), String> {
        let doc: DocumentMut = text.parse().map_err(|e| format!("settings.toml: {e}"))?;
        let mut notices = Vec::new();
        for (section, item) in doc.iter() {
            let Some(keys) = KNOWN.iter().find(|(s, _)| *s == section).map(|(_, k)| *k) else {
                notices.push(format!("settings.toml: unknown section [{section}]"));
                continue;
            };
            for (key, _) in item.as_table_like().into_iter().flat_map(|t| t.iter()) {
                if !keys.contains(&key) {
                    notices.push(format!("settings.toml: unknown setting {section}.{key}"));
                }
            }
        }
        let get = |section: &str, key: &str| doc.get(section).and_then(|s| s.get(key)).and_then(Item::as_value);
        let mut out = Self::default();
        let text_of = |section: &str, key: &str, notices: &mut Vec<String>| match get(section, key) {
            None => None,
            Some(v) => match v.as_str() {
                Some(t) => Some(t.trim().to_string()),
                None => {
                    notices.push(format!("settings.toml: {section}.{key} must be text in quotes"));
                    None
                }
            },
        };
        let flag = |section: &str, key: &str, notices: &mut Vec<String>| match get(section, key) {
            None => false,
            Some(v) => v.as_bool().unwrap_or_else(|| {
                notices.push(format!("settings.toml: {section}.{key} must be true or false"));
                false
            }),
        };
        match text_of("appearance", "theme", &mut notices).as_deref() {
            None | Some("system") => {}
            Some("dark") => out.theme = Some(true),
            Some("light") => out.theme = Some(false),
            Some(other) => notices.push(format!("settings.toml: appearance.theme {other:?} is not \"system\", \"dark\" or \"light\"")),
        }
        match text_of("appearance", "accent", &mut notices).as_deref() {
            None | Some("app") | Some("") => {}
            Some(colour) => match kitchen_ui::Rgb::parse(colour) {
                Some(rgb) => out.accent = Some(rgb),
                None => notices.push(format!("settings.toml: appearance.accent {colour:?} is not \"app\" or \"#rrggbb\"")),
            },
        }
        out.dark_plots = flag("appearance", "dark_plots", &mut notices);
        out.reopen_last = flag("start", "reopen_last", &mut notices);
        match text_of("compute", "runtime", &mut notices).as_deref() {
            None | Some("automatic") | Some("") => {}
            Some(name) => out.runtime = Some(name.to_string()),
        }
        match text_of("compute", "memory", &mut notices).as_deref() {
            None | Some("automatic") | Some("") => {}
            Some(size) => match dspk_core::memory::parse_bytes(size) {
                Some(bytes) => out.memory = Some(bytes.max(256 << 20)),
                None => notices.push(format!("settings.toml: compute.memory {size:?} is not \"automatic\" or a size such as \"8GiB\"")),
            },
        }
        match text_of("remote", "results_dir", &mut notices).as_deref() {
            None | Some("") => {}
            Some(dir) => out.results_dir = Some(expand_home(dir)),
        }
        Ok((out, notices))
    }

    /// `current` (the file as it is; empty or unreadable: the commented template) with these
    /// values, comments and order kept.
    pub fn write_into(&self, current: &str) -> String {
        let mut doc: DocumentMut = current.parse().ok().filter(|_| !current.trim().is_empty()).unwrap_or_else(|| TEMPLATE.parse().expect("the template is TOML"));
        let theme = match self.theme {
            None => "system",
            Some(true) => "dark",
            Some(false) => "light",
        };
        set(&mut doc, "appearance", "theme", theme.into());
        set(&mut doc, "appearance", "accent", self.accent.map(|c| format!("#{:06x}", c.0)).unwrap_or_else(|| "app".into()).into());
        set(&mut doc, "appearance", "dark_plots", self.dark_plots.into());
        set(&mut doc, "start", "reopen_last", self.reopen_last.into());
        set(&mut doc, "compute", "runtime", self.runtime.clone().unwrap_or_else(|| "automatic".into()).into());
        let memory = match self.memory {
            None => "automatic".to_string(),
            Some(b) if b % (1 << 30) == 0 => format!("{}GiB", b >> 30),
            Some(b) => format!("{}MiB", b >> 20),
        };
        set(&mut doc, "compute", "memory", memory.into());
        set(&mut doc, "remote", "results_dir", self.results_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_default().into());
        doc.to_string()
    }

    /// Reads the file at `path` into `session` (a missing file changes nothing); the notices.
    pub fn load_into(session: &mut Session, path: &Path) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
        match Self::parse(&text) {
            Ok((settings, notices)) => {
                settings.apply_to(session);
                notices
            }
            Err(e) => vec![e],
        }
    }
}

/// Sets `section.key`, keeping the comment after the value.
fn set(doc: &mut DocumentMut, section: &str, key: &str, mut new: Value) {
    if doc.get(section).and_then(Item::as_table_like).is_none() {
        doc.insert(section, Item::Table(Table::new()));
    }
    let Some(table) = doc.get_mut(section).and_then(Item::as_table_like_mut) else { return };
    match table.get_mut(key).and_then(Item::as_value_mut) {
        Some(old) => {
            *new.decor_mut() = old.decor().clone();
            *old = new;
        }
        None => {
            table.insert(key, Item::Value(new));
        }
    }
}

fn expand_home(dir: &str) -> PathBuf {
    match dir.strip_prefix("~/") {
        Some(rest) => dirs::home_dir().unwrap_or_default().join(rest),
        None => PathBuf::from(dir),
    }
}

// Key bindings

/// A section's commands with their keys.
pub type Keys = Vec<(String, Vec<String>)>;

#[derive(Default)]
struct Keymap {
    /// The file: section → command → keys.
    file: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    /// The defaults of every section resolved so far, in order (what the file is written from).
    defaults: Vec<(String, Keys)>,
    notices: Vec<String>,
}

static KEYMAP: Mutex<Option<Keymap>> = Mutex::new(None);

fn with_keymap<R>(f: impl FnOnce(&mut Keymap) -> R) -> R {
    let mut guard = KEYMAP.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Keymap::default))
}

/// Reads `keybindings.toml` (a missing file: every default). Call once at start, before any
/// section is resolved.
pub fn load_keybindings(path: &Path) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let (file, notices) = parse_keybindings(&text);
    with_keymap(|k| {
        k.file = file;
        k.notices.extend(notices);
    });
}

fn parse_keybindings(text: &str) -> (BTreeMap<String, BTreeMap<String, Vec<String>>>, Vec<String>) {
    let mut notices = Vec::new();
    let mut file = BTreeMap::new();
    let doc: DocumentMut = match text.parse() {
        Ok(d) => d,
        Err(e) => return (file, vec![format!("keybindings.toml: {e}")]),
    };
    for (section, item) in doc.iter() {
        let mut out = BTreeMap::new();
        match item.as_table_like() {
            Some(table) => flatten(section, "", table, &mut out, &mut notices),
            None => notices.push(format!("keybindings.toml: {section} must be a section ([{section}])")),
        }
        file.insert(section.to_string(), out);
    }
    (file, notices)
}

/// `a.b = "key"` is read as the command `a.b` (TOML makes it a table inside a table).
fn flatten(section: &str, prefix: &str, table: &dyn toml_edit::TableLike, out: &mut BTreeMap<String, Vec<String>>, notices: &mut Vec<String>) {
    for (key, item) in table.iter() {
        let name = if prefix.is_empty() { key.to_string() } else { format!("{prefix}.{key}") };
        if let Some(text) = item.as_str() {
            out.insert(name, if text.is_empty() { Vec::new() } else { vec![text.to_string()] });
        } else if let Some(list) = item.as_array() {
            let keys: Vec<String> = list.iter().filter_map(|v| v.as_str()).filter(|t| !t.is_empty()).map(str::to_string).collect();
            if keys.len() != list.iter().filter(|v| v.as_str() != Some("")).count() {
                notices.push(format!("keybindings.toml: [{section}] {name}: every key must be text in quotes"));
            } else {
                out.insert(name, keys);
            }
        } else if let Some(inner) = item.as_table_like() {
            flatten(section, &name, inner, out, notices);
        } else {
            notices.push(format!("keybindings.toml: [{section}] {name} must be a key in quotes, or a list of keys"));
        }
    }
}

/// The keys of `section`'s commands: the file's where it names the command, else `defaults`.
/// `valid` says whether a key can be bound (one that cannot keeps the default). Commands the file
/// names that `defaults` lacks, and one key on two commands, are noticed.
pub fn resolve(section: &str, defaults: &[(&str, &[&str])], valid: impl Fn(&str) -> bool) -> Keys {
    with_keymap(|k| {
        let defaults: Keys = defaults.iter().map(|(name, keys)| (name.to_string(), keys.iter().map(|s| s.to_string()).collect())).collect();
        let empty = BTreeMap::new();
        let file = k.file.get(section).unwrap_or(&empty);
        for name in file.keys() {
            if !defaults.iter().any(|(n, _)| n == name) {
                k.notices.push(format!("keybindings.toml: [{section}] has no command {name:?}"));
            }
        }
        let mut out = Keys::new();
        for (name, default) in &defaults {
            let keys = match file.get(name) {
                Some(keys) => match keys.iter().find(|key| !valid(key)) {
                    Some(bad) => {
                        k.notices.push(format!("keybindings.toml: [{section}] {name}: {bad:?} is not a key; the default is kept"));
                        default.clone()
                    }
                    None => keys.clone(),
                },
                None => default.clone(),
            };
            out.push((name.clone(), keys));
        }
        let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
        for (name, keys) in &out {
            for key in keys {
                if let Some(other) = seen.insert(key, name)
                    && other != name
                {
                    k.notices.push(format!("keybindings.toml: [{section}] {key:?} is bound to both {other} and {name}"));
                }
            }
        }
        k.defaults.retain(|(s, _)| s != section);
        k.defaults.push((section.to_string(), defaults));
        out
    })
}

/// Writes `keybindings.toml` with every default of the sections resolved so far, when the file
/// does not exist yet.
pub fn write_default_keybindings(path: &Path) -> std::io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let text = with_keymap(|k| default_keybindings(&k.defaults));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)
}

fn default_keybindings(sections: &[(String, Keys)]) -> String {
    let mut text = String::from(
        "# dspk-app key bindings: command = \"key\", or a list of keys; \"\" removes a binding.\n# Keys are written as ctrl-shift-o, alt-g, space, left, f3, +. Read when the app starts.\n# Delete a line to go back to its default.\n",
    );
    for (section, keys) in sections {
        text.push_str(&format!("\n[{section}]\n"));
        // A command id that is also the start of another (`a.b` and `a.b.c`) cannot be a dotted key
        let bare = |name: &str| name.split('.').all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')) && !keys.iter().any(|(other, _)| other.starts_with(&format!("{name}.")));
        for (name, bound) in keys {
            let name = if bare(name) { name.clone() } else { format!("{name:?}") };
            let value = match bound.as_slice() {
                [one] => format!("{one:?}"),
                many => format!("[{}]", many.iter().map(|k| format!("{k:?}")).collect::<Vec<_>>().join(", ")),
            };
            text.push_str(&format!("{name} = {value}\n"));
        }
    }
    text
}

/// The notices collected so far (each is given once).
pub fn take_notices() -> Vec<String> {
    with_keymap(|k| std::mem::take(&mut k.notices))
}

/// A key as the files write it, for matching by hand: (control or command, alt, shift, key).
/// `None`: not a key.
pub fn parse_key(text: &str) -> Option<(bool, bool, bool, String)> {
    let (mut cmd, mut alt, mut shift) = (false, false, false);
    let mut rest = text.trim();
    loop {
        let Some((head, tail)) = rest.split_once('-').filter(|(_, tail)| !tail.is_empty()) else { break };
        match head {
            "ctrl" | "cmd" | "secondary" | "super" => cmd = true,
            "alt" => alt = true,
            "shift" => shift = true,
            _ => return None,
        }
        rest = tail;
    }
    (!rest.is_empty() && !rest.contains(' ')).then(|| (cmd, alt, shift, rest.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_keep_the_comments() {
        let s = Settings { theme: Some(false), accent: Some(kitchen_ui::Rgb(0x2b7f86)), dark_plots: true, reopen_last: true, runtime: Some("cuda".into()), results_dir: Some("/data/results".into()), memory: Some(8 << 30) };
        let text = s.write_into("");
        assert!(text.contains("theme = \"light\"        # \"system\" | \"dark\" | \"light\""), "{text}");
        assert!(text.contains("accent = \"#2b7f86\""), "{text}");
        assert_eq!(Settings::parse(&text).unwrap(), (s.clone(), vec![]));
        // The user's own comment and order survive a write from the Settings screen
        let edited = "# mine\n[compute]\nruntime = \"cpu\" # slow\n";
        let (read, notices) = Settings::parse(edited).unwrap();
        assert_eq!((read.runtime.as_deref(), notices.len()), (Some("cpu"), 0));
        let written = Settings { runtime: Some("wgpu".into()), ..read }.write_into(edited);
        assert!(written.starts_with("# mine\n[compute]\nruntime = \"wgpu\" # slow\n"), "{written}");
        assert_eq!(Settings::parse(&written).unwrap().0.runtime.as_deref(), Some("wgpu"));
    }

    #[test]
    fn the_defaults_are_the_template_and_a_missing_line_is_its_default() {
        let text = Settings::default().write_into("");
        assert_eq!(text, TEMPLATE);
        assert_eq!(Settings::parse("").unwrap(), (Settings::default(), vec![]));
        assert_eq!(Settings::parse(TEMPLATE).unwrap(), (Settings::default(), vec![]));
    }

    #[test]
    fn what_is_not_understood_keeps_its_default_and_is_named() {
        let (s, notices) = Settings::parse("[appearance]\ntheme = \"blue\"\naccent = \"rose\"\ndark_plots = \"yes\"\nfont = 3\n[colours]\n").unwrap();
        assert_eq!(s, Settings::default());
        assert_eq!(notices.len(), 5, "{notices:?}");
        assert!(notices.iter().any(|n| n.contains("appearance.font")) && notices.iter().any(|n| n.contains("[colours]")));
        assert!(Settings::parse("[appearance\n").is_err());
    }

    #[test]
    fn key_bindings_override_defaults_by_command() {
        let (file, notices) = parse_keybindings("[plot]\nplay_pause = \"p\"\nzoom_in = [\"+\", \"=\"]\npan_back = \"\"\n[commands]\nfile.open = \"ctrl-shift-o\"\n\"view.dark\" = \"f9\"\nbad = 3\n");
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert_eq!(file["plot"]["play_pause"], vec!["p"]);
        assert_eq!(file["plot"]["zoom_in"], vec!["+", "="]);
        assert!(file["plot"]["pan_back"].is_empty(), "an empty key removes the binding");
        assert_eq!(file["commands"]["file.open"], vec!["ctrl-shift-o"], "a dotted key is the command's id");
        assert_eq!(file["commands"]["view.dark"], vec!["f9"]);
    }

    #[test]
    fn the_default_file_reads_back_as_the_defaults() {
        let sections = vec![("commands".to_string(), vec![("file.open".to_string(), vec!["ctrl-o".to_string()]), ("a.b".to_string(), vec![]), ("a.b.c".to_string(), vec!["f2".to_string(), "f3".to_string()])])];
        let text = default_keybindings(&sections);
        let (file, notices) = parse_keybindings(&text);
        assert!(notices.is_empty(), "{notices:?}\n{text}");
        assert_eq!(file["commands"]["file.open"], vec!["ctrl-o"]);
        assert!(file["commands"]["a.b"].is_empty());
        assert_eq!(file["commands"]["a.b.c"], vec!["f2", "f3"]);
    }

    #[test]
    fn keys_parse_for_matching_by_hand() {
        assert_eq!(parse_key("alt-g"), Some((false, true, false, "g".into())));
        assert_eq!(parse_key("ctrl-shift-z"), Some((true, false, true, "z".into())));
        assert_eq!(parse_key("shift-space"), Some((false, false, true, "space".into())));
        assert_eq!(parse_key("-"), Some((false, false, false, "-".into())));
        assert_eq!(parse_key("ctrl--"), Some((true, false, false, "-".into())));
        assert_eq!(parse_key("hyper-g"), None);
        assert_eq!(parse_key(""), None);
    }
}
