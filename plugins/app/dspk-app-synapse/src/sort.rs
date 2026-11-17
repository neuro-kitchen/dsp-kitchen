//! Sorting from the app: runs the separate `dspk-cli-sorter` program (every sorter, GPL-3.0; the
//! app never links it), the same program `dspk sort` runs, and links its `.sorting.zarr` to the
//! view.

use std::path::{Path, PathBuf};
use std::process::Command;

use dspk_app_core::data::jobs::{self, Cancel, JobMessage};

/// Sorters of `dspk-cli-sorter` (its subcommands) and their names, in menu order.
pub const SORTERS: [(&str, &str); 5] = [
    ("kilosort4", "Kilosort4"),
    ("emusort", "EMUsort"),
    ("mountainsort5", "MountainSort 5"),
    ("spykingcircus2", "SpyKING CIRCUS 2"),
    ("tridesclous2", "Tridesclous 2"),
];

/// Probe presets of `--probe` (`None`: the geometry the recording carries).
pub const PROBES: [(Option<&str>, &str); 7] = [
    (None, "From the recording"),
    (Some("neuropixels1"), "Neuropixels 1.0"),
    (Some("neuropixels2"), "Neuropixels 2.0"),
    (Some("tetrode"), "Tetrode"),
    (Some("utah"), "Utah array"),
    (Some("hdemg4x8"), "HD-EMG 4×8"),
    (Some("hdemg8x8"), "HD-EMG 8×8"),
];

/// Environment variable naming a `dspk-cli-sorter` executable (as for the CLI).
pub const SORTER_BIN_ENV: &str = "DSPK_CLI_SORTER_BIN";

/// The program's file name on this platform.
fn program_file_name() -> String {
    format!("dspk-cli-sorter{}", std::env::consts::EXE_SUFFIX)
}

/// The `dspk-cli-sorter` to run: `$DSPK_CLI_SORTER_BIN`, else next to this program (a source
/// build puts both in one folder), else the one installed for this version (dspk-hub: what
/// `dspk sorters install` and this app's Install button install). Not in a browser.
pub fn locate() -> Option<PathBuf> {
    #[cfg(not(target_family = "wasm"))]
    {
        // `beside` is this program itself: the sorter is looked for in its folder
        let exe = std::env::current_exe().ok();
        dspk_hub::sorters::locate(env!("CARGO_PKG_VERSION"), exe.as_deref())
    }
    #[cfg(target_family = "wasm")]
    None
}

/// Downloads this version's `dspk-cli-sorter` for this computer from the GitHub release, checked
/// against the release's manifest (SHA-256 and size), into the user data folder; its path.
pub fn install() -> Result<PathBuf, String> {
    #[cfg(not(target_family = "wasm"))]
    {
        dspk_hub::sorters::install(env!("CARGO_PKG_VERSION")).map_err(|e| format!("{e:#}"))
    }
    #[cfg(target_family = "wasm")]
    Err("sorters run in the desktop app or on a server".into())
}

/// What to sort.
#[derive(Debug, Clone)]
pub struct SortRequest {
    /// `dspk-cli-sorter` subcommand ([`SORTERS`]).
    pub sorter: String,
    pub recording: PathBuf,
    pub source: Option<String>,
    /// `--probe` preset; `None`: the recording's own geometry.
    pub probe: Option<String>,
    /// `<name>.<sorter>.sorting.zarr`.
    pub output: PathBuf,
    /// Sorter parameters set away from their defaults: (`--flag`, value).
    pub params: Vec<(String, String)>,
    /// The part of the recording to sort (s); `None`: from the start / to the end.
    pub start_sec: Option<f64>,
    pub duration_sec: Option<f64>,
}

/// Runs `dspk-cli-sorter <sorter> <recording> -o <output>`, reporting its progress lines.
pub fn sort(req: &SortRequest, send: impl Fn(JobMessage) + Send + Sync + 'static) -> Cancel {
    let Some(program) = locate() else {
        send(JobMessage::Failed(format!("{} not found: Install sorters below (or `dspk sorters install`), or set {SORTER_BIN_ENV}", program_file_name())));
        return Cancel::default();
    };
    let mut cmd = Command::new(program);
    // Progress as JSON lines (stages and fractions), read by the job
    cmd.arg("--progress").arg("json").arg(&req.sorter).arg(&req.recording).arg("-o").arg(&req.output);
    if let Some(s) = &req.source {
        cmd.arg("--source").arg(s);
    }
    if let Some(p) = &req.probe {
        cmd.arg("--probe").arg(p);
    }
    if let Some(t) = req.start_sec {
        cmd.arg("--start-sec").arg(t.to_string());
    }
    if let Some(d) = req.duration_sec {
        cmd.arg("--duration-sec").arg(d.to_string());
    }
    for (flag, value) in &req.params {
        cmd.arg(flag).arg(value);
    }
    jobs::run_program(cmd, req.output.clone(), send)
}

/// One parameter of a sorter, as `dspk-cli-sorter describe` lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub flag: String,
    pub help: String,
    /// The default as text (lists joined by commas); empty when it has none.
    pub default: String,
    pub choices: Vec<String>,
}

/// A sorter and its parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SorterInfo {
    pub name: String,
    pub about: String,
    pub params: Vec<Param>,
}

/// What the installed program says about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Described {
    pub program: PathBuf,
    pub version: String,
    pub sorters: Vec<SorterInfo>,
}

/// Runs `dspk-cli-sorter describe` (blocking: call it off the UI thread).
pub fn describe() -> Result<Described, String> {
    let program = locate().ok_or_else(|| format!("{} not found: Install sorters below (or `dspk sorters install`), or set {SORTER_BIN_ENV}", program_file_name()))?;
    let out = Command::new(&program).arg("describe").output().map_err(|e| format!("{}: {e}", dspk_session::paths::short_path(&program)))?;
    if !out.status.success() {
        return Err(format!("{} describe: {}", dspk_session::paths::short_path(&program), String::from_utf8_lossy(&out.stderr).trim()));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("{} describe: {e}", dspk_session::paths::short_path(&program)))?;
    parse_describe(program, &v)
}

/// The sorters of a `describe` document (interface 1).
pub fn parse_describe(program: PathBuf, v: &serde_json::Value) -> Result<Described, String> {
    if v["interface"].as_u64() != Some(1) {
        return Err(format!("{}: describe interface {} (this app reads 1): update the program", dspk_session::paths::short_path(&program), v["interface"]));
    }
    let text = |x: &serde_json::Value| match x {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(a) => a.iter().map(|e| e.as_str().map_or_else(|| e.to_string(), str::to_string)).collect::<Vec<_>>().join(","),
        other => other.to_string(),
    };
    let sorters = v["sorters"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|s| SorterInfo {
                    name: text(&s["name"]),
                    about: text(&s["about"]),
                    params: s["parameters"]
                        .as_array()
                        .map(|ps| {
                            ps.iter()
                                .map(|p| Param {
                                    name: text(&p["name"]),
                                    flag: text(&p["flag"]),
                                    help: text(&p["help"]),
                                    default: text(&p["default"]),
                                    choices: p["choices"].as_array().map(|c| c.iter().map(text).collect()).unwrap_or_default(),
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Described { program, version: text(&v["version"]), sorters })
}

/// `rec.bin` sorted by `sorter` over `start`…`start + duration` s →
/// `rec.<sorter>.<start>-<end>s.sorting.zarr` next to it (the whole recording: [`sorting_path`]).
pub fn run_path(recording: &Path, sorter: &str, start_sec: Option<f64>, duration_sec: Option<f64>) -> PathBuf {
    match (start_sec, duration_sec) {
        (None, None) => sorting_path(recording, sorter),
        (s, d) => {
            let s = s.unwrap_or(0.0);
            let end = d.map_or_else(|| "end".to_string(), |d| format!("{}", trim(s + d)));
            sorting_path(recording, &format!("{sorter}.{}-{end}s", trim(s)))
        }
    }
}

fn trim(v: f64) -> String {
    let t = format!("{v:.3}");
    t.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `run` (as [`run_path`] names it) with the time it starts in its name, UTC to the minute:
/// `rec.kilosort4.0-20s.20261117-2032.sorting.zarr`, so runs of the same part never collide and
/// say when they were made.
pub fn stamped(run: &Path, unix_secs: u64) -> PathBuf {
    let t = dspk_core::time::wall::rfc3339_utc(unix_secs); // 2026-11-17T20:32:03Z
    let stamp = format!("{}{}{}-{}{}", &t[0..4], &t[5..7], &t[8..10], &t[11..13], &t[14..16]);
    let name = run.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let stem = name.strip_suffix(".sorting.zarr").unwrap_or(&name);
    run.with_file_name(format!("{stem}.{stamp}.sorting.zarr"))
}

/// Whether `other` is a run of the same sorter over the same part as `run` (unstamped, as
/// [`run_path`] names it), by name: `other` is `run`, or `run` with a stamp, or with a `-2`…
/// (the folder is not compared: a server's runs are named by their path on it)
pub fn same_run(run: &Path, other: &Path) -> bool {
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let (base, other) = (name(run), name(other));
    let Some(stem) = base.strip_suffix(".sorting.zarr") else { return false };
    let Some(rest) = other.strip_suffix(".sorting.zarr").and_then(|o| o.strip_prefix(stem)) else { return false };
    rest.is_empty() || stamp_of(rest).is_some() || rest.strip_prefix('-').is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// `.20261117-2032` → its date and time as people read it (`17 Nov 2026, 20:32 UTC`).
fn stamp_of(rest: &str) -> Option<String> {
    let s = rest.strip_prefix('.')?;
    let (date, time) = s.split_once('-')?;
    if date.len() != 8 || time.len() != 4 || !date.chars().chain(time.chars()).all(|c| c.is_ascii_digit()) {
        return None;
    }
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let month = MONTHS.get(date[4..6].parse::<usize>().ok()?.checked_sub(1)?)?;
    Some(format!("{} {month} {}, {}:{} UTC", date[6..8].trim_start_matches('0'), &date[0..4], &time[0..2], &time[2..4]))
}

/// When run `other` (of the same part as `run`) was made, from the stamp in its name.
pub fn run_time(run: &Path, other: &Path) -> Option<String> {
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = name(run).strip_suffix(".sorting.zarr")?.to_string();
    let other = name(other);
    stamp_of(other.strip_suffix(".sorting.zarr")?.strip_prefix(&stem)?)
}

/// `rec.bin` → `rec.<sorter>.sorting.zarr` next to it.
pub fn sorting_path(recording: &Path, sorter: &str) -> PathBuf {
    let name = recording.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    let stem = name.trim_end_matches(".zarr").trim_end_matches(".bin").trim_end_matches(".cbin").trim_end_matches(".nwb");
    recording.with_file_name(format!("{stem}.{sorter}.sorting.zarr"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_paths_and_describe() {
        assert_eq!(run_path(Path::new("/d/rec.bin"), "kilosort4", Some(10.0), Some(60.0)), PathBuf::from("/d/rec.kilosort4.10-70s.sorting.zarr"));
        assert_eq!(run_path(Path::new("/d/rec.bin"), "kilosort4", None, None), PathBuf::from("/d/rec.kilosort4.sorting.zarr"));
        let v = serde_json::json!({"interface": 1, "version": "0.0.5", "sorters": [{"name": "kilosort4", "about": "KS4", "parameters": [
            {"name": "th_universal", "flag": "--th-universal", "help": "h", "default": "9", "choices": [], "list": false},
            {"name": "mode", "flag": "--mode", "help": "", "default": null, "choices": ["a", "b"], "list": false}]}]});
        let d = parse_describe(PathBuf::from("p"), &v).unwrap();
        assert_eq!(d.sorters[0].params[0].default, "9");
        assert_eq!(d.sorters[0].params[1].choices, ["a", "b"]);
        assert!(parse_describe(PathBuf::from("p"), &serde_json::json!({"interface": 2})).is_err());
    }

    #[test]
    fn test_sorting_path() {
        assert_eq!(sorting_path(Path::new("/d/rec.bin"), "kilosort4"), PathBuf::from("/d/rec.kilosort4.sorting.zarr"));
        assert_eq!(sorting_path(Path::new("/d/rec.v2.processed.zarr"), "mountainsort5"), PathBuf::from("/d/rec.v2.processed.mountainsort5.sorting.zarr"));
    }

    /// A run's name carries when it started; runs of the same part are recognised, stamped or
    /// not, and say when they were made.
    #[test]
    fn runs_are_stamped_and_recognised() {
        let run = run_path(Path::new("/d/rec.bin"), "kilosort4", None, Some(20.0));
        assert_eq!(run, PathBuf::from("/d/rec.kilosort4.0-20s.sorting.zarr"));
        let stamped = stamped(&run, 1_794_947_555); // 2026-11-17T20:32:35Z
        assert_eq!(stamped, PathBuf::from("/d/rec.kilosort4.0-20s.20261117-2032.sorting.zarr"));
        assert!(same_run(&run, &stamped) && same_run(&run, &run) && same_run(&run, Path::new("/d/rec.kilosort4.0-20s-2.sorting.zarr")));
        assert!(!same_run(&run, Path::new("/d/rec.kilosort4.0-200s.sorting.zarr")), "another part");
        assert!(!same_run(&run, Path::new("/d/rec.mountainsort5.0-20s.sorting.zarr")));
        assert_eq!(run_time(&run, &stamped).as_deref(), Some("17 Nov 2026, 20:32 UTC"));
        assert_eq!(run_time(&run, &run), None);
    }
}
