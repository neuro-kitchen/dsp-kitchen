//! The JSON record of a kernel benchmark, in the schema of `plating.results` (schema 1): one file
//! per benchmark and device, `results/<benchmark>-<device>.json`, committed so runs can be compared
//! across versions and machines (`plating.figures.timings` plots it).
//!
//! The benchmarks themselves are ignored tests under `tests/`:
//!
//! ```text
//! cargo test -p dspk-benchmarks --profile validate --test bench_kernels -- --ignored --nocapture
//! DSPK_TEST_RUNTIMES=cuda cargo test -p dspk-benchmarks ...   # one runtime
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};

/// Schema version of the record (as `plating.results.SCHEMA_VERSION`).
pub const SCHEMA_VERSION: u32 = 1;

/// The cases of one benchmark on one device, written by [`Report::write`].
pub struct Report {
    benchmark: String,
    device: String,
    notes: String,
    cases: Vec<Value>,
}

impl Report {
    /// `benchmark` names it (`kernels/bandwidth`); `device` is the runtime or device it ran on.
    pub fn new(benchmark: &str, device: &str, notes: &str) -> Self {
        Self { benchmark: benchmark.into(), device: device.into(), notes: notes.into(), cases: Vec::new() }
    }

    /// One timed case: its `name`, the data `shape`, the median time, and any measures derived from
    /// it (`("gb_per_s", 410.0)`, …).
    pub fn case(&mut self, name: &str, shape: &[usize], ms: f64, extra: &[(&str, f64)]) {
        let extra: Map<String, Value> = extra.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        self.cases.push(json!({
            "name": name, "shape": shape, "dtype": "float32", "dspk_ms": ms, "reference_ms": null,
            "max_abs_err": null, "rel_err": null, "tolerance": null, "pass": null, "extra": extra,
        }));
    }

    /// Writes `<dir>/<benchmark with '/' as '-'>-<device>.json`; returns its path.
    pub fn write_to(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let record = json!({
            "schema": SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "date": today(),
            "dspk": {"version": env!("CARGO_PKG_VERSION"), "commit": git(&["rev-parse", "--short", "HEAD"]), "runtime": self.device},
            "reference": null,
            "machine": machine(),
            "notes": self.notes,
            "cases": self.cases,
        });
        let file = format!("{}-{}.json", self.benchmark.replace('/', "-"), self.device.replace([' ', '/'], "_"));
        let path = dir.join(file);
        std::fs::write(&path, serde_json::to_string_pretty(&record).expect("a record serialises") + "\n")?;
        println!("record: {}", path.display());
        Ok(path)
    }

    /// [`Self::write_to`] this crate's `results/`.
    pub fn write(&self) -> std::io::Result<PathBuf> {
        self.write_to(&Path::new(env!("CARGO_MANIFEST_DIR")).join("results"))
    }
}

fn git(args: &[&str]) -> String {
    Command::new("git").args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn today() -> String {
    Command::new("date").arg("+%F").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

/// `"<GPU> · <CPU> (<n> threads) · <OS>"`, as `plating.theme.machine()`.
pub fn machine() -> String {
    let gpu = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).lines().next().map(str::to_string))
        .unwrap_or_default();
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let threads = std::thread::available_parallelism().map_or(0, |n| n.get());
    let os = format!("{} {}", std::env::consts::OS, git_free_uname());
    [gpu, format!("{cpu} ({threads} threads)"), os].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · ")
}

fn git_free_uname() -> String {
    Command::new("uname").arg("-r").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_written_in_the_schema() {
        let dir = std::env::temp_dir().join(format!("dspk-bench-report-{}", std::process::id()));
        let mut r = Report::new("test/one", "cpu", "a test");
        r.case("scale", &[4, 10], 0.5, &[("gb_per_s", 2.0)]);
        let path = r.write_to(&dir).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["schema"], 1);
        assert_eq!(v["cases"][0]["extra"]["gb_per_s"], 2.0);
        assert_eq!(path.file_name().unwrap(), "test-one-cpu.json");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
