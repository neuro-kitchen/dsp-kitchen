"""Runs the sorter program from the benchmarks, and reads what it writes.

The sorters are not in the `dspk` Python package (Kilosort4 and EMUsort are GPL-3.0): they are the
separate program `dspk-cli-sorter`. It is found through `DSPK_CLI_SORTER_BIN`, else `dspk sorters
install`'s folder, else `target/validate`. Timings mean something only with a release build (the
program of a GitHub release, or `cargo build --release -p dspk-cli-sorter`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import dspk

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
#: Host memory sampling period (s).
SAMPLE_SEC = 0.05
#: GPU memory sampling period (s): each reading runs `nvidia-smi`, so not at the host's rate.
GPU_SAMPLE_SEC = 0.5


def binary() -> Path:
    named = os.environ.get("DSPK_CLI_SORTER_BIN")
    if named:
        return Path(named)
    data = Path(os.environ.get("DSPK_DATA_DIR", Path.home() / ".local" / "share" / "dspk"))
    for candidate in (data / "sorters" / dspk.__version__ / "dspk-cli-sorter", ROOT / "target" / "validate" / "dspk-cli-sorter"):
        if candidate.exists():
            return candidate
    raise SystemExit("dspk-cli-sorter not found: `dspk sorters install`, or set DSPK_CLI_SORTER_BIN")


def program_version() -> str:
    return subprocess.run([str(binary()), "--version"], capture_output=True, text=True, check=True).stdout.strip()


def flags_from_argv(argv: list[str]) -> dict[str, str]:
    """`name=value` arguments as sorter settings (`global_merges=false` → `--global-merges false`)."""
    return dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))


def _rss_mib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            return next((int(line.split()[1]) / 1024 for line in f if line.startswith("VmRSS:")), 0.0)
    except OSError:
        return 0.0


def _gpu_used_mib() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return float(out.split()[0])


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or None


@dataclass
class Run:
    """One run of the program: wall time, time per stage (a stage lasts from its first progress
    report to the next stage's), peak host RSS, peak GPU memory above idle (NVIDIA only)."""

    wall_sec: float
    stages_sec: dict[str, float]
    peak_rss_mib: float
    peak_gpu_mib: float | None
    units: int
    spikes: int
    settings: dict[str, str] = field(default_factory=dict)
    exit_code: int = 0


def sort(sorter: str, recording, output, *, flags: dict[str, str] | None = None, source: str | None = None, probe: str | None = None,
         probe_file=None, pitch_um: float | None = None, duration_s: float | None = None, runtime: str | None = None, templates=None) -> Run:
    """Sorts `recording` into a new `.sorting.zarr` at `output` (an earlier one is replaced)."""
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary()), sorter, str(recording), "--output", str(output), "--format", "zarr", "--progress", "json"]
    for option, value in (("--source", source), ("--probe", probe), ("--probe-file", probe_file), ("--pitch-um", pitch_um), ("--duration-sec", duration_s), ("--templates", templates)):
        if value is not None:
            cmd += [option, str(value)]
    for name, value in (flags or {}).items():
        cmd += [f"--{name.replace('_', '-')}", value]
    env = dict(os.environ)
    if runtime:
        env["DSPK_RUNTIME"] = runtime
    idle = _gpu_used_mib()
    start = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    peak = {"rss": 0.0, "gpu": 0.0}

    def sample() -> None:
        next_gpu = 0.0
        while process.poll() is None:
            peak["rss"] = max(peak["rss"], _rss_mib(process.pid))
            if time.perf_counter() >= next_gpu:
                next_gpu = time.perf_counter() + GPU_SAMPLE_SEC
                used = _gpu_used_mib()
                if used is not None:
                    peak["gpu"] = max(peak["gpu"], used)
            time.sleep(SAMPLE_SEC)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    starts: dict[str, float] = {}
    done, error = None, None
    for line in process.stdout:
        if not line.startswith("{"):  # the sorters' own messages
            print(line, end="")
            continue
        event = json.loads(line)
        if event["event"] == "progress":
            if event["stage"] not in starts:
                starts[event["stage"]] = time.perf_counter()
                print(f"  [{starts[event['stage']] - start:7.1f} s] {event['stage']}", flush=True)
        elif event["event"] == "done":
            done = event
        elif event["event"] == "error":
            error = event["message"]
    code = process.wait()
    wall = time.perf_counter() - start
    sampler.join()
    if done is None:
        raise SystemExit(f"{sorter} failed (exit {code}): {error}")
    if code != 0:  # the sorting was written; recorded, so that an abort at exit is not lost
        print(f"  warning: {sorter} finished and wrote its output, then exited with {code}", flush=True)
    order = sorted(starts, key=starts.get)
    stages = {s: e - starts[s] for s, e in zip(order, [starts[s] for s in order[1:]] + [start + wall])}
    if order:
        stages[order[0]] += starts[order[0]] - start
    return Run(wall, stages, peak["rss"], peak["gpu"] - idle if idle is not None else None, int(done["units"]), int(done["spikes"]), dict(flags or {}), code)


def load(path) -> dspk.synapse.io.Sorting:
    return dspk.synapse.io.Sorting.load(path)


def flatten(sorting) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All spikes of a sorting as (samples, unit ids, locations `[n, 2]` µm), in time order."""
    spikes = sorting.spikes()
    return np.asarray(spikes["samples"], dtype=np.int64), np.asarray(spikes["units"]), np.asarray(spikes["locations"], dtype=np.float64)[:, :2]


def identical(a, b) -> bool:
    """Whether two sortings have the same spikes and units."""
    (ta, ua, _), (tb, ub, _) = flatten(a), flatten(b)
    return bool(np.array_equal(ta, tb) and np.array_equal(ua, ub))


def save_json(name: str, record: dict) -> Path:
    record = {"date": time.strftime("%Y-%m-%d"), "program": program_version(), "machine": _machine(), **record}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"saved {path}")
    return path


def _machine() -> str:
    from plating.theme import machine

    return machine()
