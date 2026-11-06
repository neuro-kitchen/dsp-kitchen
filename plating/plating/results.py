"""The benchmark record: one JSON file per run of a benchmark, committed, so results can be
inspected and compared across versions and machines. Plots are drawn from these files
(:func:`plating.figures.timings`, :func:`plating.figures.agreement`).

Schema (``SCHEMA_VERSION`` 1)::

    {"schema": 1,
     "benchmark": "signal-operations/filters",
     "date": "2026-11-06",
     "dspk": {"version": "0.0.6", "commit": "e577a8e", "runtime": "cuda"},
     "reference": {"tool": "scipy", "version": "1.15.3"},
     "machine": "<GPU> · <CPU> (<n> threads) · <OS>",
     "notes": "what was measured, how (one paragraph)",
     "cases": [{"name": "bandpass_zero_phase", "shape": [384, 60000], "dtype": "float32",
                "dspk_ms": 1.9, "reference_ms": 410.0, "max_abs_err": 3e-6, "rel_err": 1e-7,
                "tolerance": 1e-5, "pass": true, "extra": {}}]}

``reference`` is ``null`` when nothing is compared (timings only); a case's comparison fields are
``null`` then. ``pass`` is ``rel_err <= tolerance`` when both are given.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plating.theme import machine

SCHEMA_VERSION = 1


@dataclass
class Case:
    name: str
    shape: list[int] = field(default_factory=list)
    dtype: str = "float32"
    dspk_ms: float | None = None
    reference_ms: float | None = None
    max_abs_err: float | None = None
    rel_err: float | None = None
    tolerance: float | None = None
    passed: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed is None and self.rel_err is not None and self.tolerance is not None:
            self.passed = bool(self.rel_err <= self.tolerance)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("passed")
        return d


def compare(name: str, got, expected, *, tolerance: float, dspk_ms: float | None = None, reference_ms: float | None = None, **extra) -> Case:
    """A case from two arrays: max absolute error, and that relative to ``expected``'s peak."""
    import numpy as np

    got, expected = np.asarray(got, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if got.shape != expected.shape:
        raise ValueError(f"{name}: shapes differ, {got.shape} vs {expected.shape}")
    err = float(np.max(np.abs(got - expected))) if got.size else 0.0
    peak = float(np.max(np.abs(expected))) if expected.size else 0.0
    return Case(name, list(got.shape), "float32", dspk_ms, reference_ms, err, err / peak if peak else err, tolerance, extra=extra)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def write(path: str | Path, benchmark: str, cases: list[Case], *, reference: dict[str, str] | None = None, runtime: str | None = None, notes: str = "") -> Path:
    """Writes a run's record (dspk's version, the commit, the machine and the date filled in)."""
    try:
        import dspk

        version, runtime = dspk.__version__, runtime or dspk.runtime.current()
    except Exception:  # noqa: BLE001
        version = ""
    record = {
        "schema": SCHEMA_VERSION,
        "benchmark": benchmark,
        "date": _dt.date.today().isoformat(),
        "dspk": {"version": version, "commit": _commit(), "runtime": runtime or ""},
        "reference": reference,
        "machine": machine(),
        "notes": notes,
        "cases": [c.to_json() for c in cases],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


def read(path: str | Path) -> dict[str, Any]:
    """Reads a record (checks the schema version)."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {record.get('schema')!r}, expected {SCHEMA_VERSION}")
    return record


__all__ = ["SCHEMA_VERSION", "Case", "compare", "read", "write"]
