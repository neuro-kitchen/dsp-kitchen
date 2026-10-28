"""Pipelines as data: a versioned node-graph spec, checked and run by dspk-pipeline.

A spec is a dict (or a ``.json`` / ``.toml`` file); one spec can run over many inputs (globs,
lists) and read several recordings as one (merges)::

    {"dspk_pipeline": 1,
     "nodes": {"raw": {"op": "read", "path": "recording.bin"},
               "hp":  {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
               "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"}},
     "edges": [["raw", "hp"], ["hp", "out"]]}

Processing nodes mirror :mod:`dspk.ops.stage`. The run streams the recording window by window
through the device (one upload, one download per window) and writes the canonical spec next to the
output (``<output>.pipeline.json``). Without ``execution.runtime`` the spec runs on the runtime
:mod:`dspk.runtime` selects.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

from dspk._native import pipeline as _pipeline

SPEC_VERSION: int = _pipeline.SPEC_VERSION
DEFAULT_CHUNK_SAMPLES: int = _pipeline.DEFAULT_CHUNK_SAMPLES

Spec = dict[str, Any]


def _text(spec: Spec | str | PathLike) -> tuple[str, bool, Path]:
    """Spec text, whether it is TOML, and the folder relative paths resolve against."""
    if isinstance(spec, dict):
        return json.dumps(spec), False, Path.cwd()
    path = Path(spec)
    return path.read_text(), path.suffix == ".toml", path.resolve().parent


def canonical(spec: Spec | str | PathLike) -> Spec:
    """Checks ``spec`` and returns its canonical form (every default written out).

    Raises ``ValueError`` with the reason when the spec is invalid.
    """
    text, toml, _ = _text(spec)
    return json.loads(_pipeline.canonical(text, toml=toml))


def load(path: str | PathLike) -> Spec:
    """Reads a ``.json`` or ``.toml`` spec file and returns its canonical form."""
    return canonical(Path(path))


def run(spec: Spec | str | PathLike, base_dir: str | PathLike | None = None) -> list[dict[str, Any]]:
    """Runs ``spec`` once per input; relative paths resolve against ``base_dir`` (default: the spec
    file's folder, or the working directory for a dict).

    Inputs: ``read.path`` (a file or a glob such as ``"sessions/*.bin"``) or ``read.inputs`` (paths,
    globs and ``{"merge": [...]}`` items, a merge reading several recordings side by side). With
    several inputs the write path needs ``{stem}`` or ``{index}``.

    Returns one report per input: mode, halo, runtime, stored_upload, channels, samples,
    sample_rate, windows, output, provenance, seconds, and ``bytes`` (read from the input,
    uploaded, downloaded, written)."""
    text, toml, folder = _text(spec)
    return _pipeline.run(text, Path(base_dir) if base_dir is not None else folder, toml=toml)


def schema() -> dict[str, Any]:
    """JSON Schema of the spec (for editors and validators)."""
    return json.loads(_pipeline.schema())


__all__ = ["DEFAULT_CHUNK_SAMPLES", "SPEC_VERSION", "Spec", "canonical", "load", "run", "schema"]
