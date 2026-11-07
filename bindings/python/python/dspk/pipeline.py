"""Pipelines as data: a versioned spec of nodes and edges, checked and run by dspk-pipeline.

A spec is a dict, or a ``.json`` / ``.toml`` file. One spec can run over many inputs (globs,
lists) and read several recordings as one (merges). The same spec in each format:

JSON::

    {
      "dspk_pipeline": 1,
      "nodes": {
        "raw": {"op": "read", "path": "recording.bin"},
        "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
        "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"}
      },
      "edges": [["raw", "hp"], ["hp", "out"]]
    }

TOML::

    dspk_pipeline = 1
    edges = [["raw", "hp"], ["hp", "out"]]

    [nodes.raw]
    op = "read"
    path = "recording.bin"

    [nodes.hp]
    op = "iir"
    band = "highpass"
    cutoff = [300.0]
    mode = "forward"

    [nodes.out]
    op = "write"
    path = "filtered.zarr"
    format = "zarr"

Processing nodes mirror :mod:`dspk.ops.stage`. A run streams the recording window by window
through the device (one upload, one download per window) and writes the canonical spec next to
the output (``<output>.pipeline.json``). Without ``execution.runtime`` the spec runs on the
runtime :mod:`dspk.runtime` selects.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Callable

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
    """Checks a spec and returns its canonical form, every default written out.

    Parameters
    ----------
    spec : dict, str or os.PathLike
        A spec dict, or the path of a ``.json`` / ``.toml`` spec file.

    Returns
    -------
    dict
        The canonical spec.

    Raises
    ------
    ValueError
        With the reason, when the spec is invalid.

    Examples
    --------
    >>> import dspk
    >>> spec = {
    ...     "dspk_pipeline": 1,
    ...     "nodes": {
    ...         "raw": {"op": "read", "path": "in.bin"},
    ...         "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0]},
    ...         "out": {"op": "write", "path": "out.zarr", "format": "zarr"},
    ...     },
    ...     "edges": [["raw", "hp"], ["hp", "out"]],
    ... }
    >>> full = dspk.pipeline.canonical(spec)
    """
    text, toml, _ = _text(spec)
    return json.loads(_pipeline.canonical(text, toml=toml))


def load(path: str | PathLike) -> Spec:
    """Reads a spec file and returns its canonical form.

    Parameters
    ----------
    path : str or os.PathLike
        A ``.json`` or ``.toml`` spec.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        When the spec is invalid.
    OSError
        When the file cannot be read.
    """
    return canonical(Path(path))


def plan(spec: Spec | str | PathLike, base_dir: str | PathLike | None = None) -> list[dict[str, Any]]:
    """What :func:`run` would do for each input, without touching the device (a dry run).

    Parameters
    ----------
    spec : dict, str or os.PathLike
        A spec dict, or the path of a ``.json`` / ``.toml`` spec file.
    base_dir : str or os.PathLike, optional
        Where relative paths resolve; by default the spec file's folder, or the working directory
        for a dict.

    Returns
    -------
    list of dict
        One plan per input: ``input``, ``output``, ``mode``, ``halo``, ``stored_upload``,
        ``channels``, input and output samples and rates, ``windows``, ``max_window_samples`` and
        ``bytes`` (read, uploaded).

    Raises
    ------
    ValueError
        For anything :func:`run` would refuse.
    """
    text, toml, folder = _text(spec)
    return _pipeline.plan(text, Path(base_dir) if base_dir is not None else folder, toml=toml)


Progress = Callable[[int, int, int, int], "bool | None"]


def run(
    spec: Spec | str | PathLike,
    base_dir: str | PathLike | None = None,
    *,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """Runs a spec once per input, window by window through the device.

    Inputs are ``read.path`` (a file or a glob such as ``"sessions/*.bin"``) or ``read.inputs``
    (paths, globs and ``{"merge": [...]}`` items, which read several recordings side by side). With
    several inputs, the write path needs ``{stem}`` or ``{index}``. The canonical spec is written
    next to each output (``<output>.pipeline.json``).

    Parameters
    ----------
    spec : dict, str or os.PathLike
        A spec dict, or the path of a ``.json`` / ``.toml`` spec file.
    base_dir : str or os.PathLike, optional
        Where relative paths resolve; by default the spec file's folder, or the working directory
        for a dict.
    progress : callable, optional
        ``progress(input, inputs, windows_done, windows)``, called after each window; returning
        ``False`` cancels.

    Returns
    -------
    list of dict
        One report per input: ``mode``, ``halo``, ``runtime``, ``stored_upload``, ``channels``,
        ``samples``, ``sample_rate``, ``windows``, ``output``, ``provenance``, ``seconds``,
        ``bytes`` (read, uploaded, downloaded, written) and ``timings``.

    Raises
    ------
    ValueError
        When the spec is invalid.
    OSError
        When an input cannot be read or an output written.
    KeyboardInterrupt
        When ``progress`` returns ``False`` or on Ctrl+C. The unfinished output is removed;
        finished inputs stay.

    Notes
    -----
    ``timings`` are seconds spent in ``read_wait``, ``device``, ``write_wait``, ``encode`` and
    ``disk``. Reading, the device and the writer run at the same time, so the largest of
    ``read_wait``, ``device`` and ``encode + disk`` is what limits the run.
    """
    text, toml, folder = _text(spec)
    return _pipeline.run(text, Path(base_dir) if base_dir is not None else folder, toml=toml, progress=progress)


def schema() -> dict[str, Any]:
    """The JSON Schema of a spec, for editors and validators.

    Returns
    -------
    dict
    """
    return json.loads(_pipeline.schema())


__all__ = ["DEFAULT_CHUNK_SAMPLES", "SPEC_VERSION", "Progress", "Spec", "canonical", "load", "plan", "run", "schema"]
