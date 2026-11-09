"""dspk: multi-channel signal processing on any GPU or the CPU, written once in CubeCL.

Modules mirror the Rust crates: ``dspk.runtime`` chooses the compute runtime, ``dspk.io`` reads and
writes recordings, ``dspk.ops`` runs signal-processing operations on the device, ``dspk.pipeline``
runs pipelines described as data (JSON / TOML), ``dspk.synapse`` adds electrophysiology (probes,
sortings, unit metrics; ``pip install dspk[synapse]``), ``dspk.custom`` runs your own functions over
recordings window by window, ``dspk.app`` opens the desktop app (``pip install dspk[app]``).
"""

import importlib
import importlib.util
import sys

# `dspk[synapse]` installs `dspk-synapse`: this package's extension built with Synapse in. Only one
# extension is ever loaded, so there is one GPU device per runtime and one choice of runtime.
if importlib.util.find_spec("dspk_synapse") is not None:
    sys.modules["dspk._native"] = importlib.import_module("dspk_synapse._native")

from dspk import app, custom, io, ops, pipeline, remote, runtime  # noqa: E402
from dspk._native import __version__  # noqa: E402

__all__ = ["__version__", "app", "custom", "io", "ops", "pipeline", "remote", "runtime"]
if hasattr(sys.modules["dspk._native"], "synapse"):
    __all__.append("synapse")


def __getattr__(name: str):
    if name == "synapse":
        return importlib.import_module("dspk.synapse")
    raise AttributeError(f"module 'dspk' has no attribute {name!r}")
