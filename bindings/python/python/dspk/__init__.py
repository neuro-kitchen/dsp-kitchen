"""dspk: multi-channel signal processing on any GPU or the CPU, written once in CubeCL.

Modules mirror the Rust crates: ``dspk.runtime`` chooses the compute runtime, ``dspk.io`` reads and
writes recordings, ``dspk.ops`` runs signal-processing operations on the device, ``dspk.pipeline``
runs pipelines described as data (JSON / TOML), ``dspk.synapse`` adds electrophysiology (probes,
sortings, unit metrics), ``dspk.custom`` runs your own functions over recordings window by
window.
"""

from dspk import custom, io, ops, pipeline, remote, runtime, synapse
from dspk._native import __version__

__all__ = ["__version__", "custom", "io", "ops", "pipeline", "remote", "runtime", "synapse"]
