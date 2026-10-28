"""dspk: multi-channel signal processing on any GPU or the CPU, written once in CubeCL.

Modules mirror the Rust crates: ``dspk.runtime`` chooses the compute runtime, ``dspk.io`` reads and
writes recordings, ``dspk.ops`` runs signal-processing operations on the device.
"""

from dspk import io, ops, runtime
from dspk._native import __version__

__all__ = ["__version__", "io", "ops", "runtime"]
