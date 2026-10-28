"""dspk: multi-channel signal processing on any GPU or the CPU, written once in CubeCL.

Modules mirror the Rust crates: ``dspk.runtime`` chooses the compute runtime; ``dspk.io`` and
``dspk.ops`` follow.
"""

from dspk import runtime
from dspk._native import __version__

__all__ = ["__version__", "runtime"]
