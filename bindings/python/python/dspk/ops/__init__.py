"""Signal-processing operations of dspk-ops, run on the chosen runtime (see ``dspk.runtime``).

Each call uploads its input, runs on the device and returns NumPy.
"""

from dspk.ops import filter

__all__ = ["filter"]
