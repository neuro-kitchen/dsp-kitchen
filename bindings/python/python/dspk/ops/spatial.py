"""Operations across channels on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_spatial = _ops.spatial


def car(data, *, runtime: str | None = None) -> np.ndarray:
    """Common average reference: the mean over channels subtracted at every sample."""
    return _spatial.car(_channels(data), runtime=runtime)


def cmr(data, *, runtime: str | None = None) -> np.ndarray:
    """Common median reference: the median over channels subtracted at every sample."""
    return _spatial.cmr(_channels(data), runtime=runtime)


__all__ = ["car", "cmr"]
