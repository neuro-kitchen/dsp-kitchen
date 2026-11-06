"""Snippets: the waveform around each spike, on its nearest channels."""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse
from dspk.synapse.detection import DEFAULT_K_NEIGHBORS, DEFAULT_POST_MS, DEFAULT_PRE_MS
from dspk.synapse.io import Probe

_x = _synapse.extraction


def snippets(data, spikes: dict, probe: Probe, sample_rate: float, *, k: int = DEFAULT_K_NEIGHBORS, pre_ms: float = DEFAULT_PRE_MS,
             post_ms: float = DEFAULT_POST_MS, sinc_shift: bool = False) -> dict:
    """``[t − pre_ms, t + post_ms)`` around each spike of ``spikes`` (``samples``, ``channels``) on
    its ``k`` nearest channels, from ``data`` ``(channels, samples)``; ``sinc_shift`` realigns each
    to its sub-sample trough. Spikes too near an edge are skipped (``kept``: the input indices).
    ``waveforms`` ``(spikes, k, samples)``, ``channels`` ``(spikes, k)``, ``centers``, ``offsets``."""
    pre, post = round(pre_ms * sample_rate / 1e3), round(post_ms * sample_rate / 1e3)
    return _x.snippets(np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32), np.ascontiguousarray(spikes["samples"], dtype=np.uint64),
                       np.ascontiguousarray(spikes["channels"], dtype=np.uint32), probe._native, k, pre, post, sinc_shift)


__all__ = ["snippets"]
