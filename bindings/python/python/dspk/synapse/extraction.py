"""Snippets: the waveform around each spike, on its nearest channels."""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse
from dspk.synapse.detection import DEFAULT_K_NEIGHBORS, DEFAULT_POST_MS, DEFAULT_PRE_MS
from dspk.synapse.io import Probe

_x = _synapse.extraction


def snippets(data, spikes: dict, probe: Probe, sample_rate: float, *, k: int = DEFAULT_K_NEIGHBORS, pre_ms: float = DEFAULT_PRE_MS,
             post_ms: float = DEFAULT_POST_MS, sinc_shift: bool = False) -> dict:
    """The waveform around each spike, on its nearest channels.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    spikes : dict of numpy.ndarray
        ``samples`` and ``channels``, as the detectors return.
    probe : Probe
        The probe, for each spike's nearest channels.
    sample_rate : float
        Sample rate in Hz.
    k : int
        Channels per snippet.
    pre_ms, post_ms : float
        The span ``[t - pre_ms, t + post_ms)`` around each spike.
    sinc_shift : bool
        Realign each snippet to its sub-sample trough.

    Returns
    -------
    dict of numpy.ndarray
        ``waveforms`` ``(spikes, k, samples)``, ``channels`` ``(spikes, k)``, ``centers``,
        ``offsets``, and ``kept``: the input index of each snippet (spikes too near an edge are
        skipped).

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32) * 10
    >>> x[1, 15_000:15_010] -= 120  # one spike
    >>> spikes = dspk.synapse.detection.threshold(x, 30_000.0)
    >>> probe = dspk.synapse.io.Probe.preset("tetrode")
    >>> snips = dspk.synapse.extraction.snippets(x, spikes, probe, 30_000.0, k=4)
    >>> snips["waveforms"].shape[1:]
    (4, 90)
    """
    pre, post = round(pre_ms * sample_rate / 1e3), round(post_ms * sample_rate / 1e3)
    return _x.snippets(np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32), np.ascontiguousarray(spikes["samples"], dtype=np.uint64),
                       np.ascontiguousarray(spikes["channels"], dtype=np.uint32), probe._native, k, pre, post, sinc_shift)


__all__ = ["snippets"]
