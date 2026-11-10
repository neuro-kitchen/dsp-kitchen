"""Threshold crossings as events, on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_crossings = _ops.crossings


def detect(
    data,
    level,
    *,
    release=None,
    direction: str = "rising",
    min_duration_samples: int = 0,
    refractory_samples: int = 0,
    runtime: str | None = None,
) -> dict[str, np.ndarray]:
    """Events where each channel goes past a level and comes back.

    An event starts (``onset``) at the first sample at or past ``level`` and ends (``offset``) at
    the first sample back past ``release``. With ``release`` equal to ``level`` (the default) this
    is a plain threshold crossing; a release between the baseline and the level keeps noise around
    the level from splitting one event into many. While an event is open, further crossings belong
    to it; a new event needs a fresh crossing.

    One tool for any signal: e.g. EMG contractions on an envelope (band-pass, rectify, low-pass,
    then ``detect``), deflections either way on a raw trace (``direction="both"``), a trigger line.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    level : float or array_like
        Level per channel ``(channels,)``, or one value for all, in the signal's unit.
    release : float or array_like, optional
        Level that ends an event; default: ``level``. ``"rising"``: at most the level;
        ``"falling"``: at least the level; ``"both"``: a magnitude at most the level's.
    direction : {"rising", "falling", "both"}
        ``"rising"``: up through ``level``. ``"falling"``: down through ``level``. ``"both"``:
        past ``+|level|`` upwards or ``-|level|`` downwards (an event's sign is its peak's).
    min_duration_samples : int
        Events shorter than this are dropped (default 0: none).
    refractory_samples : int
        Onsets within this many samples after an event's end are ignored (default 0).
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    dict of numpy.ndarray
        ``"channel"``, ``"onset"``, ``"offset"`` (``-1`` for an event still open at the end),
        ``"peak_sample"`` and ``"peak_value"``, by onset then channel. ``polars.DataFrame(result)``
        gives the table.

    Raises
    ------
    ValueError
        For an unknown ``direction``, a release on the wrong side of the level, levels of another
        length than the channels, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> t = np.arange(20_000) / 2_000
    >>> envelope = (np.sin(2 * np.pi * 0.5 * t) > 0.6).astype(np.float32)[None, :]
    >>> events = dspk.ops.crossings.detect(envelope, 0.5, release=0.25, refractory_samples=200)
    >>> int(events["onset"][0])
    410
    """
    x = _channels(data)
    levels = np.broadcast_to(np.asarray(level, dtype=np.float32), (x.shape[0],)).copy()
    releases = levels if release is None else np.broadcast_to(np.asarray(release, dtype=np.float32), (x.shape[0],)).copy()
    channel, onset, offset, peak_sample, peak_value = _crossings.detect(
        x,
        levels,
        releases,
        direction=direction,
        min_duration_samples=int(min_duration_samples),
        refractory_samples=int(refractory_samples),
        runtime=runtime,
    )
    return {"channel": channel, "onset": onset, "offset": offset, "peak_sample": peak_sample, "peak_value": peak_value}


__all__ = ["detect"]
