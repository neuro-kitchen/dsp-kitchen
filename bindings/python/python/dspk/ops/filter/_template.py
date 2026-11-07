"""Template subtraction on the device."""

from __future__ import annotations

import numpy as np

from dspk.ops.filter._common import _channels, _filter

TEMPLATE_DEFAULT_MAX_LAG: int = _filter.TEMPLATE_DEFAULT_MAX_LAG


def subtract_template(
    data,
    template,
    events,
    center_offset: int,
    *,
    max_lag: int = TEMPLATE_DEFAULT_MAX_LAG,
    dynamic_scaling: bool = True,
    runtime: str | None = None,
) -> np.ndarray:
    """Subtracts a template from ``data`` at every event, each aligned and scaled.

    The template's sample ``center_offset`` is placed on each event, then shifted by up to
    ``max_lag`` samples to where it best matches (cross-correlation) and, with
    ``dynamic_scaling``, scaled by least squares before it is subtracted.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    template : array_like
        The waveform ``(channels, template_samples)``, or 1-D for one channel.
    events : array_like
        Sample indices of the events (1-D).
    center_offset : int
        The template sample that sits on each event.
    max_lag : int
        How far, in samples, an event may shift to align.
    dynamic_scaling : bool
        Scale the template to each event (least squares); ``False`` subtracts it as it is.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``data`` with the events removed ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        When ``data`` is not 2-D.
    RuntimeError
        When the runtime has no device.

    Notes
    -----
    Overlapping events are subtracted in order, each aligned on what the earlier ones left, so
    where waveforms overlap a remainder can stay (about 10 % for two events 5 samples apart in
    the tests). Isolated events are removed to well under 1 %.
    """
    template = np.atleast_2d(np.ascontiguousarray(template, dtype=np.float32))
    events = np.ascontiguousarray(events, dtype=np.uint64).ravel()
    return _filter.subtract_template(
        _channels(data), template, events, center_offset,
        max_lag=max_lag, dynamic_scaling=dynamic_scaling, runtime=runtime,
    )


__all__ = ["TEMPLATE_DEFAULT_MAX_LAG", "subtract_template"]
