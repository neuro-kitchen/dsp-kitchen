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
    """Subtracts ``template`` ``(channels, template_samples)`` (or 1-D for one channel) from
    ``data`` ``(channels, samples)`` at every event (sample indices).

    The template's sample ``center_offset`` sits on the event; each event is aligned by
    cross-correlation within ``±max_lag`` samples and, with ``dynamic_scaling``, scaled by least
    squares. Overlapping events are subtracted in their order: each alignment sees the residual of
    the earlier ones, so where waveforms overlap a remainder can stay (about 10 % for two events 5
    samples apart in the tests); isolated events are removed to well under 1 %.
    """
    template = np.atleast_2d(np.ascontiguousarray(template, dtype=np.float32))
    events = np.ascontiguousarray(events, dtype=np.uint64).ravel()
    return _filter.subtract_template(
        _channels(data), template, events, center_offset,
        max_lag=max_lag, dynamic_scaling=dynamic_scaling, runtime=runtime,
    )


__all__ = ["TEMPLATE_DEFAULT_MAX_LAG", "subtract_template"]
