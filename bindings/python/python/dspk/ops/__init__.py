"""Signal-processing operations of dspk-ops, run on the chosen runtime (see ``dspk.runtime``).

One-shot functions (``dspk.ops.filter``) upload their input, run and return NumPy. A
:class:`Pipeline` chains :mod:`dspk.ops.stage` stages on the device: one upload, every stage, one
download per chunk, with filter designs and device buffers kept between chunks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels
from dspk.ops import filter, linalg, math, peaks, resample, spatial, stage


class Pipeline:
    """Stages run on the device over ``(channels, samples)`` chunks.

    ``stream=False``: every chunk on its own (zero-phase filters allowed; give chunks the context of
    :attr:`settling`). ``stream=True``: chunks are consecutive pieces of one signal; filter state
    carries over, so chunked output equals whole-signal output (forward filters only).
    """

    def __init__(
        self,
        stages: Sequence[stage.Stage],
        sample_rate: float,
        *,
        stream: bool = False,
        runtime: str | None = None,
    ) -> None:
        self._native = _ops.Pipeline(list(stages), sample_rate, stream=stream, runtime=runtime)

    def process(self, data) -> np.ndarray:
        """Runs every stage over ``data`` ``(channels, samples)``; returns float32 of the same shape.
        Integer arrays cross to the device as their integers and are converted there."""
        return self._native.process(_channels(data))

    def reset(self) -> None:
        """Forgets the device workspace: the next chunk starts a new stream."""
        self._native.reset()

    @property
    def settling(self) -> tuple[int, int]:
        """Context ``(left, right)`` in samples a chunk needs to match whole-signal output."""
        return self._native.settling

    @property
    def sample_rate(self) -> float:
        return self._native.sample_rate

    @property
    def stream(self) -> bool:
        return self._native.stream

    @property
    def runtime(self) -> str:
        return self._native.runtime

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


__all__ = ["Pipeline", "filter", "linalg", "math", "peaks", "resample", "spatial", "stage"]
