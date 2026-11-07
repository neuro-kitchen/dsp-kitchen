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
from dspk.ops import filter, layout, linalg, math, peaks, resample, spatial, stage


class Pipeline:
    """Stages run on the device over chunks of a signal: one upload, every stage, one download.

    Filter designs and device buffers are kept between chunks.

    Parameters
    ----------
    stages : sequence of Stage
        The stages, in order (:mod:`dspk.ops.stage`).
    sample_rate : float
        Sample rate of the data, in Hz.
    stream : bool
        ``False``: every chunk on its own (zero-phase filters allowed; give each chunk
        :attr:`settling` of context). ``True``: chunks are consecutive pieces of one signal and
        filter state carries over, so chunked output equals whole-signal output (forward filters
        only).
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Raises
    ------
    ValueError
        When a stage is invalid at ``sample_rate``, or ``stream=True`` has a zero-phase filter.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> chain = dspk.ops.Pipeline(
    ...     [dspk.ops.stage.iir("highpass", 300.0), dspk.ops.stage.cmr()], 30_000.0
    ... )
    >>> chain.process(x).shape
    (4, 30000)
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
        """Runs every stage over a chunk.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
            device as they are and are converted there; other dtypes and lists become float32 first.

        Returns
        -------
        numpy.ndarray
            ``(channels, samples)``, float32.

        Raises
        ------
        ValueError
            When a spatial stage was made for another channel count.
        RuntimeError
            When the runtime has no device.
        """
        return self._native.process(_channels(data))

    def reset(self) -> None:
        """Forgets the device workspace: the next chunk starts a new stream.
        """
        self._native.reset()

    @property
    def settling(self) -> tuple[int, int]:
        """Context ``(left, right)`` in samples a chunk needs for its output to match the whole signal's.
        """
        return self._native.settling

    @property
    def sample_rate(self) -> float:
        """Sample rate of the data, in Hz.
        """
        return self._native.sample_rate

    @property
    def stream(self) -> bool:
        """Whether chunks are consecutive pieces of one signal.
        """
        return self._native.stream

    @property
    def runtime(self) -> str:
        """The runtime the stages run on.
        """
        return self._native.runtime

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


__all__ = ["Pipeline", "filter", "layout", "linalg", "math", "peaks", "resample", "spatial", "stage"]
