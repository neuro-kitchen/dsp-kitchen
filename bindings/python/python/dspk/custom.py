"""Custom operations: your own Python functions over a recording, window by window.

dspk reads the recording in windows (the next one is read on another thread while your function
runs), so a recording of any length goes through a function that only ever sees a window:

- :func:`measure` calls ``fn(x, w)`` on each window and gathers what it returns into a table:
  one row per window (band power per channel, a count, a dict of several values).
- :func:`transform` wraps ``fn(x)`` (or ``fn(x, w)``) that returns the window processed, same
  shape. Windows are read with ``halo`` samples of context on each side and the result is trimmed
  back, so the windowed output equals ``fn`` on the whole array for any ``fn`` that looks at most
  ``halo`` samples away (a moving average of ``2 · halo + 1`` samples, a filter whose response has
  died out within ``halo``). :meth:`Transform.apply` returns an array; :meth:`Transform.write`
  writes the whole recording to Zarr without holding it in memory.

``x`` is ``(channels, samples)`` float32 NumPy in each channel's unit, as
:meth:`dspk.io.Recording.read` gives it; ``w`` is the :class:`Window` it came from.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike

import numpy as np

from dspk import io

#: Seconds per window when none is given (memory only: results do not depend on it).
DEFAULT_WINDOW_S: float = 1.0


@dataclass(frozen=True)
class Window:
    """Where a window's samples come from. ``start``–``stop``: its own samples (recording
    samples); ``read_start``–``read_stop``: what was read, its context included; ``inner``: the
    slice of the read array that holds its own samples; ``index``: its number (from 0)."""

    index: int
    start: int
    stop: int
    read_start: int
    read_stop: int
    sample_rate: float
    channels: tuple[int, ...]

    @property
    def inner(self) -> slice:
        return slice(self.start - self.read_start, self.stop - self.read_start)

    @property
    def start_s(self) -> float:
        return self.start / self.sample_rate

    @property
    def stop_s(self) -> float:
        return self.stop / self.sample_rate


def _span(rec, start: int, stop: int | None) -> tuple[int, int]:
    stop = rec.samples if stop is None else stop
    if not 0 <= start < stop <= rec.samples:
        raise ValueError(f"samples {start}..{stop} are not within the recording's 0..{rec.samples}")
    return start, stop


def _window_samples(rec, window_s: float) -> int:
    if not window_s > 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    return max(1, round(window_s * rec.sample_rate))


def windows(
    rec,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    halo: int = 0,
    channels: Sequence[int] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[tuple[np.ndarray, Window]]:
    """The windows of ``rec`` (``start``–``stop``, ``channels``; all by default), each read with
    ``halo`` samples of context on each side where the recording has them: yields ``(x, w)``. The
    next window is read on another thread while the caller works on this one."""
    if halo < 0:
        raise ValueError(f"halo must be at least 0, got {halo}")
    start, stop = _span(rec, start, stop)
    chans = tuple(range(rec.channel_count)) if channels is None else tuple(int(c) for c in channels)
    if not chans or any(not 0 <= c < rec.channel_count for c in chans):
        raise ValueError(f"channels must be within 0..{rec.channel_count}, got {list(chans)}")
    size = _window_samples(rec, window_s)
    spans = [(a, min(a + size, stop)) for a in range(start, stop, size)]

    def plan(i: int) -> Window:
        a, b = spans[i]
        return Window(i, a, b, max(0, a - halo), min(rec.samples, b + halo), rec.sample_rate, chans)

    def read(w: Window) -> np.ndarray:
        return rec.read(channels=list(chans), start=w.read_start, stop=w.read_stop)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dspk-custom-read") as pool:
        ahead = pool.submit(read, plan(0)) if spans else None
        for i in range(len(spans)):
            w = plan(i)
            x = ahead.result()
            ahead = pool.submit(read, plan(i + 1)) if i + 1 < len(spans) else None
            yield x, w


def _calls_with_window(fn: Callable) -> bool:
    """Whether ``fn`` takes the window too (``fn(x, w)``) or the data only (``fn(x)``)."""
    try:
        params = [p for p in inspect.signature(fn).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        varargs = any(p.kind == p.VAR_POSITIONAL for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # builtins without a signature: data only
        return False
    return varargs or len(params) >= 2


def measure(
    rec,
    fn: Callable,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    channels: Sequence[int] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> dict[str, np.ndarray]:
    """``fn(x, w)`` (or ``fn(x)``) on every window of ``rec``, its results gathered: a dict with
    ``start_s`` and ``stop_s`` (each window's span) and one array per value ``fn`` returns, stacked
    by window (a number per window gives ``(windows,)``, a row per channel ``(windows, channels)``).
    ``fn`` returns a value or a dict of values (a value alone is called ``"value"``); every window
    must return the same keys and shapes. The last window may be shorter."""
    with_w = _calls_with_window(fn)
    rows: dict[str, list] = {}
    starts, stops = [], []
    for x, w in windows(rec, window_s=window_s, channels=channels, start=start, stop=stop):
        out = fn(x, w) if with_w else fn(x)
        out = out if isinstance(out, dict) else {"value": out}
        if rows and set(out) != set(rows):
            raise ValueError(f"window {w.index}: fn returned {sorted(out)}, the first window {sorted(rows)}")
        for k, v in out.items():
            rows.setdefault(k, []).append(np.asarray(v))
        starts.append(w.start_s)
        stops.append(w.stop_s)
    table = {"start_s": np.asarray(starts), "stop_s": np.asarray(stops)}
    for k, v in rows.items():
        try:
            table[k] = np.stack(v)
        except ValueError as e:
            raise ValueError(f"{k!r}: every window must return the same shape ({e})") from None
    return table


class Transform:
    """A function of a window with ``halo`` samples of context on each side, run over recordings
    window by window (see the module). Make one with :func:`transform`."""

    def __init__(self, fn: Callable, *, halo: int = 0, window_s: float = DEFAULT_WINDOW_S):
        if halo < 0:
            raise ValueError(f"halo must be at least 0, got {halo}")
        self.fn, self.halo, self.window_s = fn, int(halo), window_s
        self._with_w = _calls_with_window(fn)

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", repr(self.fn))
        return f"Transform({name}, halo={self.halo}, window_s={self.window_s})"

    def __call__(self, x: np.ndarray, w: Window | None = None) -> np.ndarray:
        """``fn`` on one array (no windows)."""
        return self.fn(x, w) if self._with_w else self.fn(x)

    def blocks(self, rec, *, channels: Sequence[int] | None = None, start: int = 0, stop: int | None = None) -> Iterator[tuple[np.ndarray, Window]]:
        """The processed windows of ``rec``: ``(y, w)`` with ``y`` the window's own samples
        ``(channels, w.stop - w.start)`` float32."""
        for x, w in windows(rec, window_s=self.window_s, halo=self.halo, channels=channels, start=start, stop=stop):
            y = np.asarray(self.fn(x, w) if self._with_w else self.fn(x))
            if y.shape != x.shape:
                raise ValueError(f"window {w.index}: fn returned shape {y.shape} for a window of {x.shape}: a transform keeps the shape")
            yield np.ascontiguousarray(y[:, w.inner], dtype=np.float32), w

    def apply(self, rec, *, channels: Sequence[int] | None = None, start: int = 0, stop: int | None = None) -> np.ndarray:
        """``rec`` (``start``–``stop``, ``channels``) processed, as one array ``(channels, samples)``."""
        return np.concatenate([y for y, _ in self.blocks(rec, channels=channels, start=start, stop=stop)], axis=1)

    def write(
        self,
        rec,
        path: str | PathLike,
        *,
        channels: Sequence[int] | None = None,
        start: int = 0,
        stop: int | None = None,
        chunk_samples: int = io.ZARR_CHUNK_SAMPLES,
        compression: str = "zstd",
    ) -> None:
        """``rec`` processed, written to a new Zarr store at ``path`` window by window (never the
        whole recording in memory); channel names and units as ``rec``'s."""
        start, stop = _span(rec, start, stop)
        chans = list(range(rec.channel_count)) if channels is None else [int(c) for c in channels]
        units = {rec.units[c] for c in chans}
        writer = io.ZarrWriter(
            path, len(chans), stop - start, rec.sample_rate, unit=units.pop() if len(units) == 1 else "",
            chunk_samples=chunk_samples, compression=compression, channel_names=[rec.channel_names[c] for c in chans],
        )
        for y, _ in self.blocks(rec, channels=chans, start=start, stop=stop):
            writer.push(y)
        writer.finish()


def transform(fn: Callable | None = None, *, halo: int = 0, window_s: float = DEFAULT_WINDOW_S):
    """A :class:`Transform` of ``fn``; also a decorator (``@dspk.custom.transform(halo=12)``)."""
    if fn is None:
        return lambda f: Transform(f, halo=halo, window_s=window_s)
    return Transform(fn, halo=halo, window_s=window_s)


__all__ = ["DEFAULT_WINDOW_S", "Transform", "Window", "measure", "transform", "windows"]
