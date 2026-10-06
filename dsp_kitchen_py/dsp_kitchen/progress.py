"""
Progress of long runs (a sorter over a recording). Runs report ``(stage, step, steps, done,
total, unit)``; :class:`ProgressBar` draws it with ``tqdm`` when installed (terminals and
notebooks), else as one line of text on stderr, with the time left estimated from the rate.

    result = kilosort4.run(rec, probe, config)                  # a bar by default
    result = kilosort4.run(rec, probe, config, progress=False)  # silent
    result = kilosort4.run(rec, probe, config, progress=my_fn)  # my_fn(stage, step, steps, done, total, unit)
"""

import sys
import time

__all__ = ["ProgressBar", "progress_callback"]

#: Characters of the text bar.
TEXT_BAR_WIDTH = 24
#: Least time between two redraws of the text bar (seconds).
TEXT_REDRAW_SEC = 0.1


def _clock(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ProgressBar:
    """One bar per stage of a run: ``[2/4] Finding clips  12/40 windows  0:05 < 0:12``."""

    def __init__(self, file=None):
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
        self._tqdm = tqdm
        self._file = file or sys.stderr
        self._key = None
        self._bar = None
        self._start = 0.0
        self._drawn = 0.0
        self._width = 0

    def __call__(self, stage, step, steps, done, total, unit):
        key = (step, stage)
        if key != self._key:
            self.close()
            self._key = key
            self._start = time.monotonic()
            self._drawn = 0.0
            self._width = 0
            label = f"[{step}/{steps}] {stage}"
            if self._tqdm is not None:
                self._bar = self._tqdm(total=total or None, desc=label, unit=f" {unit}", leave=True)
            else:
                self._bar = label
        if self._tqdm is not None:
            if total and self._bar.total != total:
                self._bar.total = total
            self._bar.n = done
            self._bar.refresh()
        else:
            self._text(done, total, unit)
        if total and done >= total:
            self.close()

    def _text(self, done, total, unit):
        now = time.monotonic()
        finished = bool(total) and done >= total
        if not finished and now - self._drawn < TEXT_REDRAW_SEC:
            return
        self._drawn = now
        elapsed = now - self._start
        if total:
            filled = int(TEXT_BAR_WIDTH * min(done / total, 1.0))
            bar = "█" * filled + "░" * (TEXT_BAR_WIDTH - filled)
            left = f" < {_clock(elapsed * (total - done) / done)}" if 0 < done < total else ""
            line = f"{self._bar}  {bar}  {done}/{total} {unit}  {_clock(elapsed)}{left}"
        else:
            line = f"{self._bar}  {done} {unit}  {_clock(elapsed)}"
        # Pad over a longer previous line
        self._file.write("\r" + line.ljust(self._width))
        self._width = len(line)
        self._file.flush()

    def close(self):
        """Ends the current stage's bar."""
        if self._bar is None:
            return
        if self._tqdm is not None:
            self._bar.close()
        else:
            self._file.write("\n")
            self._file.flush()
        self._bar = None
        self._key = None


def progress_callback(progress):
    """The callable a run reports to: a :class:`ProgressBar` for ``True``, nothing for ``False``
    or ``None``, else ``progress`` itself."""
    if progress is True:
        return ProgressBar()
    if progress is False or progress is None:
        return None
    if callable(progress):
        return progress
    raise TypeError("progress must be True, False, None or a callable")
