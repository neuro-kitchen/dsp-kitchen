"""The desktop app, opened from Python (``pip install "dspk[app]"``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["open", "program"]

_NAME = "dspk-app"


def program() -> Path | None:
    """Where the ``dspk-app`` program is, or ``None`` when it is not installed.

    Returns
    -------
    pathlib.Path or None
        Next to this Python first (``dspk[app]`` puts it there), else from ``PATH``.
    """
    beside = Path(sys.executable).parent / (_NAME + (".exe" if os.name == "nt" else ""))
    if beside.is_file():
        return beside
    found = shutil.which(_NAME)
    return Path(found) if found else None


def open(path: str | os.PathLike | None = None) -> subprocess.Popen:  # noqa: A001 (as dspk.io.open)
    """Opens the desktop app, optionally on a recording, and returns without waiting for it.

    Parameters
    ----------
    path : str or os.PathLike, optional
        The recording to open; by default the app starts as usual.

    Returns
    -------
    subprocess.Popen
        The running app.

    Raises
    ------
    RuntimeError
        When the program is not installed.

    Examples
    --------
    >>> import dspk
    >>> app = dspk.app.open("session.bin")  # doctest: +SKIP
    """
    app = program()
    if app is None:
        raise RuntimeError('the desktop app is not installed: pip install "dspk[app]"')
    return subprocess.Popen([str(app), *([os.fspath(path)] if path is not None else [])])
