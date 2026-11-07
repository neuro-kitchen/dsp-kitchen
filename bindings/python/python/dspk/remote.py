"""A client of ``dspk serve``: a GPU server holds the data and runs the sorts; this machine looks
and asks.

.. code-block:: python

    import dspk

    with dspk.remote.Session("192.168.1.20:50051", "me.dspk-identity.pem") as server:
        server.info()                          # who the server is, what this client may do
        server.files("sessions/")              # its data root
        rec = server.open("sessions/a.zarr")   # a dspk.io.Recording, read on the server
        window = rec.read(start=0, stop=30_000)
        server.sort("mountainsort5", "sessions/a.zarr")
        server.wait()                          # follows it (progress, logs on failure)
        server.fetch(server.status()["output"], "a.ms5.sorting.zarr")

The identity file is what ``dspk clients enroll <name>`` (``--control`` to run jobs) wrote on the
server. Keep it private; it is the client's key.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from dspk._native import remote as _remote

DEFAULT_SERVER_NAME: str = _remote.DEFAULT_SERVER_NAME
#: Seconds between status requests while :meth:`Session.wait` follows a job.
DEFAULT_POLL_S: float = 1.0


class Session(_remote.Session):
    """A connection to a ``dspk serve`` server: a GPU machine that holds the data and runs the sorts.

    Use it as a context manager, or :meth:`close` it.

    Parameters
    ----------
    address : str
        The server's ``IP:port``, such as ``"192.168.1.20:50051"``.
    identity : str or os.PathLike
        The client's identity file, written by ``dspk clients enroll <name>`` on the server
        (``--control`` to run jobs). Keep it private: it is the client's key.
    server_name : str
        A name on the server's certificate (``dspk serve --name`` adds names).

    Raises
    ------
    ValueError
        When ``address`` is not an ``IP:port`` or the identity file does not exist.
    RuntimeError
        When the server cannot be reached or refuses the client.

    Examples
    --------
    >>> import dspk
    >>> with dspk.remote.Session("192.168.1.20:50051", "me.dspk-identity.pem") as server:  # doctest: +SKIP
    ...     rec = server.open("sessions/a.zarr")
    ...     window = rec.read(start=0, stop=30_000)
    """

    def wait(self, *, poll_s: float = DEFAULT_POLL_S, progress: Callable[[dict], None] | None = None, timeout_s: float | None = None) -> dict:
        """Follows the server's job until it stops.

        Parameters
        ----------
        poll_s : float
            Seconds between status requests.
        progress : callable, optional
            ``progress(status)``, called at each request while the job runs.
        timeout_s : float, optional
            Give up after this many seconds; by default wait as long as it takes.

        Returns
        -------
        dict
            The last status (see :meth:`status`).

        Raises
        ------
        RuntimeError
            When the job failed, with its last log lines.
        TimeoutError
            After ``timeout_s``.
        """
        start = time.monotonic()
        while True:
            status = self.status()
            if status["state"] != "running":
                break
            if progress is not None:
                progress(status)
            if timeout_s is not None and time.monotonic() - start > timeout_s:
                raise TimeoutError(f"the job is still running after {timeout_s} s: {status['job']}")
            time.sleep(poll_s)
        if status["state"] == "error":
            logs = "\n".join(self.logs(20))
            raise RuntimeError(f"the job failed on the server: {status['message']}\n{logs}")
        return status


__all__ = ["DEFAULT_POLL_S", "DEFAULT_SERVER_NAME", "Session"]
