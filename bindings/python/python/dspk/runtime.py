"""Compute runtime choice.

Every call that runs on a device uses, in order: its ``runtime=`` argument, the runtime chosen with
:func:`set`, the ``DSPK_RUNTIME`` environment variable, the first compiled-in runtime (GPUs first).
"""

from dspk._native import runtime as _runtime

available = _runtime.available
current = _runtime.current
set = _runtime.set  # noqa: A001 (mirrors the Rust name)

__all__ = ["available", "current", "set"]
