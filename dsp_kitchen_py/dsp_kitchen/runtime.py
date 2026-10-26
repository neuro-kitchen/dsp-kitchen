"""
Compute runtime for native calls: an explicit ``runtime=`` argument, else the runtime chosen with
``set``, else ``DSP_KITCHEN_RUNTIME``, else the first compiled-in runtime (GPUs first).
"""

from dsp_kitchen_bindings import available_runtimes as available
from dsp_kitchen_bindings import current_runtime as current
from dsp_kitchen_bindings import set_runtime as set  # noqa: A001 (module-level API name)

__all__ = ["available", "current", "set"]
