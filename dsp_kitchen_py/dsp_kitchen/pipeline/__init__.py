"""
Composable in-VRAM DSP Pipeline Engine.
Zero-host-PCIe round-trips via double-buffered ping-pong GPU memory.
"""

from .._bindings import DspSession, Pipeline

__all__ = [
    "Pipeline",
    "DspSession",
]
