"""
Composable in-VRAM DSP Pipeline Engine.
Zero-host-PCIe round-trips via double-buffered ping-pong GPU memory.
"""

from .._dsp_kitchen import Pipeline

__all__ = [
    "Pipeline",
]
