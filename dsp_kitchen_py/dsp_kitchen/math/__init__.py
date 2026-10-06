"""
Pointwise stages: ``Scale`` (``x · alpha + beta``), ``SubtractBaseline``, ``Clamp``.
"""

from dsp_kitchen_bindings import Clamp, Scale, SubtractBaseline, scale_samples

__all__ = ["Clamp", "Scale", "SubtractBaseline", "scale_samples"]
