"""
Template subtraction filter: alignment, dynamic least-squares amplitude scaling,
and artifact cancellation.
"""

from .._dsp_kitchen import TemplateFilter, subtract_template

__all__ = ["TemplateFilter", "subtract_template"]
