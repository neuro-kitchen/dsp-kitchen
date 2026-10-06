"""
Template subtraction: each event's template aligned by cross-correlation (±``max_lag``), scaled by
least squares (``dynamic_scaling``) and subtracted.
"""

from dsp_kitchen_bindings import TemplateFilter, subtract_template

__all__ = ["TemplateFilter", "subtract_template"]
