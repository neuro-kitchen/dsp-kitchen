"""
Template subtraction filter: alignment, dynamic least-squares amplitude scaling,
and artifact cancellation.
"""

from .._bindings import TemplateFilter, subtract_template

__all__ = ["TemplateFilter", "subtract_template"]
