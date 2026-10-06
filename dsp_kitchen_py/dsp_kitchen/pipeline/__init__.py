"""
``Pipeline``: stages chained on the device; intermediate results never leave it. ``fs`` is
required when a stage is a filter.
"""

from dsp_kitchen_bindings import Pipeline

__all__ = ["Pipeline"]
