"""
Non-linear filters: running median (``scipy.signal.medfilt``, width 9 and zero edges by default)
and the Teager-Kaiser energy operator.
"""

from dsp_kitchen_bindings import MedianFilter, TeagerKaiser, median_filter, teager_kaiser_filter

__all__ = ["MedianFilter", "TeagerKaiser", "median_filter", "teager_kaiser_filter"]
