"""Filters on the device, one module per family (mirroring ``dspk_ops::filter``).

- IIR designs and filtering with ``scipy.signal`` semantics: :func:`sos`, :func:`iir`,
  :func:`notch`, :func:`sosfilt`, :func:`settling_samples`.
- FIR filtering, design and Gaussian smoothing: :func:`fir`, :func:`firwin`,
  :func:`gaussian_smooth`.
- Non-linear: :func:`median`, :func:`teager_kaiser`.
- Template subtraction: :func:`subtract_template`.

The implementation files (``_iir.py``, ``_fir.py``, ...) are private; use the names here.
"""

from dspk.ops.filter._fir import FIR_DEFAULT_EDGE, GAUSSIAN_DEFAULT_EDGE, GAUSSIAN_TRUNCATE, fir, firwin, gaussian_smooth
from dspk.ops.filter._iir import DEFAULT_ORDER, SETTLING_TOLERANCE, iir, notch, settling_samples, sos, sosfilt
from dspk.ops.filter._non_linear import MAX_MEDIAN_WIDTH, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE, median, teager_kaiser
from dspk.ops.filter._template import TEMPLATE_DEFAULT_MAX_LAG, subtract_template

__all__ = [
    "DEFAULT_ORDER", "FIR_DEFAULT_EDGE", "GAUSSIAN_DEFAULT_EDGE", "GAUSSIAN_TRUNCATE", "MAX_MEDIAN_WIDTH",
    "MEDIAN_DEFAULT_EDGE", "SETTLING_TOLERANCE", "TEAGER_KAISER_DEFAULT_EDGE", "TEMPLATE_DEFAULT_MAX_LAG",
    "fir", "firwin", "gaussian_smooth", "iir", "median", "notch", "settling_samples", "sos", "sosfilt",
    "subtract_template", "teager_kaiser",
]
