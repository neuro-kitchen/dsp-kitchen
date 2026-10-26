"""
Filters, each a pipeline stage object and a function of the same name in lower case:

- ``iir``: Butterworth (band/high/low/stop), Chebyshev I, notch; zero phase by default.
- ``fir``: Gaussian smoothing.
- ``non_linear``: running median, Teager-Kaiser energy.
- ``template``: template alignment, scaling and subtraction.
"""

from . import fir, iir, non_linear, template

__all__ = ["fir", "iir", "non_linear", "template"]
