"""How dsp-kitchen's figures are presented ("plating": how a dish is presented).

- :mod:`plating.theme`: the look (palette, fonts, sizes), :func:`~plating.theme.figure` and
  :func:`~plating.theme.save`.
- :mod:`plating.figures`: ready figures (traces, before / after, spectrum, filter response,
  raster, waveforms, benchmark timings and agreement).
- :mod:`plating.signals`: a synthetic recording whose content is known.
- :mod:`plating.results`: the benchmark JSON (write, read).

Every playground chapter and benchmark draws through this package, so all figures look like one
set and nothing about the style lives in the scripts.
"""

from plating import figures, results, signals, theme
from plating.theme import figure, save, use

__all__ = ["figure", "figures", "results", "save", "signals", "theme", "use"]
