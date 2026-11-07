"""dspk.synapse: electrophysiology on top of dspk — probes and sortings (``io``), spike detection
(``detection``), snippets (``extraction``), waveform features (``features``), clustering
(``clustering``), templates and matching (``matching``), localisation and drift (``spatial``),
and unit quality, correlograms, firing rates, evoked responses and sorting comparison
(``metrics``): the building blocks of a sorter, on NumPy arrays.
"""

from dspk.synapse import clustering, detection, extraction, features, io, matching, metrics, spatial

__all__ = ["clustering", "detection", "extraction", "features", "io", "matching", "metrics", "spatial"]
