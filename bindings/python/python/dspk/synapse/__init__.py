"""dspk.synapse: electrophysiology on top of dspk — probes and sortings (``io``), and unit quality,
correlograms, firing rates, evoked responses and sorting comparison (``metrics``), spike detection
(``detection``), snippets (``extraction``) and waveform features (``features``). Clustering,
matching and localisation follow (dspk-synapse's building blocks).
"""

from dspk.synapse import detection, extraction, features, io, metrics

__all__ = ["detection", "extraction", "features", "io", "metrics"]
