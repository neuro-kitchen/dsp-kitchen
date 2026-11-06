"""dspk.synapse: electrophysiology on top of dspk — probes and sortings (``io``), and unit quality,
correlograms, firing rates, evoked responses and sorting comparison (``metrics``). Detection,
features, clustering, matching and localisation follow (dspk-synapse's building blocks).
"""

from dspk.synapse import io, metrics

__all__ = ["io", "metrics"]
