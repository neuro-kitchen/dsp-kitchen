# Reference outputs of SpikeInterface's isosplit (isosplit_isocut.py) for the Rust test
# `spikeinterface_variant_matches_upstream`. Usage: uv run --with numpy --with numba --with scipy python
#   isosplit_si_reference.py <folder containing isosplit_isocut.py> <out.json>
import importlib.util
import sys, json
import numpy as np
sys.path.insert(0, sys.argv[1])
from isosplit_isocut import isosplit
rng = np.random.default_rng(3)
cases = []
for case in range(4):
    centres = rng.normal(0, 6, size=(3 + case, 3))
    X = np.concatenate([c + rng.normal(0, 1, size=(150, 3)) for c in centres]).astype(np.float64)
    # initial labels: 12 groups by a fixed projection's rank
    proj = X @ np.array([0.3, -0.5, 0.8])
    init = (np.argsort(np.argsort(proj)) * 12 // len(X)).astype(np.int64)
    out = isosplit(X, initial_labels=init, min_cluster_size=10, max_iterations_per_pass=500, isocut_threshold=2.0)
    cases.append(dict(x=X.ravel().tolist(), n=len(X), m=3, init=init.tolist(), out=np.asarray(out).tolist()))
json.dump(cases, open(sys.argv[2], "w"))
print("cases", [max(c["out"]) + 1 for c in cases])
