# %% [markdown]
# # Kilosort4 validation: our output vs Kilosort4's saved results
# 1. **Export**: the sorting written as a `.sorting.zarr` and as a Phy folder reads back the same,
#    and the Phy folder has the files Kilosort4 writes.
# 2. **Spikes per channel** and **units**, against Kilosort4's saved results (`ks4_comparison.py`).
#
#     uv run python benchmarks/sorters/kilosort4_validation.py [--rerun] [name=value ...]
#
# Writes `results/kilosort4_validation.json`. Needs `dspk-cli-sorter` (see `sorter_cli.py`).
#
# Data (local, git-ignored): `.datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`),
# `.datasets/kilosort4/saved_results/` (see benchmarks/README.md).

# %%
import json
import shutil

import numpy as np

from ks4_comparison import validate
from ks4_reference import KS4_RESULTS, read_params
from sorter_cli import RESULTS, flatten, load

record, output = validate("kilosort4", near_um=40.0)

# %% Export round trip
print("\n[Export]")
phy = output.parent / "phy"
shutil.rmtree(phy, ignore_errors=True)
ours = load(output)
ours.save(phy, "phy")
back = load(phy)
(ta, ua, _), (tb, ub, _) = flatten(ours), flatten(back)
have = {p.name for p in phy.iterdir()}
theirs = {p.name for p in KS4_RESULTS.iterdir() if p.suffix in (".npy", ".tsv", ".py")}
# Kilosort-internal files Phy does not need (Phy assumes identity whitening when absent)
optional = {"ops.npy", "kept_spikes.npy", "templates_ind.npy", "whitening_mat_dat.npy", "whitening_mat.npy", "whitening_mat_inv.npy", "cluster_Amplitude.tsv", "cluster_ContamPct.tsv"}
missing = sorted(theirs - have - optional)
checks = {
    "same spike times": bool(np.array_equal(ta, tb)),
    "same units": bool(np.array_equal(np.unique(ua, return_inverse=True)[1], np.unique(ub, return_inverse=True)[1])),
    "phy sample rate": float(read_params(phy)["sample_rate"]) == float(read_params(KS4_RESULTS)["sample_rate"]),
    f"Phy files Kilosort4 writes (missing: {missing or 'none'})": not missing,
}
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
path = RESULTS / "kilosort4_validation.json"
saved = json.loads(path.read_text())
saved["export_checks"] = checks
path.write_text(json.dumps(saved, indent=1) + "\n")
