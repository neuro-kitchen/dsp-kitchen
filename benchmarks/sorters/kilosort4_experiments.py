# %% [markdown]
# # Kilosort4 experiments: where does our output depart from Kilosort4's?
# - `learned`: our run as is, its spikes against Kilosort4's.
# - `ks4-templates`: our detection with Kilosort4's learned universal templates instead of ours. If
#   the match improves, template learning is where we differ; if not, detection or preprocessing.
#
#     uv run python benchmarks/sorters/kilosort4_experiments.py [learned] [ks4-templates] [--rerun]
#
# Each run's sorting stays in `results/kilosort4_experiments/`. Needs `dspk-cli-sorter`.

# %%
import sys

import numpy as np

from ks4_comparison import probe_file
from ks4_reference import BIN_PATH, Reference, compare
from sorter_cli import RESULTS, flatten, load, save_json, sort

OUTPUT = RESULTS / "kilosort4_experiments"
STEPS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["learned", "ks4-templates"]
RERUN = "--rerun" in sys.argv
ref = Reference.load()
OUTPUT.mkdir(parents=True, exist_ok=True)
outcome = {}


def run(name, **options):
    path = OUTPUT / f"{name}.sorting.zarr"
    if RERUN or not path.exists():
        sort("kilosort4", BIN_PATH, path, probe_file=probe_file(), **options)
    times, _, locs = flatten(load(path))
    print(f"\n== {name}")
    outcome[name] = compare(ref, times, ref.nearest_channel(locs), label=name, table_rows=8)


# %% Our run
if "learned" in STEPS:
    run("learned")

# %% Our detection with Kilosort4's templates
if "ks4-templates" in STEPS:
    npz = OUTPUT / "ks4_learned_templates.npz"
    np.savez(npz, wPCA=ref.ops["wPCA"].astype(np.float32), wTEMP=ref.ops["wTEMP"].astype(np.float32))
    run("ks4_templates", templates=npz, flags={"templates_from_data": "false"})

save_json("kilosort4_experiments", {"outcome": outcome})
