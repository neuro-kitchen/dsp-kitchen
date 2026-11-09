# %% [markdown]
# # Tridesclous 2 validation: our output vs Kilosort4's saved results
# There is no Tridesclous 2 reference on this recording, so its units are compared with Kilosort4's
# (`.datasets/kilosort4/saved_results/`), as a second, independent sorter on the same data.
#
#     uv run python benchmarks/sorters/tridesclous2_validation.py [--rerun] [name=value ...]
#
# `--rerun` sorts again instead of reading an earlier run; `name=value` sets a sorter flag. Writes
# `results/tridesclous2_validation.json`. Needs `dspk-cli-sorter` (see `sorter_cli.py`).
#
# Data (local, git-ignored): `.datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`),
# `.datasets/kilosort4/saved_results/` (see benchmarks/README.md).

# %%
from ks4_comparison import validate

validate("tridesclous2", near_um=60.0)
