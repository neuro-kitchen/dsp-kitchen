# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): Kilosort4 Front End on Neuropixels
# Kilosort4 (Pachitariu et al., Nature Methods 2024). `run_front_end` runs the stages implemented
# so far over the recording, in Rust and on the device (halo windows of `batch_size`):
# preprocessing and whitening, universal templates learned from the data, detection. Then:
# - learned templates vs Kilosort4's predefined `wTEMP.npz` (hub, verified download)
# - detected spike times vs Kilosort4's own results (its final spikes, after clustering and
#   deconvolution, which are not implemented here yet)
#
# Data (local, git-ignored), see playground/README.md:
#   `data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`), `data/kilosort4/saved_results/`.

# %% [1] Recording, Probe & Settings
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording
from dsp_kitchen.synapse.ml import kilosort4

# %%
# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(
    os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data")
)
BIN_PATH = DATA_DIR / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
KS4_RESULTS = DATA_DIR / "kilosort4" / "saved_results"

print(kilosort4.provenance().citation())
rec = Recording(str(BIN_PATH))
probe = syn.ProbeLayout.from_recording(str(BIN_PATH))
if probe is None:
    raise SystemExit("the recording carries no probe geometry")
config = kilosort4.Config()
print(f"\n{rec}\n{probe}\n{config}")

# %% [2] Run Kilosort4 over the Whole Recording
result = kilosort4.run(rec, probe, config)
ours = sorted(result.spikes()["sample"])
print(f"\n{result} on {dk.runtime.current()}")

# %% [3] Learned vs Predefined Universal Templates


def best_match(a, b):
    """For each row of `a`, the largest |cosine similarity| with any row of `b`."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.abs(a @ b.T).max(axis=1)


learned = result.templates
predefined = (
    kilosort4.UniversalTemplates.from_hub()
    if hasattr(kilosort4.UniversalTemplates, "from_hub")
    else None
)
if predefined is not None:
    print(
        f"wTEMP best |cos| (predefined → learned): {np.round(best_match(predefined.wtemp, learned.wtemp), 3)}"
    )
    print(
        f"wPCA  best |cos| (predefined → learned): {np.round(best_match(predefined.wpca, learned.wpca), 3)}"
    )

# %% [4] Detection vs Kilosort4's Results
if KS4_RESULTS.exists():
    ks4 = syn.load_sorting(str(KS4_RESULTS))
    theirs = sorted(int(t) for u in ks4.unit_ids() for t in ks4.spike_train(u))
    cmp = syn.compare_spike_trains(theirs, ours, fs=rec.sample_rate)
    print(
        f"Kilosort4 ({len(theirs):,} spikes) vs our detection ({len(ours):,}): recall {cmp['recall']:.3f}, precision {cmp['precision']:.3f}"
    )

# %% [5] Plot Templates
if HAS_PLT:
    t_ms = (np.arange(config.nt) - config.resolved_nt0min()) / rec.sample_rate * 1e3
    panels = [("learned", learned)] + (
        [("predefined", predefined)] if predefined is not None else []
    )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 4), squeeze=False
    )
    for ax, (name, templates) in zip(axes[0], panels):
        for row in templates.wtemp:
            ax.plot(t_ms, row, linewidth=1.2)
        ax.set_title(f"Universal templates ({name})", fontweight="bold")
        ax.set_xlabel("Time from peak (ms)")
    plt.tight_layout()
    plt.show()

# %%
