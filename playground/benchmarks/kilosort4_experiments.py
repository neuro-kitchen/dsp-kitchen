# %% [markdown]
# # Kilosort4 experiments: where does our output depart from Kilosort4's?
# Settings-only experiments against `data/kilosort4/saved_results/` (no change to the sorter):
# - `learned`: our run as is; its universal templates vs the ones Kilosort4 learned
#   (`ops['wPCA']`, `ops['wTEMP']`), then its spikes vs Kilosort4's.
# - `ks4-templates`: our detection with Kilosort4's learned templates instead of ours: if the
#   match improves, template learning is where we differ; if not, detection or preprocessing.
#
# Spike times are compared as reported: both sorters report waveform troughs.
#
#     uv run python playground/benchmarks/kilosort4_experiments.py learned ks4-templates
#
# Each run's spikes are cached in `playground/output/kilosort4_experiments/` (`--rerun` re-sorts).

# %% [1] Settings
import sys
from pathlib import Path

import numpy as np

import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording
from dsp_kitchen.synapse.ml import kilosort4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ks4_reference import BIN_PATH, KS4_RESULTS, Reference, best_cosine, compare  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "kilosort4_experiments"
STEPS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["learned", "ks4-templates"]
RERUN = "--rerun" in sys.argv

ref = Reference.load()
OUTPUT.mkdir(parents=True, exist_ok=True)


def recording_and_probe():
    rec = Recording(str(BIN_PATH))
    probe = syn.ProbeLayout.from_recording(str(BIN_PATH)) or syn.load_sorting(str(KS4_RESULTS)).probe
    if rec.channels != probe.total_channels:
        rec = rec.slice_samples(channels=probe.channel_ids())
    return rec, probe


def sort(name, templates=None):
    """Our run (cached by `name`): spike times, Kilosort4-space channels, and the templates used."""
    cache = OUTPUT / f"{name}.npz"
    if cache.exists() and not RERUN:
        z = np.load(cache)
        return z["times"], z["channel"], z["wpca"], z["wtemp"]
    rec, probe = recording_and_probe()
    config = kilosort4.Config()
    config.whitening_range = min(config.whitening_range, probe.total_channels)
    if templates is not None:
        config.templates_from_data = False
    result = kilosort4.run(rec, probe, config, templates=templates)
    sorting = result.to_sorting_output(probe)
    times, locs = [], []
    for u in sorting.unit_ids():
        times.append(np.asarray(sorting.spike_train(u), dtype=np.int64))
        locs.append(np.asarray(sorting.spike_locations(u), dtype=np.float64)[:, :2])
    times, channel = np.concatenate(times), ref.nearest_channel(np.concatenate(locs))
    wpca, wtemp = np.asarray(result.templates.wpca), np.asarray(result.templates.wtemp)
    np.savez(cache, times=times, channel=channel, wpca=wpca, wtemp=wtemp)
    return times, channel, wpca, wtemp


# %% [2] Our Learned Templates vs Kilosort4's
if "learned" in STEPS:
    print("\n== learned: our run as is")
    times, channel, wpca, wtemp = sort("learned")
    print(f"  wPCA  best |cos| per Kilosort4 PC:       {np.round(best_cosine(wpca, ref.ops['wPCA']), 3)}")
    print(f"  wTEMP best |cos| per Kilosort4 template: {np.round(best_cosine(wtemp, ref.ops['wTEMP']), 3)}")
    compare(ref, times, channel, label="learned", table_rows=8)

# %% [3] Our Detection with Kilosort4's Templates
if "ks4-templates" in STEPS:
    print("\n== ks4-templates: our detection with Kilosort4's learned wPCA / wTEMP")
    npz = OUTPUT / "ks4_learned_templates.npz"
    np.savez(npz, wPCA=ref.ops["wPCA"].astype(np.float32), wTEMP=ref.ops["wTEMP"].astype(np.float32))
    times, channel, _, _ = sort("ks4_templates", templates=kilosort4.UniversalTemplates.from_npz(str(npz)))
    compare(ref, times, channel, label="ks4-templates", table_rows=8)

# %%
