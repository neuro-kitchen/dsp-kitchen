# %% [markdown]
# # 09 · Sorting — 2. Judging a sorting: against a reference, and unit by unit
# A sorting is a list of units, each a spike train. Two questions about it:
#
# - **Does it agree with a reference?** Here dspk's Kilosort4 (`dspk sort kilosort4`) against
#   Kilosort4's own results on the same recording: units are paired 1-to-1 by spike-time
#   agreement (spikes within 0.4 ms), `compare_sortings`.
# - **Is each unit one neuron?** A neuron cannot fire twice within ~1.5 ms (refractory period), so
#   inter-spike-interval violations reveal mixed units; firing over the whole recording (presence)
#   and a full amplitude distribution (amplitude cutoff) reveal units that drift away or are cut by
#   the threshold. The autocorrelogram of a clean unit has a hole at zero lag.
#
# Needs the recording (`00_fetch_kilosort_data.py`) and the two sortings next to it:
# `<recording>.kilosort4.sorting.zarr` (from `dspk sort kilosort4 <recording> -o …`) and
# Kilosort4's `saved_results/` (its Phy folder).

# %%
from pathlib import Path

import numpy as np

import plating
from dspk.synapse import io, metrics

FIG = Path(__file__).parent / "figures"
DATA = Path(__file__).resolve().parents[1] / "data" / "kilosort4"
OURS = DATA / "ZFM-02370_mini.imec0.ap.short.kilosort4.sorting.zarr"
REFERENCE = DATA / "saved_results"
for path in (OURS, REFERENCE):
    if not path.exists():
        raise SystemExit(f"{path} is missing (see this chapter's README)")

ours, reference = io.Sorting.load(OURS), io.Sorting.load(REFERENCE)
print("dspk:     ", ours)
print("Kilosort4:", reference)

# Agreement: each reference unit's best match among ours
cmp = metrics.compare_sortings(reference, ours)
best = cmp["agreement"].max(axis=1)
print(f"{len(cmp['matched'])} units paired; {len(cmp['well_detected'])} of {len(reference)} reference units found at ≥ {metrics.DEFAULT_AGREEMENT_THRESHOLD:.0%} agreement")

# Quality of each of our units (computed when the sorting was saved / loaded)
units = ours.units()
rate, isi, presence = np.asarray(units["firing_rate_hz"]), np.asarray(units["isi_violation_ratio"]), np.asarray(units["presence_ratio"])
single = np.asarray(units["label"]) == "SingleUnit"
print(f"{single.sum()} of {single.size} units labelled single units")

# A clean, busy unit, and its autocorrelogram
ids = np.asarray(units["unit_id"])
clean = ids[single][np.argmax(np.asarray(units["spikes"])[single])]
acg = metrics.autocorrelogram(ours.spike_train(int(clean)), ours.sample_rate, bin_ms=0.5, window_ms=50.0)

# %% Figure
p = plating.theme.palette()
fig, axes = plating.figure("Judging a sorting: dspk's Kilosort4 against Kilosort4, and unit by unit", cols=3, size=(14.0, 4.8))
axes[0].hist(best * 100, bins=np.linspace(0, 100, 21), color=p.series[0])
plating.theme.reference_line(axes[0], metrics.DEFAULT_AGREEMENT_THRESHOLD * 100, "well detected", horizontal=False)
axes[0].set_xlabel("best agreement of each Kilosort4 unit (%)")
axes[0].set_ylabel("units")
axes[0].set_title(f"{len(cmp['well_detected'])} of {len(reference)} reference units found")
axes[1].scatter(rate[~single], isi[~single] * 100, s=12, color=p.muted, label="multi-unit")
axes[1].scatter(rate[single], isi[single] * 100, s=12, color=p.series[1], label="single unit")
axes[1].set_xscale("log")
axes[1].set_xlabel("firing rate (Hz)")
axes[1].set_ylabel("ISI violations (% of expected)")
axes[1].set_title("refractory violations by firing rate")
axes[1].legend(loc="upper left")
lags = np.asarray(acg["lags_ms"]) if "lags_ms" in acg else (np.arange(len(acg["counts"])) - len(acg["counts"]) // 2) * 0.5
axes[2].bar(lags, acg["counts"], width=0.5, color=p.series[2])
axes[2].set_xlabel("lag (ms)")
axes[2].set_ylabel("spike pairs")
axes[2].set_title(f"unit {clean}: autocorrelogram (refractory dip at 0)")
plating.save(fig, FIG / "02_sorting_quality.png", note="metrics.compare_sortings (0.4 ms) of dspk's kilosort4 against Kilosort4's saved results; io.Sorting.units() quality; metrics.autocorrelogram.")
