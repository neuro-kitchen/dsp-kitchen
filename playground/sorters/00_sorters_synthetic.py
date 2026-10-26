# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): Kilosort4 and EMUsort on Synthetic Ground Truth
# Checks the sorter runners with no data (`io.SyntheticRecording`, known spike times):
# 1. Channel delays on the device: known delays recovered by `estimate_channel_delays` and
#    removed by `apply_channel_delays`
# 2. Kilosort4 over the recording (halo windows, device), detection vs ground truth
# 3. EMUsort over the same recording with known per-channel delays: delays recovered by the run,
#    detection vs ground truth (spike times are in the reference channel's frame)
# 4. The same run on every compiled runtime (one device path, any device)
# 5. Export as a `SortingOutput`
#
# Clustering, deconvolution and merging are not implemented yet: units are universal templates,
# so only detection is scored here. A delay is only recoverable on channels that share spikes
# with the reference channel (synthetic units cover a few neighbouring channels), so expect
# exact delays near the reference and noise far from it.

# %% [1] Ground-Truth Recording
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.filter.iir import HighpassFilter
from dsp_kitchen.io import Recording, SyntheticRecording
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.synapse.ml import emusort, kilosort4

FS = 30_000.0
CHANNELS = 32
DURATION_SEC = 10.0
UNITS = 6
SEED = 7
CONTACT_PITCH_UM = 25.0  # linear probe; synthetic units spread over neighbouring channels
HIGHPASS_HZ = 300.0  # Kilosort4's high-pass, used here only for the stand-alone delay check
MATCH_MS = 0.4  # spike-matching tolerance (SpikeInterface default)
# Known delays (samples) for EMUsort: a slow gradient along the probe, within its ±2 ms search
DELAY_STEP_SAMPLES = 1
OUTPUT = Path(__file__).resolve().parents[1] / "output"

truth = SyntheticRecording(CHANNELS, FS, DURATION_SEC, units=UNITS, seed=SEED)
rec = truth.recording()
probe = syn.ProbeLayout.from_positions(
    "linear-32", [(0.0, i * CONTACT_PITCH_UM) for i in range(CHANNELS)]
)
gt = sorted(s for u in range(truth.unit_count) for s in truth.spike_times(u))
print(f"{rec}: {truth.unit_count} units, {len(gt):,} ground-truth spikes")
print(f"Runtimes compiled in: {dk.runtime.available()}; default: {dk.runtime.current()}")


def score(detected, reference, label):
    cmp = syn.compare_spike_trains(
        list(reference), list(detected), fs=FS, delta_time_ms=MATCH_MS
    )
    print(
        f"  {label}: {len(detected):,} detected vs {len(reference):,} true — "
        f"recall {cmp['recall']:.3f}, precision {cmp['precision']:.3f}"
    )
    return cmp


# Delayed copy: channel c lags by c · DELAY_STEP_SAMPLES (a MUAP reaching contacts in turn)
raw = rec.read(0, rec.samples)
true_delays = np.arange(CHANNELS) * DELAY_STEP_SAMPLES
delayed = np.stack([np.roll(raw[c], int(d)) for c, d in enumerate(true_delays)])
rec_delayed = Recording.from_array(delayed.astype(np.float32), FS, name="synthetic-delayed")

# %% [2] Channel Delays on the Device (stand-alone)
# High-passed one-second batches with `max_lag` samples of padding, as the runner sees them
max_lag = emusort.Config().max_delay_samples(FS)  # ±2 ms, EMUsort's search range
batch = int(FS)
filtered = Pipeline([HighpassFilter(HIGHPASS_HZ)]).run(delayed, fs=FS)
batches = [
    np.ascontiguousarray(filtered[:, start : start + batch])
    for start in range(0, filtered.shape[1] - batch + 1, batch)
]
delays, reference = emusort.estimate_channel_delays(batches, pad=max_lag, max_lag=max_lag)
expected = true_delays - true_delays[reference]
ok = np.asarray(delays) == expected
print(f"\n[Delays] reference channel {reference}; {ok.sum()}/{CHANNELS} channels exact")
print(f"  recovered: {list(delays)}")
print(f"  expected:  {list(expected)}")

aligned = emusort.apply_channel_delays(
    np.ascontiguousarray(filtered[:, :batch]), delays
)
inner = slice(max_lag + CHANNELS * DELAY_STEP_SAMPLES, batch - max_lag - CHANNELS * DELAY_STEP_SAMPLES)
undelayed = Pipeline([HighpassFilter(HIGHPASS_HZ)]).run(raw[:, :batch], fs=FS)
# Aligned channels equal the undelayed signal shifted into the reference channel's frame
residual = max(
    np.abs(aligned[c, inner] - np.roll(undelayed[c], true_delays[reference])[inner]).max()
    for c in np.flatnonzero(ok)
)
print(f"  alignment residual on exact channels (µV, filter edge effects only): {residual:.3g}")

# %% [3] Kilosort4 over the Recording
ks_config = kilosort4.Config()
ks_config.whitening_range = min(ks_config.whitening_range, CHANNELS)
t0 = time.perf_counter()
ks = kilosort4.run(rec, probe, ks_config)
print(f"\n{ks} ({time.perf_counter() - t0:.1f} s)")
ks_spikes = sorted(ks.spikes()["sample"])
score(ks_spikes, gt, "Kilosort4")

# %% [4] EMUsort over the Delayed Recording
emu_config = emusort.Config()
emu_config.whitening_range = min(emu_config.whitening_range, CHANNELS)
t0 = time.perf_counter()
emu = emusort.run(rec_delayed, probe, emu_config)
print(f"\n{emu} ({time.perf_counter() - t0:.1f} s)")
run_delays, run_reference = emu.channel_delays
run_expected = true_delays - true_delays[run_reference]
print(
    f"  run delays: reference {run_reference}, "
    f"{(np.asarray(run_delays) == run_expected).sum()}/{CHANNELS} channels exact"
)
emu_spikes = sorted(emu.spikes()["sample"])
# Detection is in the reference channel's frame: shift the truth into it
score(emu_spikes, [s + int(true_delays[run_reference]) for s in gt], "EMUsort")

# %% [5] Every Compiled Runtime
print("\n[Runtimes] Kilosort4 detection per runtime")
per_runtime = {dk.runtime.current(): ks_spikes}  # section [3] ran on the default runtime
print(f"  {dk.runtime.current()}: {len(ks_spikes):,} spikes (section [3])")
for name in dk.runtime.available():
    if name in per_runtime:
        continue
    t0 = time.perf_counter()
    res = kilosort4.run(rec, probe, ks_config, runtime=name)
    per_runtime[name] = sorted(res.spikes()["sample"])
    print(f"  {name}: {len(per_runtime[name]):,} spikes in {time.perf_counter() - t0:.1f} s")
names = list(per_runtime)
for other in names[1:]:
    cmp = syn.compare_spike_trains(
        per_runtime[names[0]], per_runtime[other], fs=FS, delta_time_ms=MATCH_MS
    )
    print(f"  {names[0]} vs {other}: agreement {cmp['agreement_score']:.3f}")

# %% [6] Export
sorting = emu.to_sorting_output(probe)
print(f"\n{sorting}")
for row in sorting.summary_table()[:5]:
    print(f"  {row}")
OUTPUT.mkdir(exist_ok=True)
sorting.save(str(OUTPUT / "emusort_synthetic.sorting.zarr"))
print(f"Saved to {OUTPUT / 'emusort_synthetic.sorting.zarr'}")

# %% [7] Plot
if HAS_PLT:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(expected, "k--", label="expected")
    axes[0].plot(delays, "o", label="stand-alone")
    axes[0].plot(run_delays, "x", label="EMUsort run")
    axes[0].set_title("Channel delays (samples)", fontweight="bold")
    axes[0].set_xlabel("Channel")
    axes[0].legend()
    t_ms = (np.arange(ks_config.nt) - ks_config.resolved_nt0min()) / FS * 1e3
    for row in ks.templates.wtemp:
        axes[1].plot(t_ms, row, linewidth=1.2)
    axes[1].set_title("Kilosort4 universal templates", fontweight="bold")
    axes[1].set_xlabel("Time from peak (ms)")
    s = ks.spikes()
    axes[2].scatter(np.asarray(s["sample"]) / FS, s["y_um"], s=3, c=s["template"], cmap="tab10")
    axes[2].set_title("Kilosort4 spikes: time vs position", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

# %% [8] Inspect One Channel
# The run's own preprocessing of the synthetic recording, with Kilosort4's units on the reference
# channel; change channel / window in the figure.
if HAS_PLT:
    from inspection import inspect, preprocessed_segment

    ch = int(run_reference)
    signal, offset = preprocessed_segment(ks, rec, 0.0, rec.samples / FS)
    inspector = inspect(ks.to_sorting_output(probe), signal, FS, offset=offset, channel=ch, start=0.0, end=1.5, title=f"Kilosort4 synthetic: channel {ch}")

# %%
