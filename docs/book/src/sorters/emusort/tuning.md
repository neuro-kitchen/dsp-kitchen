# EMUsort: tuning

What to change to improve a sort, from the EMUsort paper (O'Connell et al., *eLife* 2026, RP110417,
Table 5 and Methods) and our measurements. Everything on the [Kilosort4 tuning](../kilosort4/tuning.md)
page applies too; this page covers what is different for EMG. Read the
[notes on our implementation](#notes-on-our-implementation) first: two of them change how the
geometry parameters behave here.

## Parameter sweeps

The paper's main tuning tool is a **sweep**: many sorts with different parameters, ranked by its
composite score (refractory violations, presence ratio, amplitude cutoff, firing-rate validity, SNR;
not implemented here yet). Its default sweep links the detection thresholds in pairs:

| `Th_universal` | 9 | 10 | 7 | 5 | 2 |
|---|---|---|---|---|---|
| `Th_learned` | 8 | 4 | 3 | 2 | 1 |

combined with sets of `Th_single_ch`. Lowering both finds more motor units (and more noise); the
paper suggests lowering `Th_universal` when MUAPs are missed, and `Th_learned` when a unit's spikes
show gaps over time.

## Thresholds of the template clips

`th_single_ch` (`[6, 9, 12, 15]`): EMUsort pools clips over several thresholds, so both small and
large motor units seed templates. Aim for at least several hundred non-overlapping clips; very low
thresholds add noise (HDBSCAN removes most of it, `hdbscan_min_cluster_size`, 20).

## Template window: `nt`

**The most important setting to check for each dataset.** Table 5: *"Important parameter that should
be matched to the observed sample width of the waveforms for a given dataset. Take note of your
sample rate and time width of the MUAPs in your data."* The paper itself used 2 ms (rat) and 4 ms
(monkey, 121 samples at 30 kHz).

`nt` is in samples, so it depends on the sampling rate: keep the **duration**. Measured on our HD-EMG
recording (4 × 8 grid, 24.4 kHz, first 200 s):

| `nt` | Duration | Spikes | Units | Units with > 1% / > 5% of ISIs < 2 ms | Template energy at the edges | Run (200 s) |
|---|---|---|---|---|---|---|
| 61 | 2.5 ms | 125 603 | 62 | 38 / 13 | 4.1% | 17.1 s |
| **121** | **5.0 ms** | 103 362 | 51 | 30 / 9 | **0.8%** | 17.6 s |
| 181 | 7.4 ms | 101 883 | 49 | 28 / 10 | 0.5% | 19.8 s |

At 2.5 ms the window cuts the MUAPs and matching pursuit matches their later phases as new spikes;
5 ms removes that, and longer gains nothing. We use **121** for this recording (the same 5 ms at
30 kHz would be 151). `nt0min` defaults to about a third of `nt`.

## Spatial extent of the templates

In EMUsort the channels sit on a dense line 2 µm apart (see the notes below), so the template widths
count channels: `min_template_size = 10` µm with `template_sizes = 5` tries widths of about 5, 10, 15,
20 and 25 channels. The paper found the wider ranges better because most MUAPs span at least 5
channels; lower `min_template_size` (e.g. 4 µm: about 2–12 channels) when MUAPs span fewer.

| Parameter | Recommendation (paper) |
|---|---|
| `nearest_chans` | the largest number of channels a MUAP spans (at most the channel count) |
| `nearest_templates` | at most the number of channels, for numerical stability |
| `whitening_range` | all channels of the array |
| `dminx` | not needed (one column) |

## Other settings

| Parameter | Recommendation (paper) |
|---|---|
| `nskip` (2) | at least 1 and at most half the number of batches; lower uses more MUAPs for the templates |
| `remove_channel_delays` (on), ±2 ms | longer conduction delays need a larger bound |
| `remove_spike_outliers` (on), `hdbscan_min_cluster_size` (20) | outliers (artefacts) out of the template learning |
| `n_templates`, `n_pcs` (9, 9) | richer MUAP shapes than Kilosort4's 6, 6 |
| `acg_threshold` (0.2), `ccg_threshold` (0.25) | raise `ccg_threshold` to merge more, lower `acg_threshold` to split more (here `clustering.refractory`, Rust only so far) |
| recording length | more than 2 minutes (over 200 spikes per unit); beyond 10–20 minutes MUAP shapes may drift |
| drift correction | off (`nblocks = 0`): fewer than 64 channels |

## Line-noise notch: off by default

EMUsort's paper filters with a 60 Hz notch. Here the notch is a setting (`do_notch`, `notch_hz`,
`notch_q`), **off by default**, because the band-pass already removes 60 Hz and the notch slows
every run.

**The band-pass already removes 60 Hz.** A Butterworth high-pass edge of order `N` at `fc` passes a
frequency `f` below it with gain `1/√(1 + (fc/f)^(2N))`; ours is order 3 at 300 Hz, applied forward
and backward (the gain squared):

| Frequency | Attenuation by the 300 Hz edge |
|---|---|
| 60 Hz (line) | 84 dB (≈ 16 000×) |
| 120 Hz (2nd harmonic) | 48 dB |
| 180 Hz (3rd) | 27 dB |
| 240 Hz (4th) | 14 dB |
| 300 Hz and above (5th, 6th, …) | pass, notch or not |

A 60 Hz notch removes only the fundamental, which is already gone; the harmonics that reach the band
(300, 360, … Hz) pass with or without it. If a recording shows line peaks there, a notch at those
frequencies (or a comb) is the tool, or a common reference (line noise is usually shared by all
channels). Kilosort4 does not notch either.

**Why the notch costs ~45% more time.** A narrow notch rings for a long time: at 60 Hz with `Q = 30`
(2 Hz wide) it settles over ~1 s. Every window of the recording is read with margins on both sides
long enough for its filters to settle, so the notch adds ~1 s on each side of every ~2.5 s batch
(each batch reads 1.83× its length instead of 1.01×), in each of the four passes over the recording
(fit, template clips, detection, matching). On the 200 s HD-EMG test segment (warm runs):

| | Run | Spikes | Units | Units with > 1% / > 5% ISIs < 2 ms |
|---|---|---|---|---|
| no notch (default) | 16.4 s, 12.2× real time | 129 817 | 66 | 28 / 9 |
| 60 Hz notch | 23.9 s, 8.4× real time | 125 254 | 64 | 29 / 10 |

**When to turn it on**: strong line pickup that survives the band (a lower band edge, e.g.
`bandpass_low_hz` near 100 Hz, or saturating 60 Hz). Set `notch_hz` to 50 outside the Americas;
a lower `notch_q` (a wider notch) settles faster.

## Notes on our implementation

Differences from the paper found while reading it, to decide before changing code (recorded in
`refactoring/GPU_TASKS.md`):

- **Channel map.** EMUsort replaces the array's geometry with a dense **linear** map, channels 2 µm
  apart, *"to make KS4 include more channels in the creation of templates"*. We use the physical
  geometry: on a 100 µm grid every spatial template (10–50 µm) collapses onto one contact. That makes
  the spatial parameters above behave differently here, and it is the root cause of the tied
  detections we found (now removed at detection). To implement: detection and clustering on the
  linear map, spike positions mapped back onto the real grid for export and inspection.
- **Filtering.** EMUsort band-passes 250–5000 Hz (`emg_passband`, SpikeInterface) and then Kilosort4
  high-passes again at 300 Hz inside its preprocessing. The two high-passes are redundant (the 300 Hz
  one sets the corner); only the 5000 Hz low-pass adds something. Planned: **one** band-pass,
  300–5000 Hz (the low-pass edge kept below Nyquist, or dropped at low sampling rates). Not
  bit-identical to upstream's cascade (a few Hz of corner, the skirt's steepness).
- **Notch.** EMUsort notches 60 Hz. After a 300 Hz high-pass (order 3, forward-backward) 60 Hz is
  already ~80 dB down, so the notch does almost nothing; line noise that reaches the spike band is its
  harmonics (300, 360, 420 Hz, …). Planned: a 60 Hz notch (50 Hz outside the Americas) as an option;
  a comb on the in-band harmonics only if a recording shows the problem.
- **`nt` and the sampling rate.** `nt` counts samples, so the same window needs a different `nt` at
  another rate. Proposed: an option to set the window in milliseconds, converted per recording
  (`round(ms · fs)`, made odd, `nt0min ≈ nt / 3`), alongside `nt`.
- **Composite score** for ranking sweeps and filtering units (`cluster_score_threshold`, 0.95
  recommended): not implemented.
