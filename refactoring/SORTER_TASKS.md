# Sorter tasks: what is missing from Kilosort4 and EMUsort (2026-10-08)

Status: **not started** — wait for the user's go-ahead per task. Written from the papers (Kilosort4:
Pachitariu et al., *Nature Methods* 2024, Methods; EMUsort: O'Connell et al., *eLife* 2026, RP110417,
Table 5 and Methods); the upstream code is GPL-3.0 and is not ported. Same working rules as
`GPU_TASKS.md`: GPU first (count every host↔device crossing), dtype preserved, no random constants
(seeds and thresholds are named settings), reproducible by default, tests that compare results, timeouts
≤ 5 min, each step written up in the mdBook (sorter pages, *Benchmarks*, GPU case study).

| # | Task | Sorters | Decision | Depends on |
|---|------|---------|----------|------------|
| S1 | Auto- and cross-correlograms with the paper's refractory test | both | to do (building block) | — |
| S2 | Refractory criterion in the clustering splits | both | to do | S1 |
| S3 | Global merges (stage 5) | both | to do | S1 |
| S4 | Duplicate-spike removal | both | to do | — |
| S5 | Unit labels (*good* / *mua*) | both | to do | S1 |
| S6 | EMUsort filtering: one 300–5000 Hz band-pass + optional notch | EMUsort | **decided** (user) | — |
| S7 | Template window in milliseconds (`nt` from `fs`) | both | **decided**: keep `nt = 121` for the HD-EMG data; ms option proposed | — |
| S8 | EMUsort linear channel map (2 µm) | EMUsort | **to decide** | — |
| S9 | EMUsort composite score | EMUsort | to do | S1 (S5) |
| S10 | Drift correction (stage 0) | Kilosort4 | later (not needed for the test data) | — |
| S11 | EMUsort citation → eLife reviewed preprint | EMUsort | to do (small) | — |

**Rule for every task (user, 2026-10-08):** each new parameter is exposed to the user with its
default visible before they run anything: a field of the Rust config with the value in `Default`, and a
keyword argument with that default in the Python signature and docstring (units included).

Suggested order: S1 → S2, S3, S5 → S4 → S6, S7 → S8 → S9 → S11; S10 when a recording needs it.

---

## S1. Auto- and cross-correlograms with the refractory test

**What (Kilosort4 paper, *Refractory auto- and cross-correlograms*).** ACG / CCG of spike times in 1 ms
bins from −0.5 s to +0.5 s. With `n_k` the coincidences in the central `−k..+k` bins and `R` the baseline
rate (the larger of the left and right shoulders, to allow asymmetric CCGs):
- `R12 = min_k n_k / ((2k + 1)·R)` (refractory coincidences vs expected);
- `Q12 = min_k P_k`, `P_k = ½ (1 + erf((n_k − λ_k) / (ε + 2λ_k)^½))`, `λ_k = (2k + 1)·R`, `ε = 10⁻¹⁰`
  (Gaussian approximation of the Poisson probability of seeing `n_k` or fewer).
- **Refractory CCG**: `R12 < ccg_threshold (0.25)` and `Q12 < 0.05`. **Refractory ACG**:
  `R12 < acg_threshold (0.2)` and `Q12 < 0.2`.

**Where.** A reusable module (dsp-synapse `metrics` or `sorting`), on spike-time arrays. Device: many
pairs at once (merges test hundreds of pairs); histogram of time differences per pair with integer
atomics; the paper's statistics on the host from the `1001` bins. Check what the existing quality
metrics (`isi_violation_ratio`, CCG code if any) already provide before writing new code.

**Done when.** Unit tests: Poisson trains are not refractory; trains with a dead time are; two halves
of one refractory train give a refractory CCG; thresholds named in config (`acg_threshold`,
`ccg_threshold`, the 0.05 / 0.2 probability limits).

## S2. Refractory criterion in the clustering splits

**What (paper, *Split/merge criteria*).** Walking the merging tree from the top: if the two halves'
CCG is refractory, **never split** (one neuron); otherwise split iff the regression axis is bimodal (as
now), and always split below modularity 0.2. Used in the clustering of the **matched** spikes (the
first clustering uses only bimodality, as the paper says).

**Where.** `kilosort4/clustering.rs` `decide`: the split callback gets the halves' spike times.

**Done when.** Test: a tree whose bimodal halves come from one refractory train is not split. Against
Kilosort4's saved results: the "not split" and accuracy columns of `kilosort4_validation.py` improve or
hold; EMUsort: fewer units with > 1% ISIs < 2 ms.

## S3. Global merges (stage 5)

**What (paper, *Global merges*).** Units sorted by spike count, largest first; for each, every other unit
with waveform similarity (maximum correlation over lags) above 0.5, from most to least similar; merge if
the CCG is refractory; re-test the merged unit; when nothing merges it is complete and leaves the pool.

**Where.** After the clustering of the matched spikes (runner stage after `STAGE_RECLUSTERING`).
Similarities of all pairs at all lags: reuse the PC-space `matmul` + `pair_similarity_kernel` of
`learned.rs`. Merged templates: spike-count-weighted means (and spike times shifted by the best lag, to
check against the paper).

**Done when.** Test: two halves of one unit (same template, refractory CCG) merge; two units with
similar templates but independent firing do not. Validation: Kilosort4 unit count closer to 267, "not
split" and accuracy up; book updated (pipeline stage 8 → implemented, Benchmarks re-measured).

## S4. Duplicate-spike removal

**What.** Spikes of one unit closer than the duplicate window are artefacts of matching pursuit (a
template re-matching its own residual): keep one. Kilosort4: `duplicate_spike_ms` (0.25 ms); EMUsort
Table 5: `duplicate_spike_bins = 7` (samples); EMUsort config template: `duplicate_spike_ms = 0.5`.
Settle which to use per sorter, in samples derived from `fs`.

**Where.** Final step of the runner, per unit (host: spikes are on the host already).

**Done when.** Test with planted duplicates; EMUsort check: ISIs < 0.5 ms drop to ~0.

## S5. Unit labels

**What.** A unit is *good* when its ACG is refractory (S1, `acg_threshold`), else *mua*; exported as
`cluster_KSLabel` / `cluster_group` in the Phy folder and as `quality_label` in `SortingOutput`.

**Done when.** Labels in the Phy export match the format Kilosort4 writes; validation compares our good
units with Kilosort4's 127.

## S6. EMUsort filtering (decided)

**What (user decision, 2026-10-08).** One band-pass **300–5000 Hz** instead of upstream's cascade
(SpikeInterface 250–5000 Hz, then Kilosort4's own 300 Hz high-pass: the two high-passes are redundant).
The 5000 Hz edge stays below Nyquist; drop the low-pass when it cannot (low sampling rates).
**Notch** at 60 Hz as an option (`notch_hz`, 50 outside the Americas), default as the paper (60). Note:
after the 300 Hz high-pass 60 Hz is ~80 dB down; in-band harmonics would need a comb, only if data
shows line noise.

**Where.** `EmusortConfig` (`emg_passband`, `notch_hz`), the fitted preprocessing pipeline.

**User-adjustable (user, 2026-10-08).** Both band edges and the notch are settings the user can
change, not constants: e.g. `low_hz: float = 300.0`, `high_hz: float = 5000.0`, `notch_hz: float | None
= 60.0`, shown with their defaults in the Python signature (as `Kilosort4Config(*, nt: int = 61, …)`)
and in the Rust `Default`. Same rule for Kilosort4's filtering (`highpass_cutoff_hz`, already a setting).

**Done when.** The filter response is checked (scipy fixture as the other filters); EMUsort checks
re-run; book (EMUsort pipeline, parameters, tuning notes) updated.

## S7. Template window in milliseconds (decided: keep 121)

**What.** `nt` counts samples (5 ms is 121 at 24.4 kHz, 151 at 30 kHz). Add an optional window in ms,
converted per recording (`round(ms · fs)`, made odd, `nt0min ≈ nt / 3`), alongside `nt` (Kilosort4-
compatible configs keep working). Library default stays `nt = 61`; the HD-EMG script keeps 121.

**Done when.** Config test (odd, `nt0min` follows); docs (tuning pages) updated.

## S8. EMUsort linear channel map (to decide)

**What (EMUsort paper, *Automatic channel map creation and adjustment*).** EMUsort replaces the array's
geometry with a dense **linear** map, channels 2 µm apart, so Kilosort4's spatial templates (10–50 µm)
span ~5–25 channels. We use the physical geometry: on the 100 µm HD-EMG grid every template collapses
onto one contact (the root cause of the tied detections, now removed at detection).

**Plan if accepted.** Detection and clustering on the linear map (channel `i` at `(0, 2·i)` µm, in
recording order); spike positions mapped back onto the physical grid (interpolating between channels)
for the `SortingOutput`, exports and the inspector. Measure before / after on the 200 s segment
(units, refractory violations, template truncation, tied detections without the tie fix).

## S9. EMUsort composite score

**What (EMUsort paper, *Producing composite scores for agnostic estimation of sort quality*).** Per
cluster, the product of four components: type I (1 − Llobet contamination ratio), type II (presence
ratio and amplitude cutoff), firing-rate validity, SNR (sigmoidal); the sort's score is their mean.
Used to filter units (`cluster_score_threshold`, 0.95 recommended, 0 = off) and to rank parameter
sweeps. Read the formulas in the paper (Methods, pages 25–26) before writing.

**Done when.** Per-unit scores in `SortingOutput` metrics and the Phy tables; tests on synthetic units;
the EMUsort checks script reports them.

## S10. Drift correction (later)

Kilosort4's stage 0 (rigid / non-rigid motion from spike positions, data corrected before detection).
Off in EMUsort by design (`nblocks = 0`, fewer than 64 channels). On the Neuropixels test file Kilosort4's
own correction is ±0.5 µm, so it changes nothing there. dsp-synapse has drift estimation and kriging
that a driver can use. Do when a recording needs it.

## S11. EMUsort citation

Provenance (`emusort_provenance`) and the book cite the bioRxiv DOI `10.64898/2026.01.06.697952`; the
paper is now an eLife reviewed preprint: O'Connell et al. 2026, *eLife* 15:RP110417,
doi:10.7554/eLife.110417.1. Update both (keep the bioRxiv DOI as the preprint).
