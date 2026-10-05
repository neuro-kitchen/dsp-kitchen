# dsp-synapse review notes (working file)

Started 2026-10-05 on `versions/v0.14`. Review only — no code changes yet. Each finding records the
file and line so it can be verified before acting.

## Inventory

- ~12.5k lines, 11 modules: core, detection, extraction, features, metrics, probe, sorting, spatial,
  storage, streaming (+ lib.rs), 3 integration tests + a SpikeInterface reference script.
- Dependencies: dsp-core (compute), dsp-base, **dsp-stream (not reviewed yet)**, cubecl, serde,
  serde_json, thiserror, tracing, **zarrs (direct)**.

## Findings

### Manifest
- M1 `Cargo.toml`: `cubecl = { features = ["wgpu"] }` forces WGPU on; there is no `wgpu` feature, so a
  CPU-only / CUDA-only build still compiles WGPU. Inconsistent with dsp-core / dsp-base feature model.
- M2 `zarrs` used directly (storage); dsp-io's `container::zarr` now owns this (phase 2 moved
  `zarr_store.rs` there already).
- M3 No dependency on `dsp-io` — needed for probe geometry (`SensorLayout` now in `dsp_io::neuro::probe`)
  and containers.

### lib.rs
- L1 Compatibility aliases: `pub use core::{bands, traits}`, `metrics::correlogram`, `sorting as
  clustering`, `sorting as matching`, `spatial as localization`, `spatial as motion`, plus a merged
  `pub mod kernels` re-exporting four kernel modules. Same pattern removed from dsp-core. Check who
  uses the alias paths before removing.
- L2 ~70 flat root re-exports; makes the module structure irrelevant to callers.

### core/ (domain types)
- C1 `traits.rs`: stage traits (`SpikeDetector`, `SpikeMatcher`, …) take host `&[f32]`; no device
  buffer path. `PeakLocalizer` uses `dsp_core::SensorLayout` (moved → `dsp_io::neuro::probe`).
- C2 `events.rs`: `peak_amplitude_uv` naming — neuro layer may keep µV, but it should come from the
  recording's `SignalUnit`, not be assumed.
- C3 `snippets.rs`: two representations (`WaveformSnippet` per spike with own `Vec`s, contiguous
  `SnippetBatch`) with copying conversions. `from_snippets` silently drops mismatched snippets.
- C4 `template.rs`:
  - `compute_mean_template` host-only; `streaming/kernels/template_reduce.rs` reduces templates on
    the device → probable duplicate (verify).
  - `DenseTemplates` pack/unpack of Phy `[n, samples, channels]` vs Zarr/NWB `[n, channels, samples]`
    is file-schema work (belongs with dsp-io phase 3).
  - `unpack_unit` invents `std = 1.0` when the file has none (hidden default).
  - `UnitQualityLabel::{Good, Mua}` consts alias the variants (alias cruft).
  - Imports `dsp_base::resampler::minmax::peak_to_peak` (parked) → broken.
- C5 `sorting_output.rs`:
  - `SortedUnit` couples data with metrics: constructors compute ISI / presence / amplitude-cutoff /
    SNR / label with hidden constants: presence bin `(duration/10).clamp(0.5, 60)` s, noise floor
    `sd.max(1e-3)`, fallback noise `10.0` µV when a channel sigma is missing
    (`from_clustered_spikes`).
  - `from_motor_units`: overwrites `snr` with `pnr_db` (dB in a ratio field), labels with hard-coded
    `20.0` dB / CoV `0.35`, `primary_channel = 0`, amplitude fallback `1.0`, noise `1.0`.
  - `flattened_spikes`: amplitude fallback `1.0`, location fallback `[0,0,0]` (fabricated values).
  - Time as `u64` samples + `f64` rate; core now has `SampleRate` / `RationalTime`.
  - NWB ragged pack/unpack (`to_ragged_spikes`, `unpack_ragged_spikes`) and Phy grouping
    (`flattened_spikes`, `group_spikes_by_cluster`) are file-schema helpers → dsp-io phase 3.
  - `probe: Option<SensorLayout>` from dsp_core (moved).

### detection/
- D1 **Duplicated peak picking**: the "local extremum past threshold, then skip refractory" loop is
  written 5 times (`threshold.rs`, `adaptive.rs`, `neo.rs`, `matched_filter.rs`, and the host half of
  `kernels/threshold.rs`). All greedy (first extremum wins, not the largest within the refractory
  window as SpikeInterface `locally_exclusive`).
- D2 **BUG** `neo.rs:57`, `matched_filter.rs:90`: refractory test `events.is_empty() || t > last_spike +
  refractory` — `last_spike` resets per channel but `events` is shared across channels, so from the
  second channel on, a spike within the first `refractory_samples` samples is dropped.
- D3 Host vs device: `threshold.rs` (host, 3 polarities) and `kernels/threshold.rs` (device count →
  scan → compact, autotuned; **negative polarity only**, **f32 only**, raw `client.empty(n *
  size_of)` instead of dsp-base `core::buffer`). Adaptive / NEO / matched filter are host-only.
- D4 **Reimplemented elsewhere**:
  - `compute_neo_energy_1d` = dsp-base `execute_teager_kaiser` (host copy, with a `max(0)` floor
    and copied end samples).
  - matched filter correlation = FIR with reversed taps (dsp-base `execute_fir`); host `O(S·K)` loop;
    last `k_len` samples of `corr` never written (stay 0).
  - `noise.rs` is only a re-export of dsp-base stats (and `mod.rs` re-exports it again).
- D5 Threshold statistics: NEO / matched filter threshold on `estimate_noise_std(psi)` /
  `(corr)` — the MAD-σ rule assumes zero-mean Gaussian samples; ψ ≥ 0, so this is a heuristic scale,
  not a noise σ (document or replace).
- D6 Hidden constants: defaults 4.5 σ / 1 ms / 8.0 (NEO) as literals in `Default` impls; adaptive
  `block_samples.clamp(32, …)`, `max(1e-6)`, `alpha.clamp(0.01, 1.0)`; matched filter prototype
  `0.38`, `len/10`, `len/6`, `max(9)`, `1e-8`.
- D7 `SpikeEvent.peak_amplitude_uv` filled with whatever unit the trace is in.

### detection/dedup
- DD1 Design is sound: exact "locally exclusive" rule, order-independent, streaming form
  (`StreamingDedup`) equal to the whole-recording result; well tested.
- DD2 **BUG (polarity)** `dedup.rs` `deeper()` and `kernels/dedup.rs`: "deeper" = smaller amplitude,
  i.e. assumes negative troughs. For `SpikePolarity::Positive` / `Both` crossings the *smaller*
  positive peak survives. Needs a magnitude / polarity-aware comparison.
- DD3 Device version (`deduplicate_spikes_spatial_gpu`): builds a dense `C×C` distance matrix on the
  host on every call, re-derives `participating_channels` on the host anyway (same `O(n·window)` work
  as the CPU path), and is not chained to device detection (detection downloads events, dedup
  uploads them again). f32 only; raw `client.empty(n * 4)`.
- DD4 Uses `dsp_core::layout::{SensorLayout, Position3D}` (moved to dsp-io).
- DD5 Tests hard-code `WgpuRuntime` without a feature gate.

### extraction/
- E1 **Three windowed-sinc interpolators**: `alignment::interpolate_window` (normalized fixed
  weights, taps must stay in the row), `alignment::resample_sinc_1d` (renormalizes truncated windows
  at edges → different edge behaviour), and the device kernel `kernels/sinc.rs` (same formula as
  `interpolate_window`). The device kernel re-implements Blackman-Harris and sinc inline with literal
  coefficients (`0.35875…`, `1e-7`), duplicating `dsp_base::math::windows` (host only today).
- E2 Device kernel inefficiency: one unit per output value recomputes the parabolic offset and all
  `2r+1` sinc/window weights (3 `cos` + 1 `sin` per tap) — weights depend only on the spike's shift,
  so they could be computed once per spike.
- E3 Host extraction (`extract_snippet_batch_multichannel`) duplicates the device extractor; KNN
  table recomputed per call; `extract_snippets_multichannel` = batch + `to_snippets()` copy;
  `_channels` parameters unused.
- E4 `read_snippets` reads through `RecordingSource` (good), clones the buffer per snippet.
- E5 Hidden constants: `1e-6` (parabolic denominator), `1e-4` / `1e-5` (shift skip thresholds — two
  different values for the same decision), `1e-7`.
- E6 f32 only; `spike_center_samples` as `u32` (window-local, OK while chunks < 4 G samples).
