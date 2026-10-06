# Architecture

## Layers

```text
            dsp-cli · dsp-app · dsp_kitchen_py                 (entry points: choose a device)
                 │                          │
   dsp-synapse-ml ── dsp-synapse-hub    dsp-stream             (sorters, artifacts │ network)
                 │                          │
            dsp-synapse                  dsp-view              (spikes, units │ envelopes, pyramids)
                 │                          │
        ┌────────┴────────┐                 │
     dsp-base           dsp-io              │                  (DSP primitives · files, probes)
        └────────┬────────┴─────────────────┘
             dsp-core                                          (time, buffers, recordings,
                                                                window streaming, compute)
```

Dependencies point down only (dsp-stream uses dsp-view for the envelopes it serves; both depend
on dsp-core alone). Each crate page states what the crate **owns** and what it
**must not contain**; for example, `dsp-core` knows nothing about neurons, `dsp-io` holds file
layouts but no algorithms, `dsp-base` holds generic DSP but no domain types, `dsp-view` holds all
viewing code (host and device) and no rendering, and `dsp-synapse-hub` is the only crate with
network access for model artifacts.

## Compute

### Choosing a runtime

```rust,ignore
use dsp_core::compute::{ComputeTarget, ComputeTask};
use cubecl::prelude::*;

struct Filter<'a> { data: &'a [f32] }

impl ComputeTask for Filter<'_> {
    type Output = Vec<f32>;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Vec<f32> {
        // any dsp-base / dsp-synapse algorithm, generic over R
        # unimplemented!()
    }
}

let target = ComputeTarget::from_env()?;   // DSP_KITCHEN_RUNTIME, else the first compiled-in
let out = target.run(Filter { data: &samples })?;
```

- Runtimes are Cargo features of `dsp-core` (`wgpu`, `cuda`, `hip`, `cpu`); `available()` lists the
  compiled-in ones, GPUs first.
- **Libraries never call `from_env`.** Every algorithm takes a `ComputeClient<R>` (or, for the few
  host-orchestrated APIs, an explicit `ComputeTarget`). This keeps one device per run and lets the
  caller decide.

### Kernels

- Generic over the float type: kernels take `F: Float`, dispatchers `<R: Runtime, F: DspFloat>`.
- Buffers are typed (`dsp_base::core::buffer`): sizes come from the element type, never a byte
  literal; reusable buffers go through `Scratch` or are kept by the struct that uses them;
  `download_prefix` / `download_range` read only the part of a buffer in use.
- Launch geometry from the runtime (`LaunchGeometry::{elementwise, channels_samples, per_sample,
  per_channel, per_row}`); shared-memory paths are chosen from the device's limits, not its type.
- Stencils take an explicit `EdgeMode` with scipy defaults.
- Device-dependent choices are CubeCL autotune candidates keyed by device, element type, shape and
  size class.

## Data movement

| Where | What stays on the device | What crosses the bus |
|---|---|---|
| `dsp_base::PipelineWorkspace` | every stage's input / output (ping-pong buffers), filter states, spatial operators | the raw chunk in (int16 as stored), the result out only when asked |
| `dsp_base::peaks::find_peak_candidates` | the trace, per-block counts and offsets | the candidates (sample, value) |
| `dsp_base::peaks::find_peak_candidates_on_device` | the trace and the candidates (sample, value, channel) | the per-channel counts |
| `dsp_base::linalg::SecondMomentAccumulator` | every batch's contribution, partial sums | the `[C, C]` result once |
| `dsp_synapse::StreamingDetector` | filtered windows, detection, snippets, per-channel template moments | per-window candidates and template statistics |
| `dsp_synapse` noise calibration | filtered calibration chunks | one σ per channel |
| `dsp_synapse` dedup | the radius-neighbour table (built once per probe) | survival flags and participating-channel bitmasks |
| `dsp_synapse` GMM | features, responsibilities | component parameters and their sums per iteration |
| `dsp_synapse_ml` Kilosort4 detection (`UniversalDetector`) | templates and centre tables (uploaded once), correlations, centre responses, scores, candidates | per window: candidate counts, then the spikes and their features |
| `dsp_synapse_ml` EMUsort delays | envelopes, the cross-correlation sum | the cross-correlation once |
| `dsp_synapse_ml` universal templates | the scaled clips (uploaded once), Gram matrix, gathered inliers | eigenvectors, HDBSCAN labels, k-means sums per iteration |
| `dsp_synapse` k-means, HDBSCAN (`DevicePoints`) | the points, distances, assignments, core distances | `k · d` sums per iteration; one weight block per k-means++ draw; `O(n)` cheapest edges per Borůvka round |

Reading is out of core: `dsp_core::WindowLoader` streams halo windows of any `RecordingSource`
with the next one read on a background thread, so disk and decompression overlap device work and
host memory is bounded by two windows (see *Orchestration* in the dsp-core page).

**Rule: a window crosses the bus at most once each way.** The raw window goes up once; every
stage after that reads device buffers; only small results come down (σ per channel, spikes,
clips, statistics accumulated over windows such as `SecondMomentAccumulator`). Never download a
window to upload it again for the next stage: add a device entry point (a column range, an
accumulator) instead. Constant data (templates, tables, thresholds) is uploaded once per run, not
per window, and scratch buffers are kept. When a buffer is sized for the longest window, read only
the part in use (`buffer::download_prefix`).

## Exactness across chunks

Windows overlap by **halos** sized from the pipeline's settling time, the snippet span, the
refractory period and the realignment margin
(`dsp_synapse::StreamingDetectionConfig::compute_halos`). Detection uses a locally exclusive
distance rule (`dsp_base::peaks::DistanceRule::LocallyExclusive`) that depends only on neighbours
within the halo, and deduplication finalizes spikes only once their neighbourhood is complete, so
any batch size gives the whole-recording result.
