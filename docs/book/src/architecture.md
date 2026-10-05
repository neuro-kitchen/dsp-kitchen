# Architecture

## Layers

```text
                 dsp-cli · dsp-app · dsp_kitchen_py            (entry points: choose a device)
                                  │
            dsp-synapse-ml ── dsp-synapse-hub                  (sorters / models · artifacts)
                                  │
                             dsp-synapse                       (spikes, units, sortings)
                                  │
                 ┌────────────────┴───────────────┐
              dsp-base                          dsp-io         (DSP primitives · files, probes)
                 └────────────────┬───────────────┘
                              dsp-core                         (time, buffers, recordings, compute)
```

Dependencies point down only. Each crate page states what the crate **owns** and what it **must
not contain**; for example, `dsp-core` knows nothing about neurons, `dsp-io` holds file layouts but
no algorithms, `dsp-base` holds generic DSP but no domain types, and `dsp-synapse-hub` is the only
crate with network access for model artifacts. `dsp-stream` (network transport of signals) sits
beside the stack and depends only on its own protocol types.

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
  literal; reusable buffers go through `Scratch`.
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
| `dsp_synapse::StreamingDetector` | filtered windows, detection, snippets, per-channel template moments | per-window candidates and template statistics |
| `dsp_synapse` noise calibration | filtered calibration chunks | one σ per channel |
| `dsp_synapse` dedup | the radius-neighbour table (built once per probe) | survival flags and participating-channel bitmasks |
| `dsp_synapse` GMM | features, responsibilities | component parameters and their sums per iteration |
| `dsp_synapse_ml` Kilosort4 detection | correlations, centre responses, scores | detected spikes and their features |

Reading is out of core: `dsp_io::PrefetchReader` streams halo windows of any `RecordingSource`
with the next one read on a background thread, so disk and decompression overlap device work and
host memory is bounded by two windows.

## Exactness across chunks

Windows overlap by **halos** sized from the pipeline's settling time, the snippet span, the
refractory period and the realignment margin
(`dsp_synapse::StreamingDetectionConfig::compute_halos`). Detection uses a locally exclusive
distance rule (`dsp_base::peaks::DistanceRule::LocallyExclusive`) that depends only on neighbours
within the halo, and deduplication finalizes spikes only once their neighbourhood is complete, so
any batch size gives the whole-recording result.
