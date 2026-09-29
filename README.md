# dsp-kitchen

**High-Performance GPU-Accelerated Digital Signal Processing & Electrophysiology Engine in Rust**

`dsp-kitchen` is a modular, high-throughput DSP framework powered by [CubeCL](https://github.com/tracel-ai/cubecl) (WebGPU/Vulkan/Metal/CUDA) and SIMD hardware acceleration. It delivers deterministic in-VRAM signal filtering, out-of-core streaming, and classical electrophysiology spike sorting pipelines with zero-copy Python interoperability.

---

## Architecture Overview

`dsp-kitchen` is structured into 6 focused crates with strict domain separation:

| Crate | Purpose | Key Capabilities |
|:---|:---|:---|
| **`dsp-core`** | Domain-Agnostic Foundations | Rational sample clocks (`RationalTime`), `SensorLayout`, bitset `ChannelMask`, memory buffers |
| **`dsp-base`** | Deterministic DSP & Linear Algebra | IIR (Notch, Butterworth), FIR, Non-linear (9p Median, TKEO), `TemplateFilter`, PCA, in-VRAM `Pipeline` |
| **`dsp-stream`** | Continuous Streaming & Storage | Lock-free `MultiChannelRingBuffer`, Zarr v3 chunking, MinMax LOD decimation, QUIC/UDP transport |
| **`dsp-synapse`** | Electrophysiology & Spike Sorting | Neural frequency bands, Quiroga MAD noise thresholding, spatial deduplication, sinc realignment, snippets, ISI/SNR |
| **`dsp-bridge`** | Zero-Copy Python Interop | PyO3 bindings and C Buffer Protocol views mirroring Rust engines |
| **`dsp-cli`** | Headless CLI Harness | Multi-channel throughput sweeps ($1 \dots 1024$ channels), synthetic signal generation, hardware inspect |

---

## Installation & Building

### Prerequisites

1. **Rust Toolchain** (1.80+ recommended):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **Build Essentials** (Linux / macOS):
   ```bash
   # Debian / Ubuntu
   sudo apt-get install build-essential cmake pkg-config libssl-dev
   # Arch Linux
   sudo pacman -S base-devel cmake openssl
   # Fedora
   sudo dnf groupinstall "Development Tools"
   ```
3. **Python (Optional for Python bridge)**:
   Python 3.10+ (managed via `uv` or standard virtual environment).

---

### Building with Cargo

#### 1. Build the Entire Workspace in Release Mode
```bash
cargo build --release
```
This compiles all Rust crates (`dsp-core`, `dsp-base`, `dsp-stream`, `dsp-synapse`, `dsp-bridge`, `dsp-cli`) with full optimizations and GPU shader compilation.

#### 2. Run All Workspace Unit Tests
```bash
cargo test --workspace
```
Runs the complete test suite (27 unit and integration tests across DSP filters, PCA, ring buffers, QUIC transport, and spike detection).

#### 3. Install the Command-Line Tool (`dsp-cli`)
Install the headless `dsp-cli` binary directly into your Cargo binary path (`~/.cargo/bin`):
```bash
cargo install --path crates/dsp-cli
```
Or execute commands directly via Cargo:
```bash
# Run hardware launch geometry inspection
cargo run -p dsp-cli --release -- inspect

# Run dynamic 1..1024 multi-channel throughput sweep
cargo run -p dsp-cli --release -- benchmark --sweep

# Serve a dataset over encrypted QUIC transport (clock-paced real-time simulation)
cargo run -p dsp-cli --release -- serve --file playground/data/mearec_32ch_10s.bin

# In another terminal: Connect and benchmark latency, jitter, throughput & packet loss
cargo run -p dsp-cli --release -- receive --addr 127.0.0.1:50051 --duration 10.0

# Line-rate stress testing (unthrottled line-rate throughput)
cargo run -p dsp-cli --release -- serve --file playground/data/mock_signal_384ch.bin --no-realtime
cargo run -p dsp-cli --release -- receive --addr 127.0.0.1:50051 --max-frames 300
```


#### 4. Build and Install Python Bindings (`dsp-bridge`)

When building the Python native extension module:
```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Compile dsp-bridge targeting your active Python environment
PYO3_PYTHON=$(which python) cargo build -p dsp-bridge --release

# 3. Copy the compiled shared library into the Python package directory
cp target/release/lib_dsp_kitchen.so src/dsp_kitchen/_dsp_kitchen.so
```

Alternatively, install in editable mode using `uv` or `pip`:
```bash
uv pip install -e .
# or: pip install -e .
```

---

## Verification & Interactive Benchmarks

### 1. MEArec Ground-Truth Electrophysiology Evaluation
Benchmark the spike sorting pipeline against the 32-channel MEArec ground-truth dataset ($10\text{ s}$ @ $32\text{ kHz}$, 747 ground-truth spikes across 10 units):

```bash
uv run python playground/spike_sorting_evaluation.py
```
This script computes noise estimation, threshold crossings, spatial deduplication, sub-sample sinc realignment, multi-channel snippet cuts, and PCA feature projections, saving 6 diagnostic figures in `playground/figures/`:
* `01_detection_overlay.png`: Continuous traces with threshold and detected/ground-truth markers.
* `02_alignment_before_after.png`: Raw clock jitter vs sub-sample continuous sinc realignment.
* `03_multichannel_templates.png`: Spatial multi-channel footprints across $K=7$ nearest neighbors.
* `04_pca_feature_space.png`: 2D/3D PCA cluster separations color-coded by unit.
* `05_isi_distributions.png`: Refractory period ($1.5\text{ ms}$) histograms (0.0% violations).
* `06_accuracy_scorecard.png`: Executive dashboard showing **99.1% sensitivity** and timing error jitter.

---

## Quickstart Examples

### In-VRAM GPU Chained Pipeline (Rust)
```rust
use dsp_base::pipeline::{Pipeline, PipelineStage};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

let device = WgpuDevice::default();
let client = WgpuRuntime::client(&device);

// Zero host round-trips: Scale -> CAR -> Notch directly in GPU VRAM
let mut pipeline = Pipeline::new();
pipeline
    .add(PipelineStage::Scale { alpha: 0.195, beta: 0.0 })
    .add(PipelineStage::CommonAverageReference)
    .add(PipelineStage::Notch { freq_hz: 60.0, q: 30.0 });

let out_handle = pipeline.execute::<WgpuRuntime>(
    &client,
    &in_handle,
    384,     // channels
    30000,   // samples
    30000.0, // sample rate
    false,   // GPU mode
);
```

### Python Template Subtraction Filter
```python
import numpy as np
import dsp_kitchen.filter as flt

# Subtract recurring artifact waveforms with physical alignment & least-squares scaling
cleaned_signal = flt.subtract_template(
    data=raw_signal,              # 1D or 2D [channels, samples]
    template=artifact_template,   # Prototype artifact waveform
    event_indices=event_samples,  # Trigger timestamps
    center_offset=anchor_idx,     # Anchor point within template
    max_lag=8,                    # Cross-correlation lag search radius
    dynamic_scaling=True          # Least-squares amplitude matching
)
```

---

## License

Dual-licensed under either of:
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT License ([LICENSE-MIT](LICENSE-MIT))