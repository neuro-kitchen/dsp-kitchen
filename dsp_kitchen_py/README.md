# `dsp_kitchen_py` — Python SDK & PyO3 Native Bindings

This directory contains the Rerun-style Python SDK and PyO3 native extension for **`dsp-kitchen`**:

- **`src/`**: Rust PyO3 native module (`dsp_kitchen_bindings`) bridging `dsp-core`, `dsp-base`, `dsp-io`, `dsp-stream`, `dsp-synapse`, and `dsp-synapse-ml` (including the Burn-ONNX `onnx-ir` runtime).
- **`dsp_kitchen_bindings/`**: Typed stub package (`dsp_kitchen_bindings.pyi`) for the compiled Rust `.so` / `.pyd` library.
- **`dsp_kitchen_sdk/dsp_kitchen/`**: Pure-Python SDK (`import dsp_kitchen as dk`) with subpackages `filter`, `spatial`, `math`, `linalg`, `pipeline`, `io`, and `synapse` (`synapse.ml`, `synapse.onnx`).
- **`tests/`**: Python test suite verifying filters, pipelines, spike sorting, and `dsp-synapse-ml` / Burn-ONNX inference.
