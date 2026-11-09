---
title: Add an operation
description: "From a kernel to the pipeline, Python and the app."
sidebar:
  order: 3
---

An operation has one pattern, from kernel to every tool. Each step is small, and skipping one leaves a
tool without the operation.

| Step | Where | What |
|---|---|---|
| 1. Kernel | `crates/dspk-ops/src/<family>/kernels/` | The CubeCL `#[cube]` kernel, beside the code that launches it |
| 2. Function | `crates/dspk-ops/src/<family>/` | `execute_<name>`: data first with its shape, then `sample_rate`, then one settings struct |
| 3. Stage | `crates/dspk-ops/src/pipeline/stage.rs` | A `PipelineStage` variant, with its `settling` (context it needs) |
| 4. Spec node | `engine/dspk-pipeline/src/spec.rs` | A `Node` variant with visible defaults, mapped in `prepare.rs` |
| 5. Python | `bindings/python/` | The native function, the `dspk.ops.<family>` and `dspk.ops.stage` wrappers, a numpy-style docstring |
| 6. App | `app/dspk-app-core/src/screens/pipeline/catalog.rs` | A catalog entry, with presets |
| 7. Docs | `docs/` | Regenerate the references; add it to [Operations](/dsp-kitchen/operations/) |

## Rules

- **Test against a host reference** on every runtime, with odd sizes.
- **Every tunable value is a setting** with a visible default, not a buried constant.
- **Keep the stored dtype** to the device, and convert in the kernel.
- **No loop on a flag** set inside it. Bound every loop by a counter or a compile-time constant
  ([why](/dsp-kitchen/gpu/pitfalls/)).
- **Several exact implementations?** Let the autotuner choose, and test each against the scalar version.

After changing bindings, regenerate the stubs:

```sh
PY=$(realpath .venv/bin/python)
LIB=$($PY -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
cd bindings/python && LD_LIBRARY_PATH=$LIB PYO3_PYTHON=$PY cargo run --profile validate --bin stub_gen
```

Then the generated pages:

```sh
uv run --no-project --with griffe==2.4.1 python -I docs/scripts/python_api.py
python3 docs/scripts/cli_reference.py    # needs dspk and dspk-cli-sorter built
```
