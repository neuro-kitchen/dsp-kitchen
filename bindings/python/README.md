# dspk (Python)

> [!WARNING]
> **Early development.** The bindings are being rewritten; APIs change.

Python bindings of [dspk](https://github.com/neuro-kitchen/dsp-kitchen): multi-channel signal
processing written once in CubeCL, run on any GPU (wgpu, CUDA, HIP) or the CPU, with the device
chosen at run time.

```python
import dspk

dspk.runtime.available()   # e.g. ['wgpu', 'cuda', 'cpu']
dspk.runtime.set("cuda")   # every later call runs on CUDA unless given runtime=
```

## Development

From this folder, with the repository's environment (`uv sync` at the root):

```bash
uv run --project ../.. maturin develop --profile validate   # build and install dspk
uv run --project ../.. python -m pytest tests                # tests (scipy references)
```

**Stubs.** `python/dspk/_native/**/*.pyi` are generated from the Rust bindings; regenerate them
after changing a binding (`tests/test_stubs.py` fails when they disagree with the extension):

```bash
LIB=$(../../.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LD_LIBRARY_PATH=$LIB PYO3_PYTHON=../../.venv/bin/python cargo run --bin stub_gen
```

The generator links `libpython`; `LD_LIBRARY_PATH` points the loader at uv's Python.
