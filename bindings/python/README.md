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

Development build (from this folder): `maturin develop --profile validate`.
