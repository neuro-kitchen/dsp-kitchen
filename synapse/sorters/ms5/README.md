# dspk-ms5

**MountainSort 5** (Chung et al., Neuron 2017), ported from `flatironinstitute/mountainsort5` to
GPU kernels through CubeCL: any GPU (wgpu, CUDA) or the CPU.

- Stages, named as upstream's: `detect/`, `snippets/`, `scheme1/` (clustering, isosplit6
  subdivision), `scheme2/` (classifiers), `templates/`, driven by `runner/`.
- **Licence: Apache-2.0** (`LICENSE`), like upstream; attribution in `NOTICE.md`.
