# Notice

`dspk-ms5` is a port of **MountainSort 5** to Rust and GPU kernels (CubeCL).

- Upstream: <https://github.com/flatironinstitute/mountainsort5>, v0.5.9, by Jeremy Magland
  (Flatiron Institute), licensed under the Apache License 2.0.
- Clustering: isosplit6, <https://github.com/magland/isosplit6>, by Jeremy Magland, Apache License
  2.0 (ported in `dspk-synapse`, `sorting::isosplit6`).
- Paper: Chung, Magland, Barnett et al., "A fully automated approach to spike sorting", Neuron 2017.

## Changes from upstream

The stages (`detect`, `snippets`, `scheme1` with the isosplit6 subdivision, `scheme2`
classifiers, `templates`) follow upstream's `core/` and `schemes/` modules; the computation is
rewritten for the GPU (CubeCL kernels next to each stage), and recording windows are streamed to
the device through `dspk-sort`. Defaults are upstream's; differences are listed in each module.

This port is distributed under the Apache License 2.0 (`LICENSE`), the same as upstream.
