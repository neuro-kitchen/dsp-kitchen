# Summary

[Introduction](introduction.md)
[Architecture](architecture.md)

# Crates

- [dsp-core](crates/dsp-core.md)
- [dsp-io](crates/dsp-io.md)
- [dsp-base](crates/dsp-base.md)
- [dsp-synapse](crates/dsp-synapse.md)
- [dsp-synapse-ml](crates/dsp-synapse-ml.md)
- [dsp-synapse-hub](crates/dsp-synapse-hub.md)
- [dsp-stream](crates/dsp-stream.md)
- [dsp-view](crates/dsp-view.md)
- [dsp-app](crates/dsp-app.md)

# GPU engineering

- [Overview](gpu/index.md)
  - [Why matrix multiplication matters](gpu/matmul.md)
  - [Case study: covariance](gpu/covariance.md)
  - [Case study: HDBSCAN](gpu/hdbscan.md)
  - [Case study: exact median](gpu/selection.md)
  - [Case study: when the review was wrong](gpu/eigen.md)
  - [Case study: Kilosort4 detection, measured](gpu/detection.md)
  - [Case study: Kilosort4 clustering and matching](gpu/clustering.md)
  - [Moving data to the device](gpu/data-movement.md)
  - [Pitfalls](gpu/pitfalls.md)

# Sorters

- [Overview](sorters/index.md)
- [Kilosort4](sorters/kilosort4/intro.md)
  - [Pipeline](sorters/kilosort4/pipeline.md)
  - [Parameters](sorters/kilosort4/parameters.md)
  - [Tuning](sorters/kilosort4/tuning.md)
- [EMUsort](sorters/emusort/intro.md)
  - [Pipeline](sorters/emusort/pipeline.md)
  - [Parameters](sorters/emusort/parameters.md)
  - [Tuning](sorters/emusort/tuning.md)
- [MountainSort 5](sorters/mountainsort5/intro.md)
  - [Pipeline](sorters/mountainsort5/pipeline.md)
  - [Parameters](sorters/mountainsort5/parameters.md)
  - [Tuning](sorters/mountainsort5/tuning.md)
- [Benchmarks](sorters/benchmarks.md)

# Guides

- [Adding a format](guides/adding-a-format.md)
- [Adding a sorter](guides/adding-a-sorter.md)

# Reference

- [Citations](reference/citations.md)
