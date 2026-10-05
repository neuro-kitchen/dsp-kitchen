# Electrophysiology & Spike Sorter Guides (`docs/sorters/`)

This directory contains technical guides, algorithmic specifications, implementation walkthroughs, and parameter optimization manuals for high-density spike sorters integrated into the `dsp-kitchen` ecosystem (`dsp-synapse` and `dsp-synapse-ml`).

---

## Supported Sorters

### 1. [Kilosort4](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/kilosort4/01_intro.md)
* **Domain**: High-density extracellular neural electrophysiology (Neuropixels 1.0/2.0, Utah arrays, tetrodes).
* **Architecture**: Universal matched-filter template matching (`wTEMP`), non-rigid drift tracking, graph-based clustering (subsampled nearest neighbors via `faiss`), temporal PCA basis projection (`wPCA`), and convolutional matching pursuit deconvolution.
* **Pages**:
  - [01. Introduction & Academic Citations](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/kilosort4/01_intro.md)
  - [02. Step-by-Step Pipeline Guide (`py` & `rs`)](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/kilosort4/02_pipeline_guide.md)
  - [03. Parameter Optimization Manual](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/kilosort4/03_parameter_optimization.md)

---

### 2. [EMUsort (Myomatrix)](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/emusort/01_intro.md)
* **Domain**: High-definition intramuscular and surface electromyography (Myomatrix 32-channel grids, 8-channel threads, 64-channel arrays).
* **Architecture**: 150-sample universal Motor Unit Action Potential (MUAP) matched filtering, multi-contact conduction latency alignment, 12-PC spatiotemporal muscle basis embedding (`wPCA_EMG`), and contact-localized unit formation.
* **Pages**:
  - [01. Introduction & Academic Citations](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/emusort/01_intro.md)
  - [02. Step-by-Step Pipeline Guide (`py` & `rs`)](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/emusort/02_pipeline_guide.md)
  - [03. Parameter Optimization Manual](file:///home/yezuss1/dev/rustProjects/neuro-kitchen/dsp-kitchen/docs/sorters/emusort/03_parameter_optimization.md)
