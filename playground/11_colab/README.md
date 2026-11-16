# 11 · In Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuro-kitchen/dsp-kitchen/blob/main/playground/11_colab/dspk_neuropixels.ipynb)

`dspk_neuropixels.ipynb` needs nothing installed on your computer: it installs `dspk` from PyPI,
downloads Kilosort4's test recording (Neuropixels 1.0, 385 channels, 30 kHz, 45 s; 1 GB), cleans
a second of it on the GPU, sorts it with dspk's Kilosort4 and plots the units.

Choose **Runtime → Change runtime type → T4 GPU** first. Without a GPU it runs on the CPU, slower.

It was run from start to end with `dspk` 0.0.10 from PyPI on a laptop (RTX 2070 Max-Q, CUDA: the
sort took 61 s). The Linux wheels are `manylinux_2_34`, which Colab's Ubuntu 22.04 (glibc 2.35)
accepts.
