---
title: The published image
description: "ghcr.io/neuro-kitchen/dspk: the command line, the server and every sorter, nothing to build."
sidebar:
  order: 0
---

From v0.0.11 every release publishes an image, so nothing has to be built:

```sh
podman pull ghcr.io/neuro-kitchen/dspk:latest      # or a version: ghcr.io/neuro-kitchen/dspk:0.0.11
podman run --rm ghcr.io/neuro-kitchen/dspk doctor   # which runtimes work here
```

It holds the `dspk` command line, the server (`dspk serve`) and every sorter (`dspk-cli-sorter`,
GPL-3.0), on Debian slim, run as a non-root user. It has no desktop app and no recording. The
runtimes in it are wgpu, on any vendor's GPU through Vulkan, and the CPU.

## Give it the GPU

```sh
# NVIDIA (the host's driver, through the NVIDIA Container Toolkit)
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable ghcr.io/neuro-kitchen/dspk doctor
# AMD, Intel
podman run --rm --device /dev/dri --group-add keep-groups --security-opt=label=disable ghcr.io/neuro-kitchen/dspk doctor
```

## Work on your data

Mount the folder that holds the recording:

```sh
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable -v ./data:/data:Z \
  ghcr.io/neuro-kitchen/dspk sort kilosort4 /data/rec.bin --probe neuropixels1 -o /data/rec.kilosort4.sorting.zarr
```

To serve recordings to the app or to Python, see [run the server](/dsp-kitchen/deploy/run/).

## CUDA

The image holds no NVIDIA software. CUDA is compiled into `dspk` and works once NVRTC, CUDA's
run-time compiler, is in the image; the driver does not bring it. Either
[build the `nvidia` target](/dsp-kitchen/deploy/build/), or add NVRTC on top of the published image:

```dockerfile
FROM ghcr.io/neuro-kitchen/dspk:latest
USER root
ADD https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends cuda-nvrtc-13-0 \
    && rm -rf /var/lib/apt/lists/* \
    && echo /usr/local/cuda-13.0/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda-nvrtc.conf && ldconfig
USER dspk
```

Then `-e DSPK_RUNTIME=cuda`, or `--runtime cuda`, chooses it. NVRTC is NVIDIA's, under the CUDA
EULA; installing it is your choice, which is why the published image leaves it out.
