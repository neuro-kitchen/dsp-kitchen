---
title: Build the image
description: "One Containerfile, four targets: cpu, nvidia, mesa and gpu."
sidebar:
  order: 1
---

The image holds the `dspk` command line and `dspk-cli-sorter`, on Debian slim, run as a non-root
user. Releases publish it ([the published image](/dsp-kitchen/deploy/image/)); build it yourself
for CUDA or for a change of your own. Four targets share everything but the system libraries each
GPU needs:

| Target | For | Adds | Run with |
|---|---|---|---|
| `cpu` (default) | no GPU | nothing (`DSPK_RUNTIME=cpu`) | |
| `nvidia` | NVIDIA: CUDA and Vulkan | NVRTC 13.0 and the Vulkan loader; the driver comes from the host | `--device nvidia.com/gpu=all` |
| `mesa` | AMD, Intel: Vulkan (wgpu) | the Vulkan loader and Mesa's Vulkan drivers | `--device /dev/dri` |
| `gpu` | any GPU through wgpu, or the CPU (the published image) | the Vulkan loader, Mesa's drivers and what NVIDIA's Vulkan driver needs; no NVIDIA software | `--device nvidia.com/gpu=all` or `--device /dev/dri` |

Build from the repository root, because the build needs the whole workspace:

```sh
podman build -f deploy/Containerfile --ignorefile deploy/.containerignore \
  --target nvidia -t dspk-serve:nvidia .
```

The first build compiles everything and takes a while. Another target afterwards compiles nothing:
they share the build stage and Podman's caches.

## Build options

| Option | What it does |
|---|---|
| `--build-arg CUDA_NVRTC=13-5` | A newer NVRTC, at most the CUDA version of the host's driver (`nvidia-smi` shows it) |
| `--build-arg FEATURES=cpu,hub` | Compile the CPU runtime only |

Next: [run the server](/dsp-kitchen/deploy/run/).
