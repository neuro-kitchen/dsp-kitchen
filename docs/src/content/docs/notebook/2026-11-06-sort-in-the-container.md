---
title: "2026-11-06: Kilosort4 in the container and through the server"
description: "A Neuropixels recording sorted on CUDA, in the container and from a client."
sidebar:
  label: "2026-11-06: sorting on a server"
  order: 1
---

**Question.** Can a recording be sorted on a GPU inside the hardened container, and started from
another client?

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX 2070, driver 615.71 |
| Container | Rootless Podman 6.1, Fedora 45, the `nvidia` image target |
| Runtime | CUDA |
| Recording | The Kilosort4 test recording: Neuropixels, 45 s at 30 kHz |

## In the container, 30 s

```sh
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable --userns=keep-id:uid=10001,gid=10001 \
  -v ~/dspk-data:/data:Z dspk-serve:nvidia \
  sort --runtime cuda kilosort4 /data/recording.ap.bin --probe neuropixels1 \
  --output /data/recording.kilosort4.sorting.zarr --duration-sec 30
```

**Result:** 136 units, 63,014 spikes, in 28.6 s.

`dspk doctor` in the same image listed wgpu and CUDA as usable.

## Through the server, whole recording

A client enrolled with `--control` asked the server to sort the whole 45 s recording on CUDA.

**Result:** 287 units, 133,544 spikes, in 46 s. The sorting, fetched back to the client, was identical.

The same recording sorted locally with `dspk sort kilosort4` also gave 287 units and 133,544 spikes.

**Found on the way, and fixed the same day:** a stopped or failed job left its partial output and blocked reruns, and the server's runtime wasn't visible to clients.
