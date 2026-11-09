---
title: Pass the GPU in
description: "NVIDIA, AMD and Intel GPUs in the container, and what SELinux needs."
sidebar:
  order: 5
---

Check what the container sees:

```sh
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable dspk-serve:nvidia doctor
```

Every runtime is listed as usable, or with the reason it isn't. The `Selected:` line is the one
commands use (`DSPK_RUNTIME`, else the first compiled in), and `doctor` warns when it has no device
and names one that has.

| GPU | What to do |
|---|---|
| NVIDIA | Install the NVIDIA Container Toolkit and generate its CDI file: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Generate it again after a driver update. Then `--device nvidia.com/gpu=all` passes the driver in |
| AMD, Intel | `--device /dev/dri` with the `mesa` target. Rootless containers also need `--group-add keep-groups` to keep your `render` and `video` groups |
| None | The `cpu` target |

`dspk serve` itself only streams. The GPU is for the sorts the server runs.

## SELinux (Fedora, RHEL)

SELinux lets a container touch only files labelled for it. The GPU's device files aren't, so with
the GPU passed in CUDA reports `CUDA_ERROR_NO_DEVICE` and Vulkan finds no adapter. There are two ways around it:

| | `--security-opt=label=disable` | `setsebool -P container_use_devices on` |
|---|---|---|
| What it does | Turns labels off for this one container | Keeps labels on for every container and lets them use device files in general |
| Needs | Nothing | Root, once per machine |
| Then | `SecurityLabelDisable=true` in the service file | Drop `label=disable` from the commands and the service file |

Either way the other protections stay: non-root user, read-only root, no capabilities,
no-new-privileges, the pids limit, and only enrolled clients.

## Tested

On 2026-11-06, with an RTX 2070, driver 615.71, rootless Podman 6.1 and Fedora 45: `doctor` in the
`nvidia` target showed wgpu and CUDA usable, and Kilosort4 on 30 s of a 384-channel Neuropixels
recording ran in the container on CUDA in 28.6 s (136 units, 63,014 spikes).
