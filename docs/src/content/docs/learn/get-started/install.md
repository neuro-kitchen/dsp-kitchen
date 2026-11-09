---
title: Install
description: "Install dsp-kitchen for Python, Rust, the desktop app or a server."
sidebar:
  order: 1
---

Install the part you need. They share one version number.

| You want | Install |
|---|---|
| Python | `pip install dspk` |
| Python, with electrophysiology | `pip install "dspk[synapse]"` |
| Python, with the desktop app | `pip install "dspk[app]"` (`"dspk[all]"` for both) |
| A Rust library | `cargo add dspk` |
| The desktop app | `cargo install dspk-app` |
| The `dspk` command line | From a clone: `cargo install --path app/dspk-cli` |
| A server | The [container](/dsp-kitchen/deploy/build/) |

## Python

Wheels exist for Linux x86_64 (wgpu, CUDA, HIP and the CPU runtime), Windows x64 (wgpu, CUDA and the
CPU runtime) and macOS arm64 (wgpu and CPU), on CPython 3.10 or later. You choose the runtime when
your code runs, not when you install ([how](/dsp-kitchen/reference/python/runtime/)), and no CUDA or
ROCm toolkit is needed to install. HIP is included but untested: we have no AMD GPU to run it on.

`pip install dspk` is the core: recordings in Zarr and raw binary, operations and pipelines.
`dspk[synapse]` adds `dspk.synapse` and the neural formats (NWB, SpikeGLX, mtscomp) to `dspk.io.open`.
`dspk[app]` adds the desktop app: run `dspk-app` or `dspk.app.open("recording.bin")`.

## Rust

```toml
[dependencies]
dspk = "0.0.9"                     # wgpu, CUDA and CPU
# A lighter build, wgpu only:
# dspk = { version = "0.0.9", default-features = false, features = ["wgpu", "zarr"] }
```

CUDA is loaded when the program runs, and so is HIP (a `hip` feature, Linux only), so building with them needs no toolkit installed. The
CPU runtime downloads a prebuilt LLVM on its first build, cached in `~/.local/share/tracel`.

## The desktop app on Linux

The app's UI builds against these system libraries:

```sh
sudo apt-get install libfontconfig-dev libfreetype-dev libwayland-dev \
  libxkbcommon-dev libxkbcommon-x11-dev libx11-xcb-dev
```

## Spike sorters

The sorters are one separate program, because Kilosort4 and EMUsort are GPL-3.0 and the rest of
dsp-kitchen is MIT or Apache-2.0. The `dspk` command, which `pip install dspk` also gives you, fetches it:

```sh
dspk sorters install
```

:::caution[Early development]
dsp-kitchen is at 0.0.x, and APIs can change between releases. See the
[Changelog](/dsp-kitchen/changelog/).
:::

## Tune your device once

The first run on a device compiles kernels and benchmarks some of them. To do that up front, run this
once, or use **File → Tune this device** in the app. It took about a minute per GPU runtime and 14 minutes
on the CPU runtime on an RTX 2070 Max-Q and an i7-9750H; `DSPK_RUNTIME=wgpu dspk autotune` tunes only one:

```sh
dspk autotune
```

Next: [check your device](/dsp-kitchen/learn/get-started/check-your-device/).
