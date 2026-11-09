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
| A Rust library | `cargo add dspk` |
| The desktop app | `cargo install dspk-app` |
| The `dspk` command line | From a clone: `cargo install --path app/dspk-cli` |
| A server | The [container](/dsp-kitchen/deploy/build/) |

## Python

Wheels exist for Linux x86_64 and Windows x64 (wgpu, CUDA and the CPU runtime) and macOS arm64
(wgpu and CPU), on CPython 3.10 or later.

## Rust

```toml
[dependencies]
dspk = "0.0.8"                     # wgpu, CUDA and CPU
# A lighter build, wgpu only:
# dspk = { version = "0.0.8", default-features = false, features = ["wgpu", "zarr"] }
```

CUDA and HIP are loaded when the program runs, so building with them needs no toolkit installed. The
CPU runtime downloads a prebuilt LLVM on its first build, cached in `~/.local/share/tracel`.

## The desktop app on Linux

The app's UI builds against these system libraries:

```sh
sudo apt-get install libfontconfig-dev libfreetype-dev libwayland-dev \
  libxkbcommon-dev libxkbcommon-x11-dev libx11-xcb-dev
```

## Spike sorters

The sorters are one separate program, because Kilosort4 and EMUsort are GPL-3.0 and the rest of
dsp-kitchen is MIT or Apache-2.0. The command line fetches it for you:

```sh
dspk sorters install
```

:::caution[Early development]
dsp-kitchen is at 0.0.x, and APIs can change between releases. See the
[Changelog](/dsp-kitchen/changelog/).
:::

Next: [check your device](/dsp-kitchen/learn/get-started/check-your-device/).
