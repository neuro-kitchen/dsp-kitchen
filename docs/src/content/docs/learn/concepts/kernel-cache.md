---
title: The kernel cache
description: "Why the first run on a device is slow and later runs are not."
sidebar:
  order: 6
---

The first time a kernel runs on a device, it is compiled, and for kernels with several exact
implementations each one is timed. Both results are saved, so later runs start in about a second.

## Where it is

One SQLite database, `default.db`, in CubeCL's folder of your cache directory:

| OS | Location |
|---|---|
| Linux, macOS | `$XDG_CACHE_HOME/cubecl/`, by default `~/.cache/cubecl/` |
| Windows | `%LOCALAPPDATA%\cubecl\` |
| No user directory | The temporary directory, `cubecl/` |

## Controls

| Setting | Effect |
|---|---|
| `DSPK_CACHE_DIR=/some/dir` | Puts the cache there (shared machines, CI, read-only home) |
| `DSPK_KERNEL_CACHE=0` | No compiled-kernel cache; autotune decisions still persist |

Deleting the folder is safe: the next run tunes and compiles again. It grows with every runtime,
device and kernel variant, to about 130 MB after the full test suite on three runtimes.

## Don't confuse it with LLVM

The CPU runtime's LLVM (`~/.local/share/tracel`) is a build-time download, not this cache. It's bundled
in the Python wheel.

## If a process aborts at exit

An upstream issue ([tracel-ai/cubecl#1981](https://github.com/tracel-ai/cubecl/issues/1981)) can abort
a process after its work is done, with `thread local panicked on drop` (exit code 134). dsp-kitchen
works around it. Code that opens devices through CubeCL directly, or calls
`dspk_core::compute::shutdown_devices` itself, can still hit it; `DSPK_KERNEL_CACHE=0` avoids it, at
the cost of compiling kernels in every process.
