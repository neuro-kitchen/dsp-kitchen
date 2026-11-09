---
title: Contributing
description: "Set up, build once, test, and send a change."
sidebar:
  order: 6
---

Issues are welcome for bugs, results that differ from their reference, and features. The forms ask for
what's needed to reproduce a problem. For a security problem, see the
[security policy](https://github.com/neuro-kitchen/dsp-kitchen/blob/main/SECURITY.md) instead.

## Set up

```sh
git clone https://github.com/neuro-kitchen/dsp-kitchen && cd dsp-kitchen
uv sync
cargo build --workspace --profile validate
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
```

## One build

Every crate builds with the wgpu, CUDA and CPU runtimes by default, and development uses one profile,
`--profile validate`. Other feature sets or profiles recompile CubeCL and wgpu from scratch, so:

- Build and test as separate steps: build with `--no-run`, then run the test binaries.
- Don't use `cargo test -p A -p B`: a package subset resolves features differently and rebuilds everything.
- Pick runtimes when running, not when building: `DSPK_TEST_RUNTIMES=cuda`.

## Tests

CI has no GPU and runs the CPU runtime only, so run the GPU runtimes locally when you change a kernel.

## Commits and pull requests

Commit messages are `type(scope): summary` (`feat`, `fix`, `perf`, `docs`, `test`, `ci`, `refactor`,
`chore`), with what changed and how it was checked in the body.

Keep a pull request to one area, so it stays reviewable.

## Licences

dsp-kitchen is MIT or Apache-2.0, and each sorter follows its upstream. No MIT or Apache crate may depend
on `dspk-sorter-kilosort`: `cargo tree -i dspk-sorter-kilosort` must show only `dspk-cli-sorter`.

The full guide is
[CONTRIBUTING.md](https://github.com/neuro-kitchen/dsp-kitchen/blob/main/CONTRIBUTING.md).
