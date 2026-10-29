# Contributing to dspk

Thanks for helping. Issues are welcome for bugs, results that differ from their reference, and
features (the forms ask for what we need to reproduce). For a security problem, see
[SECURITY.md](SECURITY.md) instead.

## Setup

```bash
git clone https://github.com/neuro-kitchen/dsp-kitchen && cd dsp-kitchen
uv sync                                                        # Python development environment
cargo build --workspace --profile validate                     # every crate, wgpu + CUDA + CPU runtime
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
```

CUDA and HIP are loaded at run time, so no toolkit is needed to build. The CPU runtime downloads
a prebuilt LLVM on its first build.

## One build

Every crate builds with the wgpu, CUDA and CPU runtimes by default, and development uses one
profile, `--profile validate` (dependencies optimized, our crates with debug assertions). Other
feature sets or profiles recompile CubeCL and wgpu from scratch, so:

- build and test as separate steps: build with `--no-run` first, then run the test binaries;
- never `cargo test -p A -p B`: a package subset resolves features differently and rebuilds
  everything. Build the workspace (`--workspace --exclude dspk-app --exclude dspk-py`) and run
  the binaries you need;
- pick runtimes at run time, not at build time: `DSPK_TEST_RUNTIMES=cuda` (or `wgpu`, `cpu`).

## Tests

- Every kernel is tested on every runtime compiled in, against a host reference (scipy / NumPy
  results, or a direct host computation). CI has no GPU and runs the CPU runtime only, so run the
  GPU runtimes locally when you change a kernel.
- Kernels with several implementations (vector widths, tiles, sorting networks) are autotuned;
  test each choice against the scalar one (vector versions are bit-identical by construction).
- Timings: `cargo test -p dspk-ops --profile validate --test bench_kernels -- --ignored --nocapture`
  in a fresh process; see [benchmarks](benchmarks/README.md).

## Code

- Match the surrounding code: naming, comment density, module layout (one concept per module,
  kernels beside the code that launches them).
- Every tunable value is a setting with a visible default (a config field, a Python keyword with
  its default in the signature), not a constant buried in a function.
- Recordings keep their stored dtype to the device; conversion to float happens in kernels.
- Commit messages: `type(scope): summary` (`feat`, `fix`, `perf`, `docs`, `test`, `ci`,
  `refactor`, `chore`), with what changed and how it was checked in the body.

## Licences

dspk is MIT OR Apache-2.0. The spike sorters follow their upstreams: `dspk-ks4` (Kilosort4,
EMUsort) is GPL-3.0, `dspk-ms5` Apache-2.0, `dspk-sc2` / `dspk-tdc2` carry SpikeInterface's MIT
notice. Rules:

- No MIT / Apache crate depends on `dspk-ks4`; the sorters reach users through the separate
  `dspk-sorters` program (`cargo tree -i dspk-ks4` must show only `dspk-sorters`).
- Code written from another project's source keeps its licence and notice: say where it comes
  from (file, version) in the module docs and the crate's `NOTICE.md`, and keep GPL code in a GPL
  crate. Code written from a paper says so and cites it.

By contributing you agree that your contribution is licensed as the code it changes (MIT OR
Apache-2.0, or the crate's own licence).
