# benchmarks

What dspk measures about itself: time, and agreement with other tools. Every benchmark writes a
**JSON record** (the schema of [`plating.results`](../plating/plating/results.py): dspk's version
and commit, the runtime, the reference tool and its version, the machine, and per case the times,
the error and pass / fail). Records and the figures drawn from them (with
[`plating`](../plating/README.md)) are committed, so runs can be inspected and compared across
versions and machines; anything else a run leaves in `results/` is not.

```text
benchmarks/
├── signal-operations/   dspk vs scipy / numpy, from Python: time and agreement per operation
│   └── results/         scipy_timing-<runtime>.json + its figures
├── kernels/             Rust crate `dspk-benchmarks`: device-only timings of dspk-ops / dspk-synapse kernels
│   └── results/         <benchmark>-<device>.json
└── sorters/             our sorters vs Kilosort4 and the others; sort time and memory
    ├── kernels/         Rust crate `dspk-benchmarks-sorters` (GPL-3.0, like the sorter it measures)
    └── results/
```

## signal-operations

`scipy_timing.py`: one 384 × 60 000 block (2 s of Neuropixels at 30 kHz, float32) through
zero-phase and causal band-pass, notch, FIR, median filter, polyphase resampling and common median
reference, on every runtime compiled in, against scipy / numpy in float64. dspk's time is end to
end from a NumPy array (upload, kernels, download), the time a Python caller sees; at this size most
of it is the ~92 MB transfer, which is why chained stages (`dspk.ops.Pipeline`, pipeline specs) gain
the most. Tolerances (relative to scipy's peak): 1e-5 for the filters and resampling (float32 on the
device against float64), 1e-6 for median and CMR (exact).

```bash
uv run python benchmarks/signal-operations/scipy_timing.py [--repeats 5] [--channels 384] [--samples 60000]
```

Build the Python package with `maturin develop --release` before timing it (`--profile validate`
optimizes the dependencies but not dspk itself). Run each benchmark in a fresh process.

## kernels

The device alone, no host transfers: ignored Rust tests that print a table and write a record.

| Test | Measures |
|---|---|
| `bench_kernels` | every kernel family on one recording-sized block, as bandwidth and share of the streaming ceiling |
| `bench_vector` | every kernel choice (scalar, tiled, each vector width) per operation |
| `bench_reduce` | per-channel reductions and transposes at sorter sizes |
| `bench_linalg` | covariance, second moment, dense spatial operator, projection |
| `bench_eigen` | the batched symmetric eigensolver, with its residual |
| `bench_upload` | host → device upload strategies for one window |
| `bench_hdbscan` | HDBSCAN on synthetic clips (`HDBSCAN_POINTS`, `HDBSCAN_REFERENCE` to compare labels) |

```bash
cargo test -p dspk-benchmarks --profile validate --test bench_kernels -- --ignored --nocapture
DSPK_TEST_RUNTIMES=cuda cargo test -p dspk-benchmarks ...   # one runtime
```

`bench_reduce`, `bench_eigen` and `bench_linalg` open the discrete and integrated GPUs through wgpu
by name; the others run on every runtime of `DSPK_TEST_RUNTIMES`.

## sorters

`kernels/` (Rust, GPL-3.0): Kilosort4's universal-template detection and EMUsort's channel-delay
estimation, per window:

```bash
cargo test -p dspk-benchmarks-sorters --profile validate --test bench_detection -- --ignored --nocapture
```

The Python scripts below run the sorter program `dspk-cli-sorter` (found through
`DSPK_CLI_SORTER_BIN`, `dspk sorters install`'s folder, else `target/validate`; see
`sorter_cli.py`) and write a JSON summary in `results/`. Timings mean something only with a release
build of the program.

| Script | What it does |
|---|---|
| `kilosort4_validation.py` | our Kilosort4 vs Kilosort4's saved results, per channel |
| `kilosort4_experiments.py` | settings-only experiments: where our output departs from Kilosort4's |
| `kilosort4_benchmark.py` | Kilosort4 speed and outputs vs Kilosort4's own results |
| `mountainsort5_validation.py`, `spykingcircus2_validation.py`, `tridesclous2_validation.py` | each sorter's units vs Kilosort4's (a second, independent sorter on the same data) |
| `emusort_checks.py` | EMUsort on HD-EMG: reproducibility, refractory violations, presence, templates |
| `sorter_resources.py`, `sorter_resources_table.py` | time and memory of a whole sort, and the table image |
| `ks4_reference.py` | helper: Kilosort4's Phy folder as a reference, per-channel matching |
| `isosplit_si_reference.py` | reference outputs of SpikeInterface's isosplit for the Rust test `spikeinterface_variant_matches_upstream` (`ISOSPLIT_SI_REF`) |

Data is local and not in the repository: `<repository>/.datasets/`, or the folder in
`DSP_KITCHEN_DATA`.

| Path under `.datasets/` | Used by |
|---|---|
| `kilosort4/ZFM-02370_mini.imec0.ap.short.bin` + `.meta` (Neuropixels 1.0, SpikeGLX) | `kilosort4_*`, `*_validation`, `sorter_resources` |
| `kilosort4/saved_results/` (Kilosort4's output for that file, Phy folder) | `kilosort4_*`, `*_validation` (through `ks4_reference.py`) |
| `nwb/15-25-33_meps.nwb.zarr` (HD-EMG, 4 × 8 grid, `HDEMG` series) | `emusort_checks`, `sorter_resources` |
