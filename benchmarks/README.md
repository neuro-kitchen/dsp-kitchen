# benchmarks

Timings and validations against other tools. Each folder writes what it measures to its own
`results/` (git-ignored); the reports record the machine and versions next to the numbers.

```text
benchmarks/
├── signal-operations/   dspk vs scipy / numpy: time and agreement per operation
│   └── results/
└── sorters/             our sorters vs Kilosort4's saved results; sorter time and memory
    └── results/
```

Kernel-level timings (the device alone, no host transfers) are Rust tests next to the kernels,
ignored by default:

```bash
cargo test -p dspk-ops --profile validate --test bench_kernels -- --ignored --nocapture   # also bench_reduce, bench_linalg, bench_eigen, bench_upload
DSPK_TEST_RUNTIMES=cuda cargo test ...                                                   # one runtime
```

Build the Python package with `maturin develop --release` before timing it (`--profile validate`
optimizes the dependencies but not dspk itself). Run each benchmark in a fresh process.

## signal-operations

`scipy_timing.py`: one 384 × 60 000 block (2 s of Neuropixels at 30 kHz, float32) through
zero-phase and causal band-pass, notch, FIR, median filter, polyphase resampling and common median
reference, on every runtime compiled in, against scipy / numpy. dspk's time is end to end from a
NumPy array (upload, kernel, download): at this size most of it is the ~92 MB transfer, so
the GPU times are close to each other and chained stages (`dspk.ops.Pipeline`) gain the most.

```bash
uv run python benchmarks/signal-operations/scipy_timing.py [--repeats 5] [--channels 384] [--samples 60000]
```

First run (2026-10-29, dspk 0.0.3 built with `--profile validate`, scipy 1.18.1, RTX 2070 Max-Q,
12-thread x86-64; 2 repeats): every operation agrees with scipy to ≤ 4e-6 of the peak (median and
common median reference exactly); scipy 0.33–2.1 s, dspk 0.15–0.28 s on wgpu / CUDA. CPU runtime:
0.23–0.46 s, except common median reference (1.7 s, scalar code; see `.tasks/12_cpu-vectorization.md`).

## sorters

> [!NOTE]
> These scripts use the previous Python API (`dsp_kitchen`) and wait for the synapse bindings;
> they do not run on the current `dspk` package.

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

Data is local and not in the repository: `<repository>/data/`, or the folder in
`DSP_KITCHEN_DATA`.

| Path under `data/` | Used by |
|---|---|
| `kilosort4/ZFM-02370_mini.imec0.ap.short.bin` + `.meta` (Neuropixels 1.0, SpikeGLX) | `kilosort4_*`, `*_validation`, `sorter_resources` |
| `kilosort4/saved_results/` (Kilosort4's output for that file, Phy folder) | `kilosort4_*`, `*_validation` (through `ks4_reference.py`) |
| `nwb/15-25-33_meps.nwb.zarr` (HD-EMG, 4 × 8 grid, `HDEMG` series) | `emusort_checks`, `sorter_resources` |
