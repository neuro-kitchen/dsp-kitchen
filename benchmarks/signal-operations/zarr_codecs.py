# %% [markdown]
# # Zarr codecs on float32: size against speed
# Writes a real Neuropixels excerpt as a Zarr store with each codec, raw and high-passed (the two
# kinds of float32 a pipeline writes), and records the size on disk, the write time and the read
# time, the median of `REPEATS`. Every store is read back and compared with what was written.
#
#     uv run python benchmarks/signal-operations/zarr_codecs.py
#
# Writes `results/zarr_codecs.json`. Read times come from the page cache (the store was just
# written): they measure decompression, not the disk.
#
# Data (local, git-ignored): `.datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`).

# %%
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

import dspk
from plating import results

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", ROOT / ".datasets"))
RECORDING = DATA_DIR / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
#: On the real disk, not tmpfs, inside the build folder (git-ignored).
WORK = ROOT / "target" / "zarr_codecs"
OUTPUT = Path(__file__).resolve().parent / "results" / "zarr_codecs.json"
SECONDS = 10
#: The recording's last channel is the sync line, not a signal.
SIGNAL_CHANNELS = 384
HIGHPASS_HZ = 300.0
REPEATS = 3
CODECS = [
    ("none", {"compression": "none"}),
    ("zstd 1", {"compression": "zstd", "level": 1}),
    ("zstd 3 (default)", {"compression": "zstd", "level": 3}),
    ("zstd 3, one thread", {"compression": "zstd", "level": 3, "parallel": False}),
    ("zstd 9", {"compression": "zstd", "level": 9}),
    ("blosc_lz4", {"compression": "blosc_lz4"}),
]

# %% The two signals
rec = dspk.io.open(RECORDING)
fs = rec.sample_rate
raw = np.asarray(rec.read(start=0, stop=int(SECONDS * fs)), dtype=np.float32)[:SIGNAL_CHANNELS]
signals = {"raw": raw, "high-passed": np.asarray(dspk.ops.filter.iir(raw, fs, "highpass", HIGHPASS_HZ), dtype=np.float32)}


def size_of(store: Path) -> int:
    return sum(p.stat().st_size for p in store.rglob("*") if p.is_file())


# %% Measure
cases = []
for signal, data in signals.items():
    for name, options in CODECS:
        store = WORK / "store.zarr"
        writes, reads, back = [], [], None
        for _ in range(REPEATS):
            shutil.rmtree(store, ignore_errors=True)
            start = time.perf_counter()
            dspk.io.write_zarr(data, store, fs, unit="uV", **options)
            writes.append((time.perf_counter() - start) * 1e3)
            start = time.perf_counter()
            back = np.asarray(dspk.io.open(store).read(start=0, stop=data.shape[1]), dtype=np.float32)
            reads.append((time.perf_counter() - start) * 1e3)
        size = size_of(store)
        case = results.compare(f"{signal} / {name}", back, data, tolerance=0.0, dspk_ms=statistics.median(writes), read_ms=statistics.median(reads), size_bytes=size, ratio=data.nbytes / size)
        cases.append(case)
        print(f"{case.name:34s} {size / 2**20:8.1f} MiB  ratio {data.nbytes / size:5.2f}  write {case.dspk_ms:8.1f} ms  read {statistics.median(reads):8.1f} ms  exact {case.passed}", flush=True)
shutil.rmtree(WORK, ignore_errors=True)

# %% Save
notes = f"{SECONDS} s of {SIGNAL_CHANNELS} Neuropixels channels at {fs:g} Hz, float32 in µV, default chunking. Median of {REPEATS} runs. dspk_ms is the write; extra holds read_ms (from the page cache), size_bytes and ratio."
print(f"saved {results.write(OUTPUT, 'zarr_codecs', cases, runtime='host', notes=notes)}")
