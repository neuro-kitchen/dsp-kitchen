# %% [markdown]
# # 09 · Sorting — 0. The Kilosort test recording
# From here on the playground uses real data: Kilosort4's test recording, a Neuropixels 1.0 probe
# (385 channels, 30 kHz, 45 s, int16, SpikeGLX). This downloads it once into
# `playground/datasets/kilosort4/` (git-ignored), checks it against the SHA-256 the playground was
# written with, and writes the JSON `.meta` dspk reads raw binaries with.
#
#     uv run python playground/09_sorting/00_fetch_kilosort_data.py [--url URL] [--dest DIR]
#
# The address is the one Kilosort4's documentation downloads it from; if it moves, pass the new
# one with `--url` (the checksum still guards the file). Already have it? `--from FILE` copies a
# local `.bin` instead (it is checked the same way).

# %%
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

#: Where Kilosort4's documentation downloads the recording from (a zip of the .bin).
DEFAULT_URL = "https://www.kilosort.org/downloads/ZFM-02370_mini.imec0.ap.short.zip"
NAME = "ZFM-02370_mini.imec0.ap.short.bin"
#: SHA-256 of the .bin this playground was written with.
SHA256 = "918be4edccddbb1a0d420afae649cf0f33f2f6720365d743fa9cb0923da7b2b9"
SIZE = 1_039_500_000
#: dspk's raw-binary sidecar for it (SpikeGLX imec AP band: 2.34375 µV per step).
META = {"channels": 385, "sample_rate_hz": 30000.0, "format": "int16", "order": "time_major", "gain_uv": 2.34375, "offset_uv": 0.0, "header_bytes": 0, "samples": 1350000}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default=DEFAULT_URL)
parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "datasets" / "kilosort4")
parser.add_argument("--from", dest="source", type=Path, help="copy this local .bin instead of downloading")
args, _ = parser.parse_known_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# %% Get the file (once)
args.dest.mkdir(parents=True, exist_ok=True)
target = args.dest / NAME
if target.exists() and target.stat().st_size == SIZE and sha256(target) == SHA256:
    print(f"already here: {target}")
else:
    if args.source:
        shutil.copyfile(args.source, target)
    else:
        archive = args.dest / "download.zip"
        print(f"downloading {args.url} …")
        urllib.request.urlretrieve(args.url, archive)
        with zipfile.ZipFile(archive) as z:
            member = next(m for m in z.namelist() if m.endswith(NAME))
            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        archive.unlink()
    got = sha256(target)
    if got != SHA256:
        target.unlink()
        raise SystemExit(f"checksum mismatch: expected {SHA256}, got {got}; the file was removed")
    print(f"verified: {target}")
(args.dest / NAME.replace(".bin", ".meta")).write_text(json.dumps(META, indent=2) + "\n")
print("ready: dspk.io.open(", target, ")")
