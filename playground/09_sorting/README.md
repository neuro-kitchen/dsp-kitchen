# 09 · Sorting

Real data from here on: Kilosort4's Neuropixels 1.0 test recording (45 s, 384 sites), fetched once
and checked by SHA-256 into `playground/datasets/` (git-ignored):

```bash
uv run python playground/09_sorting/00_fetch_kilosort_data.py          # or --from <a local copy of the .bin>
```

| Script | What it shows |
|---|---|
| `01_detect_and_locate` | detection over the whole file (streamed on the device), every spike located on the probe, the drift map and the probe's motion |
| `02_sorting_quality` | a sorting judged against a reference (dspk's Kilosort4 against Kilosort4's results) and unit by unit (refractory violations, autocorrelogram) |

`02_sorting_quality` reads two sortings next to the recording:

- `<recording>.kilosort4.sorting.zarr`: `dspk sort kilosort4 playground/datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.bin -o playground/datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.kilosort4.sorting.zarr`
  (`dspk sort kilosort4 --help`; `--probe-file probe.json` / `--channel-map map.toml` when the
  file's channel order is not the probe's);
- `saved_results/`: Kilosort4's own output on the recording (a Phy folder).

The sorters run from the command line or a server (`dspk sort`, `dspk.remote.Session.sort`); they
are not in the Python wheel (Kilosort4 and EMUsort are GPL-3.0).

Also here: `specs/neuropixels_preprocess.json` (band-pass + common median reference into Zarr,
`dspk run …`), and `inspection.py` (an interactive look at sorted units on one channel).
