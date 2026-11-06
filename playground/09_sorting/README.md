# 09 · Sorting

Real data from here on: start with `00_fetch_kilosort_data.py` (Kilosort4's Neuropixels test
recording, checked by SHA-256, into `playground/data/`, git-ignored).

```bash
uv run python playground/09_sorting/00_fetch_kilosort_data.py
```

The sorting chapters (Kilosort4, EMUsort, MountainSort5, SpykingCircus2, Tridesclous2 run from
Python, then their units inspected) wait for the synapse Python bindings (`.tasks/27`). Until
then:

- `specs/neuropixels_preprocess.json`: band-pass 300–6000 Hz + common median reference of the
  recording into Zarr; run it with `dspk run playground/09_sorting/specs/neuropixels_preprocess.json`
  (edit its `read` path to `../../data/kilosort4/…` once fetched).
- Sort from the command line: `dspk sort kilosort4 <recording> -o out.sorting.zarr` (see
  `dspk sort kilosort4 --help`; `--probe-file probe.json` / `--channel-map map.toml` when the
  file's channel order is not the probe's).
- `kilosort4_spikes_figure.py`, `inspection.py`: material for the figures of this chapter.
