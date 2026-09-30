# neuro-convert

Read neurophysiology recordings into one neutral model and convert them to publication formats.
Pure Rust, streaming (multi-hour recordings never sit in memory), standalone (no `dsp-*` crates).

```
neuro-convert formats                         # inputs / outputs this build supports
neuro-convert inspect <recording> [--json]    # streams, events, tables, metadata, warnings
neuro-convert convert <recording> -m meta.yaml -o out.nwb.zarr [--dry-run] [--gzip 1]
neuro-convert validate out.nwb.zarr           # structural checks of the NWB store
```

## Supported
| Input | Versions |
|---|---|
| TDT block or tank | Synapse / OpenEx; TEV and SEV (v0–v3, hour files) streams, rawpacked, snips (+ `--sort` offline sorts), epocs, scalars, runtime notes, impedance CSVs; `--block` for tanks |

| Output | Format |
|---|---|
| NWB 2.11.0 | Zarr v3 store in hdmf-zarr's layout (readable by pynwb / hdmf-zarr, DANDI), schema cached |

Not yet: snippets → NWB, SpikeGLX, Open Ephys, Intan, NWB/HDF5.

## Metadata file
The source files never say everything NWB needs (session description, subject species/age, time
zone, electrode placement) nor how each stream should be exported. A YAML file supplies both; keys
are the source's own names, so any lab's naming works. Start from
[`metadata/session.example.yaml`](metadata/session.example.yaml); a real example is
[`metadata/examples/tdt-15-25-33_meps.yaml`](metadata/examples/tdt-15-25-33_meps.yaml).
`convert --dry-run` prints the resulting plan with errors (blocking) and DANDI warnings.

## Layout
```
src/model/      Session, Recording (chunked reads), events, snippets, tables, metadata, provenance
src/common/     codecs, memory-mapped files, text encodings, ISO time helpers
src/inputs/     one folder per format (tdt/: tsq, tev streams, epocs, snips, notes, tin, …)
src/metadata/   the YAML metadata file
src/outputs/    nwb/: mapping (plan), types (one file per NWB type), backend (Zarr), validate
specs/          vendored NWB / HDMF schema (BSD), cached into every file
docs/           format notes and the NWB mapping
tests/          TDT vs TDT's Python reader; NWB write + pynwb read-back (validate_nwb.py)
```

## Tests
```
cargo test                      # unit + integration (the real-block test skips without data)
uv run --no-project --with pynwb --with hdmf-zarr --with nwbinspector \
    tests/validate_nwb.py target/nwb-test/small.nwb.zarr --small
```
