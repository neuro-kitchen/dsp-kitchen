# NWB output

Writer: `src/outputs/nwb/`. Target: NWB core 2.11.0 (hdmf-common 1.10.0, hdmf-experimental 0.6.0),
stored as a Zarr v3 store in the layout hdmf-zarr 0.14 writes, so pynwb / hdmf-zarr and DANDI read it.
The schema is cached under `/specifications` (vendored in `specs/`).

## Mapping (driven by the metadata file, never by source names)
| Session item | NWB | Notes |
|---|---|---|
| Recording, `type: electrical` | `/acquisition/<name>` `ElectricalSeries` | data `[time, channel]`, unit volts, `conversion` from metadata; `electrodes` region into the electrodes table |
| Recording, `type: timeseries` (default) | `/acquisition/<name>` `TimeSeries` | unit from metadata (default `a.u.`) |
| Data type | as stored when unscaled integers (e.g. TDT int16), else float32 | `conversion` scales integers to the unit |
| Recording, `include: false` | — | listed under "skipped" in the plan |
| Electrode groups (metadata) | `/general/extracellular_ephys/<group>` | linked to a device |
| Electrical channels | `/general/extracellular_ephys/electrodes` | columns `location, group, group_name, channel_name`, plus `imp` (ohms) when a group names an impedance table |
| Events (epocs, marks, notes) | `/events/<name>` `EventsTable` (NWB 2.9+) | `timestamp`, `duration` (offset − onset, when offsets exist), `value`, `annotation` (labels such as notes) |
| Multi-channel scalars | `/acquisition/<name>` `TimeSeries` `[event, channel]` | timestamps, or `starting_time` + `rate` when evenly spaced (±0.01 %) |
| Snippets | — | not exported yet (plan warning); `SpikeEventSeries` mapping pending |
| Tables (e.g. impedance CSVs) | `/analysis/<name>` `DynamicTable` | numeric columns float64, others text |
| Devices | `/general/devices/<name>` | from the source + metadata |
| Session metadata | root + `/general` | description, identifier (UUID if absent), start time with zone, lab, … |

## Checks before writing
Errors (nothing is written): missing `session.description`, start time without a zone, an
electrical stream without a declared electrode group, duplicate output names.
Warnings: DANDI-required fields missing (`subject.species`, `subject.age`), unknown stream keys.

## `neuro-convert validate <store>`
Structural checks without Python: root type and required datasets, start time zone, cached schema,
electrodes table column lengths and group references, electrode-group → device links, every
series has data + unit + (rate or matching timestamps), electrical series reference valid electrode
rows matching their channel count, interval tables have equal columns and no negative durations.

## Verification
`tests/outputs_nwb.rs` writes a small store; `tests/validate_nwb.py` reads it back with pynwb +
hdmf-zarr (exact values) and runs nwbinspector. nwbinspector 's compression checks currently raise
on every Zarr v3 file (also pynwb's own output) — an upstream issue.
