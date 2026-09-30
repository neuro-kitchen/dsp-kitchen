# TDT (Tucker-Davis Technologies)

Reader: `src/inputs/tdt/`. Reference implementations: TDT `tdt` Python package (`TDTbin2py.py`),
neo `tdtrawio.py`.

## Block files
| File | Role | Module |
|---|---|---|
| `.tsq` | 40-byte event index (authoritative list of what was recorded) | `tsq.rs` |
| `.tev` | packet data for streams and snips | `streams.rs`, `snips.rs` |
| `.sev` | one file per channel (RS4 / Synapse "discrete files"); header v0–v4 | not yet |
| `.Tbk` | cp437 store settings; can list stores that never recorded | `tbk.rs` |
| `.tnt` | OpenEx notes (`NOTEFILE_VERSION[x]`) | `notes/openex.rs` |
| `Notes.txt` | Synapse notes (Experiment, Subject, User, Start, Stop, runtime notes) | `notes/synapse.rs` |
| `StoresListing.txt` | Synapse store descriptions + hardware objects | `notes/synapse.rs` |
| `.tin` | Synapse run archive (zip; `Summary.txt` has start time and build) | `notes/tin.rs` |
| `.Tdx` | index acceleration, unused | — |
| `*.csv` | Synapse impedance exports (`-1` = not measured) | `impedance.rs` |

## TSQ record
`size i32 (32-bit words incl. header) · evtype i32 · name [u8;4] · chan u16 · sortcode u16 ·
timestamp f64 (Unix s) · offset/value u64 · format i32 · frequency f32`

- Record 0: file header (`size` = file size). Start marker: evtype `0x8801`, name = code 1.
  Stop marker (code 2): the **last** record; missing when a block did not end cleanly.
- Streams (`0x8101`, `& 0xFF0F`): packets of `(size - 10) * 4` bytes per channel.
- Epoc onsets (`0x101`, marks `0x8801`): value in the offset field as f64.
- Epoc offsets (`0x102`): separate store named like the onset with `\`; its chan + sortcode hold
  the onset store's 4-byte name.
- Scalars (`0x201`): one record per channel sharing a timestamp; value as f64.
- Data formats: 0 f32, 1 i32, 2 i16, 3 i8, 4 f64, 5 i64, 8 rawpacked (not supported).
- Times are snapped to the 195312.5 Hz device clock, as TDT's reader does.
