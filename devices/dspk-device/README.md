# dspk-device

What runs next to a sensor: one live source, a chain of live (causal) ops, taps, recordings, and
the live protocol (`dspk_stream::live`) for one director and many viewers. A lost connection never
stops a device.

> [!WARNING]
> **Draft (0.0.x).** The protocol and this crate change between releases.

```bash
# A generator, here (noise, spikes, a sine), with a pairing code for a director
dspk-device --generator --channels 32 --rate 30000 --pair
# Once, on the computer that directs it (the code from above)
dspk pair 127.0.0.1:50060 42-1234-5678 --out me.pem
# Direct it
dspk device configure --identity me.pem 127.0.0.1:50060 chain.json
dspk device start     --identity me.pem 127.0.0.1:50060 --label first
dspk device watch     --identity me.pem 127.0.0.1:50060 gen --seconds 5
dspk device stop      --identity me.pem 127.0.0.1:50060
```

A chain is a `dspk-pipeline` spec, with a `live` section and `tap` names on nodes. Ops that need
future samples (zero-phase filters, whitening, medians, …) are refused, each with its reason:

```json
{
  "dspk_pipeline": 1,
  "live": { "block_ms": 10 },
  "nodes": {
    "in": { "op": "read" },
    "hp": { "op": "iir", "band": "highpass", "cutoff": [300], "mode": "forward", "tap": "filtered" },
    "spikes": { "op": "crossings", "level": -60, "direction": "falling", "refractory_ms": 1,
                "tap": { "name": "spikes", "record": true } }
  },
  "edges": [["in", "hp"], ["hp", "spikes"]]
}
```

A recording is a folder under `<data>/recordings/`, named by its start time and label:
`raw.zarr` (the source's stored values, growing chunk by chunk), a `<tap>.zarr` per recorded
signal tap, and `session-live.jsonl` (everything needed to understand the run, crash or not). A
recording a crash left open is marked `recovered` when the device starts again.

The device's configuration: `device.toml` ([`src/config.rs`](src/config.rs); an example in
`deploy/device.toml`, the container in `deploy/Containerfile.device`).
