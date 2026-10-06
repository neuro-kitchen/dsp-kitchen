# dsp-cli — changes

## Tree (2026-10-06)

File name = command name; network commands grouped.

| Change | Note |
|---|---|
| `commands/info.rs` → `commands/open.rs`, `run_info` → `run_open` | The `open` command; `info` remained an alias of `components` (CLI1, CLI11). |
| `commands/{serve, receive}.rs` → `commands/net/{serve, receive}.rs` (+ `net/mod.rs`) | — |
| dsp-stream paths: `network::…` → `protocol::…` / `transport::…`; `dsp_stream::PrefetchReader` → `dsp_io::PrefetchReader` | — |

Not built (it depends on crates still under cleanup). Pending, per `REVIEW.md`: `components` → `doctor` (CLI1), `stream` (CLI2 / ST4: now that decimation is in dsp-view, `stream` can become real: `serve --purpose vis`), argument structs per command (CLI13).

## Breakage (2026-10-06)

dsp-stream's session protocol replaced `StreamFrame`, `StreamPurpose`, `QuicStreamServer`,
`QuicStreamClient`, `generate_server_config`, `make_client_config_with_cert` and
`make_insecure_client_config`. Rewrite `net/serve.rs` on `server_config(&ServerIdentity)` +
`Server` + `serve_recording(…, ServeOptions)`, and `net/receive.rs` on `client_config(&ServerTrust)`
+ `Session::{connect, subscribe}` + `SignalReceiver`. `receive` can then report real latency
(`sent_unix_ns`) and save through dsp-io (CLI3, CLI4). `stream.rs` (`StreamPurpose`) goes (CLI2).

## Rework (2026-10-06): REVIEW CLI1–CLI14 done, it builds

`cargo check` / `clippy` (no dsp-cli warnings) / `build` pass; smoke-run: `--help`, `doctor`,
`probe neuropixels1`, `hub list`. Tests not run (end pass).

| Finding | Change |
|---|---|
| CLI1 | `components` (fixed text) → **`doctor`**: version, enabled features, each compiled-in runtime opened and described from its properties (or "not usable"), the selected runtime, the hub cache. |
| CLI2 | `stream` removed (views are now dsp-stream `View` requests). |
| CLI3 | `receive` reports throughput, real-time factor, inter-arrival percentiles and **latency** from `SignalFrame.sent_unix_ns`; no "packet loss" over a reliable stream. |
| CLI4 | `receive --save` writes frames as they arrive (bounded memory), time-major, with a dsp-io JSON sidecar: stored words when channels share one µV gain / offset, else `f32` (with a note when units are not µV). Short last frames are handled. |
| CLI5 | `benchmark pipeline --save` writes through `dsp_io::write_raw` (sidecar). Buffer sizes from `size_of`. |
| CLI6 | Choices are `ValueEnum`s (`probe` presets, `generate` container / dtype / order): typos are rejected by clap with a non-zero exit. |
| CLI7 | `--sample-rate` wherever time is reported (`benchmark`); defaults are named constants. |
| CLI8 | `open` prints each channel's gain and unit symbol; option names lose `_uv` (`--noise`, `--line-noise`, `--step`, units in the help). |
| CLI9 | No defaults into the repository: `generate <output>`, `benchmark suite --report-dir`, `--save` paths are explicit. |
| CLI10 | `benchmark pipeline` times a `PipelineWorkspace` (upload + stages, optional `--int16` stored upload, download); `sweep` likewise across channel counts; `suite` ported to current APIs with every setting from library defaults (`StreamingDetectionConfig::default()`, matching-pursuit `DEFAULT_*`, dsp-base `*_DEFAULT_EDGE`), dsp-io `SyntheticRecording` as signal, streaming detection end to end, matching pursuit with the templates detection found. |
| CLI11 | No aliases. |
| CLI12 | Features `wgpu` (default) / `cpu` / `cuda` / `hip` forward to dsp-core, dsp-base, dsp-synapse; `hub` (default) enables the `hub` command (dsp-synapse-ml `hub`, no Burn backends). |
| CLI13 | One `#[derive(Args)]` struct per command in its module, `run(args)`; `main.rs` only parses and dispatches. |
| CLI14 | `memmap2`, `quinn` dropped (QUIC is reached through dsp-stream). |
| — | `serve` uses dsp-stream (`server_config` self-signed or PEM `--certificate/--private-key`, `--certificate-out`, `Server`, `serve_recording`), and answers views from the recording's pyramid (`Pyramid::open_or_create` next to the file, or in memory, built by `PyramidBuilder`). `receive` trusts `--certificate` (PEM/DER) or `--insecure`. |
| — | Logs default to warnings (`DEFAULT_LOG_LEVEL`); `-v/--verbose` for debug. |

Tree:
```text
src/main.rs                 Cli (global --runtime, --verbose) + dispatch
src/commands/{doctor, inspect, open, probe, generate, benchmark, hub}.rs
src/commands/net/{serve, receive}.rs
```

Open: CLI15 (new commands `detect`, `sort`, `convert`) — your decision.

