//! A dspk device: what runs next to a sensor (a container, a Raspberry Pi).
//!
//! - One live source ([`config::SourceConfig`]: the generator, the minimal sensor format, a
//!   replayed file).
//! - One chain: a `dspk-pipeline` spec checked for live input (`dspk_pipeline::live`), run block
//!   by block; its named outputs are taps.
//! - Runs: a recording stores the raw data (`raw.zarr`), derived taps marked `record`, and
//!   `session-live.jsonl`; a preview stores nothing. A lost connection never stops a run.
//! - Clients: one director, many viewers, over the live protocol (`dspk_stream::live`).
//!
//! The `dspk-device` program runs one; `dspk device` (dspk-cli, feature `device`) runs the same.
//! Design: `.tasks/next/40_realtime-protocol.md` in the repository's notes.

pub mod config;
pub mod device;
pub mod recordings;
pub mod serve;
pub mod sources;
pub mod taps;

pub use config::{DeviceConfig, SourceConfig, DEFAULT_PORT};
pub use device::LiveDevice;
pub use serve::{serve, Listening};
