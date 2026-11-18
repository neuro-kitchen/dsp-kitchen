//! Preparing continuous multi-channel signals for viewing, wherever the viewer runs: a local
//! app next to the recording, or a remote viewer served over the network (dspk-stream carries
//! what this crate produces).
//!
//! - [`envelope`]: the min/max reduction (`[min, max]` per column keeps every spike peak visible
//!   at any zoom), on the host and (feature `device`, on by default with `wgpu`) on a CubeCL
//!   device.
//! - [`view`]: what a viewer asks for ([`View`]) and gets ([`Envelope`]), locally or remotely.
//! - [`pyramid`]: the min/max pyramid of a whole recording (in memory or in a file) and its
//!   background builder.
//! - [`signal`]: a signal as a viewer sees it ([`SignalBackend`]): [`LocalSignal`] (a recording
//!   and its pyramid in this process), or a remote session (dspk-stream) answering the same views.
//!
//! Rendering and UI belong to the app; transport to dspk-stream; file formats to dspk-io.

pub mod envelope;
pub mod pyramid;
pub mod signal;
pub mod view;
mod read;

pub use envelope::{min_max_decimate, min_max_decimate_into, Block, Columns};
#[cfg(feature = "device")]
pub use envelope::{envelope_launch, envelope_on_device, envelope_pairs};
pub use pyramid::{pyramid_path, OnProgress, Progress, Pyramid, PyramidBuilder, PyramidIdentity, FILE_BASE, MEMORY_BASE};
pub use signal::{LocalSignal, SignalBackend, StoredSignal, FILE_PYRAMID_MIN_BYTES};
pub use view::{Envelope, View, RAW_BLOCK_VALUES};

/// Most values (channels × samples, the chain's halo and margin included) a view's op chain is
/// processed over at once on a computer's GPU (`dspk serve`, the desktop app): 64 Mi float32 =
/// 256 MiB on the device (384 channels: about 5.8 s at 30 kHz). Wider windows are processed in
/// tiles of this size, up to [`PROCESSED_MAX_TILES`].
pub const PROCESSED_MAX_VALUES: usize = 1 << 26;

/// The same in a browser (a page's engine on WebGPU): 16 Mi float32 = 64 MiB per buffer. A
/// browser's WebGPU binds at most 128 MiB per storage buffer by default and loses the device past
/// what it can hold, which stops every later drawing; 32 channels: about 21 s at 24.4 kHz.
pub const PROCESSED_MAX_VALUES_BROWSER: usize = 1 << 24;

/// Most blocks of those a zoomed-out processed view is made of (processed and reduced to its
/// columns on the device one after another); past them the view is drawn raw.
pub const PROCESSED_MAX_TILES: usize = 32;
