//! Preparing continuous multi-channel signals for viewing, wherever the viewer runs: a local
//! app next to the recording, or a remote viewer served over the network (dsp-stream carries
//! what this crate produces).
//!
//! - [`envelope`]: the min/max reduction (`[min, max]` per column keeps every spike peak visible
//!   at any zoom), on the host and (feature `device`, on by default with `wgpu`) on a CubeCL
//!   device.
//! - [`view`]: what a viewer asks for ([`View`]) and gets ([`Envelope`]), locally or remotely.
//! - [`pyramid`]: the min/max pyramid of a whole recording (in memory or in a file) and its
//!   background builder.
//!
//! Rendering and UI belong to the app; transport to dsp-stream; file formats to dsp-io.

pub mod envelope;
pub mod pyramid;
pub mod view;
mod read;

pub use envelope::{min_max_decimate, min_max_decimate_into, Block, Columns};
#[cfg(feature = "device")]
pub use envelope::envelope_on_device;
pub use pyramid::{pyramid_path, OnProgress, Progress, Pyramid, PyramidBuilder, PyramidIdentity, FILE_BASE, MEMORY_BASE};
pub use view::{Envelope, View};
