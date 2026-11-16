//! The dspk session document: what is open and how it is processed and shown, as plain
//! serializable data with no UI and no device.
//!
//! - [`session`]: the document (recordings, views, their processing and display).
//! - [`command`]: every change as a command, applied by a pure function; the app, an agent and a
//!   headless run send the same commands.
//! - [`graph`]: sources → operations → views → writes; views are compiled from it.
//! - [`recipe`]: named chains of operations kept in files.
//! - [`timeline`]: the timeline the views share.
//! - [`epoch`]: epochs, event sets and placements as the session names them.
//! - [`derived`]: what is saved next to a recording (`<recording>.dspk.zarr`): epochs, event sets.
//! - [`paths`]: short forms of paths for messages.

pub mod command;
// The store next to a recording is a Zarr store on disk: not in a browser
#[cfg(not(target_family = "wasm"))]
pub mod derived;
/// In a browser there are no files next to a recording: the same names, nothing stored.
#[cfg(target_family = "wasm")]
#[path = "derived_web.rs"]
pub mod derived;
pub mod epoch;
pub mod graph;
pub mod paths;
pub mod recipe;
pub mod session;
pub mod timeline;
