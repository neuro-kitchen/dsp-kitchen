//! A view's processed signal: the one place the desktop app, `dspk serve` and a browser page's
//! engine turn a view's ops into what it draws.
//!
//! - [`view_plan`]: a view's ops planned as every chain is ([`crate::chain`]).
//! - [`ProcessedChains`]: the chains made for a recording, by key, so a view whose chain starts
//!   with another's computes only its own steps; the whitenings fitted, by the chain up to them.
//! - [`ProcessedSignal`]: a chain as a viewer's signal (`dspk_view::SignalBackend`): windows
//!   processed on the device (with the chain's halo), zoomed-out views reduced there and wide
//!   windows processed in tiles ([`Preview`]), only the columns or the samples read coming back.
//!
//! **How much one block processes** ([`PREVIEW_MAX_VALUES`]) is the device's limit, here and
//! nowhere else: a computer's GPU holds more than a browser's WebGPU will bind. It is lowered to
//! fit the device budget.
//! - [`DeviceBudget`]: everything the views keep on the device, within half of the device's
//!   memory; views off screen give theirs back first.

pub mod budget;
pub mod cached;
mod chains;
pub mod preview;
pub mod processed;

pub use budget::{device_limit, size_class, DeviceBudget};
pub use chains::{FitJob, Made, ProcessedChains};
pub use cached::CachedSource;
pub use preview::{view_plan, InputCache, Preview, PreviewTimings, ViewPlan, SUMMARY_BUCKET};
pub use processed::{PreviewSource, ProcessedSignal};

/// Values (channels × samples, the chain's halo and margin included) one block processes on a
/// computer's GPU (`dspk serve`, the desktop app): 16 Mi float32 = 64 MiB per buffer (32 channels:
/// about 20 s at 24.4 kHz; [`PREVIEW_MAX_TILES`] of them are processed exactly, wider windows come
/// from the processed overview). Blocks of 64 Mi values (256 MiB buffers, 320 MiB in their size
/// class) made the device's memory pools grow pages of up to 2 GiB, and wgpu ran out of memory on
/// an 8 GB card at about 4 GB in use.
pub const PREVIEW_MAX_VALUES_DEVICE: usize = 1 << 24;

/// The same in a browser (a page's engine on WebGPU): 16 Mi float32 = 64 MiB per buffer. A
/// browser's WebGPU binds at most 128 MiB per storage buffer by default and takes the device back
/// past what it can hold, which stops every later drawing (32 channels: about 21 s at 24.4 kHz).
pub const PREVIEW_MAX_VALUES_BROWSER: usize = 1 << 24;

/// The limit where this code runs.
#[cfg(not(target_family = "wasm"))]
pub const PREVIEW_MAX_VALUES: usize = PREVIEW_MAX_VALUES_DEVICE;
#[cfg(target_family = "wasm")]
pub const PREVIEW_MAX_VALUES: usize = PREVIEW_MAX_VALUES_BROWSER;

/// Most tiles of [`PREVIEW_MAX_VALUES`] a zoomed-out view is processed in exactly (each processed
/// and reduced to its columns on the device in turn). A wider window is drawn from the chain's
/// processed overview (`Pyramid`), built from the part on screen outwards as soon as it is asked
/// for (drawn raw until the build starts): processing 50 million samples per channel to draw
/// 1200 columns took seconds and the device's memory.
pub const PREVIEW_MAX_TILES: usize = 4;
