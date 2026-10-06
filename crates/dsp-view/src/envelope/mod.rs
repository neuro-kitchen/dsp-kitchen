//! The min/max reduction every envelope goes through.
//!
//! - [`fold`](mod@fold): folds samples into `[min, max]` columns (vectorized, parallel on the rayon pool,
//!   NaN skipped), for channel-major and time-major blocks.
//! - [`buckets`]: min/max of one slice into a fixed number of buckets.
//! - [`device`] (feature `device`): the same columns from samples already on the device, so
//!   only the columns are downloaded.

pub mod buckets;
#[cfg(feature = "device")]
pub mod device;
pub mod fold;

pub use buckets::{min_max_decimate, min_max_decimate_into};
#[cfg(feature = "device")]
pub use device::{envelope_kernel, envelope_on_device};
pub use fold::{Block, Columns, EMPTY, fold, fold_block, fold_row, mean_range, merge};
