//! Peak finding, as `scipy.signal.find_peaks`:
//!
//! - [`find_peaks`] (host): local maxima, minima or both of one signal, with scipy's `height`,
//!   `threshold`, `distance`, `prominence` and `width` conditions and properties.
//! - [`find_peak_candidates`] (device): the local extrema above a per-channel height of every
//!   channel of a device buffer, compacted on the device; [`select_by_distance`] then applies
//!   `distance` on the host.
//! - [`DistanceRule`]: `distance` as scipy (default) or locally exclusive (chunk-exact).

mod device;
mod host;
pub mod kernels;

pub use device::{find_peak_candidates, find_peak_candidates_on_device, DevicePeakCandidates, PeakCandidates};
pub use host::{
    find_peaks, local_extrema, select_by_distance, DistanceRule, Interval, PeakOptions, Peaks, Polarity, DEFAULT_REL_HEIGHT,
};
