//! What SpyKING CIRCUS 2 and Tridesclous 2 share beyond `dspk-synapse`: SpikeInterface's isosplit
//! ([`isosplit`]) and waveform cutting around peaks ([`waveforms`]).

pub mod isosplit;
pub mod waveforms;

pub use isosplit::{isocut, isosplit_from_labels, isosplit_si, kmeans2_points, IsosplitOptions};
pub use waveforms::PeakWaveforms;
