//! Physical probe geometry: contact positions, shanks, nominal presets, and nearest-site queries.

pub mod neighbors;
pub mod presets;
pub mod site;
pub mod sensor;
mod source;

pub use site::{Position3D, SensorSite};
pub use sensor::SensorLayout;
pub use neighbors::{find_k_nearest_neighbors, precompute_knn_table};
pub use presets::{hdemg_4x8, hdemg_8x8, hdemg_grid, neuropixels_1_0, neuropixels_2_0, tetrode, utah_array};
pub use source::{probe_of, ProbeSource};
