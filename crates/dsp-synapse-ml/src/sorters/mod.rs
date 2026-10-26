//! Spike sorters reimplemented from their papers, each with its [`crate::Provenance`]:
//!
//! - [`kilosort4`]: Kilosort4 (Pachitariu et al., Nature Methods 2024).
//! - [`emusort`]: EMUsort (O'Connell et al., eLife 2026), a Kilosort4 fork for motor units;
//!   reuses the Kilosort4 stages with its own settings and two extra stages.
//! - [`mountainsort5`]: MountainSort 5 (Chung et al., Neuron 2017), isosplit6 clustering.
//! - [`spykingcircus2`]: SpyKING CIRCUS 2 (SpikeInterface's port of SpyKING CIRCUS), matched-filtering
//!   detection, iterative HDBSCAN, circus-omp matching.
//! - [`tridesclous2`]: Tridesclous 2 (SpikeInterface), iterative isosplit, the tdc peeler.
//!
//! Pretrained networks run as released live in [`crate::models`].

pub mod components;
pub mod emusort;
pub mod kilosort4;
pub mod mountainsort5;
pub mod result;
pub mod spykingcircus2;
pub mod tridesclous2;

pub use emusort::{Emusort, EmusortConfig};
pub use kilosort4::{Kilosort4, Kilosort4Config, Kilosort4Result};
pub use mountainsort5::{Mountainsort5, Mountainsort5Config, Mountainsort5Result};
pub use result::{AmplitudeScale, SorterResult};
pub use spykingcircus2::{Spykingcircus2, Spykingcircus2Config, Spykingcircus2Result};
pub use tridesclous2::{Tridesclous2, Tridesclous2Config, Tridesclous2Result};
