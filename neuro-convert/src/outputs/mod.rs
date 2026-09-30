//! Output formats: each folder writes a [`crate::model::Session`] to one target.

#[cfg(feature = "nwb")]
pub mod nwb;
