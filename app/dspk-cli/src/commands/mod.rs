//! One module per command (file name = command name); each defines its arguments and `run`.

pub mod benchmark;
pub mod convert;
pub mod detect;
pub mod doctor;
pub mod generate;
#[cfg(feature = "hub")]
pub mod hub;
pub mod inspect;
pub mod net;
pub mod open;
pub mod probe;
pub mod recording;
pub mod sort;
