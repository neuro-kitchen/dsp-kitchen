//! The `dspk` command line. The binary is `src/main.rs`; this library exposes its commands and
//! terminal helpers so other programs reuse them (`dspk-sorters` takes its recording and output
//! arguments and its progress bar from here).

pub mod commands;
pub mod progress;
pub mod style;
