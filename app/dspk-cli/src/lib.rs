//! The `dspk` command line. The binary is `src/main.rs`; this library exposes its commands and
//! terminal helpers so other programs reuse them (`dspk-cli-sorter` takes its recording and output
//! arguments and its progress bar from here).

pub mod commands;
pub mod progress;
pub mod sorters_program;
pub mod style;
