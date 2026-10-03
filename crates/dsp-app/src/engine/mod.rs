//! Everything without a UI type: the open recording and its sources, the shared timeline, the
//! time-view state, the rasterizers and the threads that run them. The GPUI layer (store, view
//! models, views) builds on this; `--snapshot` uses it alone.

pub mod canvas;
// Curation (steps 7–10) is built and tested ahead of its views; step 8 wires it into the UI
#[allow(dead_code)]
pub mod compute;
#[allow(dead_code, unused_imports)]
pub mod curation;
pub mod data;
pub mod palette;
pub mod work_pool;
pub mod time;
