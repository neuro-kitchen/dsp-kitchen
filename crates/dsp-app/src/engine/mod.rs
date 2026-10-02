//! Everything without a UI type: the open recording and its sources, the shared timeline, the
//! time-view state, the rasterizers and the threads that run them. The GPUI layer (store, view
//! models, views) builds on this; `--snapshot` uses it alone.

pub mod axis;
pub mod canvas;
pub mod data;
pub mod palette;
pub mod render_pool;
pub mod time;
