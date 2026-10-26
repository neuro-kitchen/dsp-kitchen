//! The background services every view shares, as a GPUI global: one work pool (renders) and
//! the hover reader.

use gpui_kit::{App, Global};

use crate::engine::time::hover::HoverReader;
use crate::engine::work_pool::WorkPool;

/// Work-pool slot of a view's frame renders.
pub const RENDER_SLOT: u32 = 0;

pub struct Services {
    pub pool: WorkPool,
    pub hover: HoverReader,
}

impl Global for Services {}

impl Services {
    pub fn install(cx: &mut App) {
        cx.set_global(Services { pool: WorkPool::new(WorkPool::default_threads()), hover: HoverReader::spawn() });
    }
}
