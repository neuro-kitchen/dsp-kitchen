//! The background services every view shares, as a GPUI global: one work pool (renders and
//! curation computations), the hover reader, and the cache of derived curation data.

use gpui_kit::{App, Global};

use crate::engine::compute::ComputeCache;
use crate::engine::time::hover::HoverReader;
use crate::engine::work_pool::WorkPool;

/// Work-pool slot of a view's frame renders (curation computations use theirs, see
/// [`crate::engine::compute::ComputeKind::slot`]).
pub const RENDER_SLOT: u32 = 0;

pub struct Services {
    pub pool: WorkPool,
    pub hover: HoverReader,
    #[allow(dead_code)] // Curation views request through it (step 8)
    pub cache: ComputeCache,
}

impl Global for Services {}

impl Services {
    pub fn install(cx: &mut App) {
        cx.set_global(Services { pool: WorkPool::new(WorkPool::default_threads()), hover: HoverReader::spawn(), cache: ComputeCache::new() });
    }
}
