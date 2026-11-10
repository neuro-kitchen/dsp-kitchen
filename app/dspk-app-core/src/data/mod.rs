//! Data layer: the open recording and its sources (`recording`), the shared app state (`store`),
//! what is remembered between runs (`preferences`), and the background services (`work_pool`,
//! `services`), and the servers it works with (`remote`). Screens reach it through view models
//! only.

pub mod derived;
pub mod epochs;
pub mod events;
pub mod jobs;
pub mod live;
pub mod preferences;
pub mod recording;
pub mod remote;
pub mod services;
pub mod store;
pub mod work_pool;
