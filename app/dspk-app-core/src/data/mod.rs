//! Data layer: the open recording and its sources (`recording`), the shared app state (`store`),
//! what is remembered between runs (`preferences`), and the background services (`work_pool`,
//! `services`). Screens reach it through view models only.

pub mod epochs;
pub mod jobs;
pub mod preferences;
pub mod recording;
pub mod services;
pub mod store;
pub mod work_pool;
