//! Data layer: the open recording and its sources (`recording`), the shared app state (`store`),
//! what is remembered between runs (`preferences`), and the background services (`work_pool`,
//! `services`), and the servers it works with (`remote`). Screens reach it through view models
//! only.

pub mod channel;
pub mod device;
pub mod jobs;
pub mod live;
pub mod preferences;
pub mod recording;
pub mod remote;
pub mod services;
pub mod settings;
pub mod store;
pub mod work_pool;

// The files saved next to a recording are the session crate's
pub use dspk_session::derived;
pub use dspk_session::derived::{epochs, events};
