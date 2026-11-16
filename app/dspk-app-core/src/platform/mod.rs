//! What differs between the desktop and a browser tab, in one place (as Kotlin Multiplatform's
//! `expect` / `actual`): each target's file gives the same names, and the rest of the app calls
//! them and has no `cfg` of its own.
//!
//! | | desktop | browser, threaded build | browser, single thread |
//! |---|---|---|---|
//! | threads | the system's | web workers | none: work runs where it is asked for |
//! | a lock on the UI thread | waits | may not wait: spins | spins |
//! | files | the disk, a file dialog | the page's server | files the user picks |
//! | servers | QUIC, a profile per server | the page's own server (WebSocket) | the same |

#[cfg(not(target_family = "wasm"))]
mod desktop;
#[cfg(not(target_family = "wasm"))]
pub use desktop::*;

#[cfg(target_family = "wasm")]
mod web;
#[cfg(target_family = "wasm")]
pub use web::*;

/// Connecting, listing, opening, fetching and remote jobs on this target.
#[cfg(not(target_family = "wasm"))]
#[path = "desktop_remote.rs"]
pub mod remote;
#[cfg(target_family = "wasm")]
#[path = "web_remote.rs"]
pub mod remote;

/// Locks `mutex`. On a browser page's main thread a lock may not wait (the page would stop), so
/// there it is tried until it is free: the app's shared locks are held for a few instructions.
pub fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    if cfg!(target_family = "wasm") {
        loop {
            match mutex.try_lock() {
                Ok(guard) => return guard,
                Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
    }
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}
