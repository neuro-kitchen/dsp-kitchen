//! Dependency wiring: installs the services every screen shares (render work pool, hover reader)
//! as GPUI globals. The shared state (`data::store::Store`) is created by `main` and handed to
//! the root view.

use gpui_kit::App;

use crate::data::services::Services;

/// Installs the shared services.
pub fn install(cx: &mut App) {
    Services::install(cx);
}
