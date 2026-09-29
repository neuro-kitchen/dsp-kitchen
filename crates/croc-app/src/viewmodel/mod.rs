//! ViewModel layer for croc-app.

pub mod app_viewmodel;
pub mod dock;
pub mod persist;
pub mod view_state;

pub use app_viewmodel::{AppViewModel, AXIS, GUTTER, SEAT_HEADER};
pub use dock::{DropSide, ViewId};
pub use persist::Session;
