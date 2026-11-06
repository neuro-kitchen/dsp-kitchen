//! One time view (traces or heatmap): its state (`view`), rasterizer (`renderer`), hover readouts
//! (`hover`), view model (`trace_vm`) and panel (`trace_panel`).

pub mod hover;
pub mod renderer;
pub mod trace_panel;
pub mod trace_vm;
pub mod view;

pub use trace_vm::TraceVm;
