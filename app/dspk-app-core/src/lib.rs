//! DSP App core: the workbench for any multichannel signal, and the plugin interface domain
//! plugins extend (electrophysiology: `dspk-app-synapse`). The `dspk-app` binary is this core
//! plus the plugins it is built with.
//!
//! Layout (Kotlin Compose Multiplatform style, as lumen-app): `commands` (every action, declared
//! once for kitchen-shell's menus, keys, search and agents), `di` (wiring), `data` (recording,
//! shared store, preferences, work pool, jobs), `domain` (plain models: session document,
//! commands, timeline), `navigation` (routes, actions), `screens/<feature>` (state, events, view
//! model, screen, components), `ui` (theme, plot primitives, widgets), `plugin` (what plugins
//! add: formats, channel layouts, trace overlays, workspaces, panel sections).

pub mod app;
pub mod commands;
pub mod data;
pub mod di;
pub mod domain;
pub mod navigation;
pub mod plugin;
pub mod screens;
pub mod ui;
