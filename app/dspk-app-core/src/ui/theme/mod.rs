//! The app's look: the neuro-kitchen design tokens from `kitchen_ui::theme` (spacing, type, radii,
//! plain-grey neutrals, the accent and its WCAG checks) with dsp-kitchen's lime accent, the sizes
//! only this app has, and the plot palettes ([`palette`]: plot data colours, not UI tokens).
//!
//! kitchen-ui's accent palette is `kitchen_ui::Palette`; [`palette::Palette`] here is the plots'.

pub mod palette;

use gpui_kit::{App, Window};
pub use kitchen_ui::theme::{
    contrast, is_dark, rgb_hsla, sp, system_dark, Look, Rgb, FONT_RANGE, FONT_SIZE, LG, MD, PANE_GAP, PANE_PADDING, RADIUS, RADIUS_LG, RADIUS_SM, SM, TARGET_MIN, XL, XS, XXL,
};

/// Height of the status bar (px).
pub const STATUS_BAR_H: f32 = 30.;

/// dsp-kitchen's accent: lime (the user can pick another in Settings).
pub const ACCENT: kitchen_ui::Palette = kitchen_ui::Palette::LIME;

/// Sets the look: dark or light (`dark`, or the system's appearance when `None`), and the
/// user's accent (`None`: lime). [`apply`] then hands it to gpui-kit.
pub fn configure(dark: Option<bool>, accent: Option<Rgb>, window: &Window) {
    let dark = dark.unwrap_or_else(|| system_dark(window));
    kitchen_ui::theme::set_look(Look::new(ACCENT).with_dark(dark).with_accent(accent));
}

/// Applies the look ([`configure`]) to gpui-kit and redraws.
pub fn apply(window: &mut Window, cx: &mut App) {
    kitchen_ui::theme::apply(cx);
    window.refresh();
}

/// The colours: kitchen-ui's neutrals and accent.
pub mod color {
    pub use kitchen_ui::theme::color::*;
}
