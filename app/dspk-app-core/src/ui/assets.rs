//! Icons: gpui-kit's component icons plus the few Lucide icons this app adds (transport,
//! workspaces, view kinds, the status bar's meters), embedded on their own so the binary does not
//! carry the whole catalog; shipped through kitchen-ui's `AppAssets`.

gpui_kit::assets::icon_assets!(
    pub ExtraIcons,
    [AudioWaveform, SlidersHorizontal, Workflow, Microscope, SkipBack, SkipForward, StepBack, StepForward, Repeat, Rows3, Flame, LayoutGrid, Table, Activity, ChartScatter, GitCommitVertical, ZoomIn, ZoomOut, Unlink, Trash, Merge, Tag, Save, Undo2, Redo2, Cpu, ChartGantt, Package, Layers, MousePointer2, Hand, Scissors, ChartColumn, ChartLine, ChartNoAxesColumn, ChartSpline, Grid2x2, MapPin, Settings2, AppWindow, Columns2, Rows2, ListOrdered, Clock, X, Server, Monitor, MemoryStick, Download, Gauge, Grip, Keyboard, Power, Scan, SquareDashed, SunMoon, Palette]
);

/// The neuro-kitchen logo (256 px PNG), for `img(BRAND_LOGO)`: the about dialog.
pub const BRAND_LOGO: &str = "brand/neuro-kitchen-logo.png";
const BRAND_LOGO_PNG: &[u8] = include_bytes!("../../assets/brand/neuro-kitchen-logo.png");

/// What the app ships: gpui-kit's component icons, [`ExtraIcons`] and the logo.
pub fn app_assets() -> kitchen_ui::icons::AppAssets {
    kitchen_ui::icons::AppAssets::new().icons(ExtraIcons).file(BRAND_LOGO, BRAND_LOGO_PNG)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every icon the app's code names (`IconName::Name`, or through an alias, here and in the
    /// app plugins) is in the assets the app ships: a name missing from [`ExtraIcons`] draws
    /// nothing and logs an error.
    #[test]
    fn every_icon_used_is_bundled() {
        let roots = [concat!(env!("CARGO_MANIFEST_DIR"), "/src"), concat!(env!("CARGO_MANIFEST_DIR"), "/../../plugins/app")];
        let used = kitchen_ui::icons::icon_names_used(&roots);
        assert!(used.len() > 20, "icons found in the sources: {used:?}");
        kitchen_ui::icons::assert_icons_bundled(&app_assets(), &roots);
        // The frame's own icons (kitchen-shell: Settings, the search, the theme, the meters)
        let frame: std::collections::BTreeSet<String> = ["Palette", "Search", "Settings", "Sun", "Moon", "Close", "Cpu", "MemoryStick", "Server", "Monitor"].iter().map(|s| s.to_string()).collect();
        assert_eq!(kitchen_ui::icons::missing_icons(&app_assets(), &frame), Vec::<String>::new(), "the frame's icons are bundled");
    }
}
