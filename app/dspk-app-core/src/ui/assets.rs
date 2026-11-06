//! Icons: gpui-kit's component icons plus the few Lucide icons this app adds (transport,
//! workspaces, view kinds), embedded on their own so the binary does not carry the whole catalog.

use std::borrow::Cow;

use gpui_kit::{AssetSource, SharedString};

gpui_kit::assets::icon_assets!(
    ExtraIcons,
    [AudioWaveform, SlidersHorizontal, Workflow, Microscope, SkipBack, SkipForward, StepBack, StepForward, Repeat, Rows3, Flame, LayoutGrid, Table, Activity, ChartScatter, GitCommitVertical, ZoomIn, ZoomOut, Unlink, Trash, Merge, Tag, Save, Undo2, Redo2, Cpu, ChartGantt, Package, Layers, MousePointer2, Hand, Scissors, ChartColumn, ChartLine, ChartNoAxesColumn, ChartSpline, Grid2x2, MapPin, Settings2]
);

/// The neuro-kitchen logo (256 px PNG), for `img(BRAND_LOGO)`: the about dialog.
pub const BRAND_LOGO: &str = "brand/neuro-kitchen-logo.png";
const BRAND_LOGO_PNG: &[u8] = include_bytes!("../../assets/brand/neuro-kitchen-logo.png");

pub struct AppAssets;

impl AssetSource for AppAssets {
    fn load(&self, path: &str) -> gpui_kit::Result<Option<Cow<'static, [u8]>>> {
        if path == BRAND_LOGO {
            return Ok(Some(Cow::Borrowed(BRAND_LOGO_PNG)));
        }
        if let Some(bytes) = ExtraIcons.load(path)? {
            return Ok(Some(bytes));
        }
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> gpui_kit::Result<Vec<SharedString>> {
        let mut paths = gpui_kit::assets::Assets.list(path)?;
        paths.extend(ExtraIcons.list(path)?);
        paths.sort();
        paths.dedup();
        Ok(paths)
    }
}
