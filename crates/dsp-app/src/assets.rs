//! Icons: gpui-kit's component icons plus the few Lucide icons this app adds (transport,
//! workspaces, view kinds), embedded on their own so the binary does not carry the whole catalog.

use std::borrow::Cow;

use gpui_kit::{AssetSource, SharedString};

gpui_kit::assets::icon_assets!(
    ExtraIcons,
    [AudioWaveform, SlidersHorizontal, Workflow, Microscope, SkipBack, SkipForward, StepBack, StepForward, Repeat, Rows3, Flame]
);

pub struct AppAssets;

impl AssetSource for AppAssets {
    fn load(&self, path: &str) -> gpui_kit::Result<Option<Cow<'static, [u8]>>> {
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
