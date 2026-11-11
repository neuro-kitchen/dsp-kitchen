//! Icons: gpui-kit's component icons plus the few Lucide icons this app adds (transport,
//! workspaces, view kinds), embedded on their own so the binary does not carry the whole catalog.

use std::borrow::Cow;

use gpui_kit::{AssetSource, SharedString};

gpui_kit::assets::icon_assets!(
    ExtraIcons,
    [AudioWaveform, SlidersHorizontal, Workflow, Microscope, SkipBack, SkipForward, StepBack, StepForward, Repeat, Rows3, Flame, LayoutGrid, Table, Activity, ChartScatter, GitCommitVertical, ZoomIn, ZoomOut, Unlink, Trash, Merge, Tag, Save, Undo2, Redo2, Cpu, ChartGantt, Package, Layers, MousePointer2, Hand, Scissors, ChartColumn, ChartLine, ChartNoAxesColumn, ChartSpline, Grid2x2, MapPin, Settings2, AppWindow, Columns2, Rows2, ListOrdered, Clock, X]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every icon the app's code names (`IconName::Name`, here and in the app plugins) is in the
    /// assets the app ships: a name missing from [`ExtraIcons`] draws nothing and logs an error.
    #[test]
    fn every_icon_used_is_bundled() {
        let roots = [concat!(env!("CARGO_MANIFEST_DIR"), "/src"), concat!(env!("CARGO_MANIFEST_DIR"), "/../../plugins/app")];
        let mut used = std::collections::BTreeSet::new();
        let mut stack: Vec<std::path::PathBuf> = roots.iter().map(Into::into).collect();
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|x| x == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    for marker in ["IconName::", "Assets::"] {
                        for part in text.split(marker).skip(1) {
                            let name: String = part.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
                            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                                used.insert(name);
                            }
                        }
                    }
                }
            }
        }
        let mut missing = Vec::new();
        for name in &used {
            let Some(icon) = gpui_kit::assets::IconName::ALL.iter().find(|i| format!("{i:?}") == *name) else { continue };
            if AppAssets.load(&icon.path()).ok().flatten().is_none() {
                missing.push(name.clone());
            }
        }
        assert!(used.len() > 20, "icons found in the sources: {used:?}");
        assert!(missing.is_empty(), "icons used but not bundled (add them to ExtraIcons): {missing:?}");
    }
}
