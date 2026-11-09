//! Choosing a recording in the system's file dialog.
//!
//! A recording is a file (`.bin`, `.cbin`) or a folder (`.zarr`, `.nwb.zarr`). macOS and Windows
//! have one dialog for both; the Linux desktop portal shows either files or folders, so there the
//! person says which they are looking for.

use std::path::PathBuf;

use gpui_kit::{App, PathPromptOptions, PromptLevel, Task, Window};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    File,
    Folder,
}

impl Pick {
    /// The dialog's options: `self` where the platform has to choose, both elsewhere.
    pub fn options(self, title: &'static str, multiple: bool, cx: &App) -> PathPromptOptions {
        let both = cx.can_select_mixed_files_and_dirs();
        PathPromptOptions { files: both || self == Pick::File, directories: both || self == Pick::Folder, multiple, prompt: Some(title.into()) }
    }
}

/// The open actions a menu offers, as `(label, what it picks)`.
pub fn open_actions(cx: &App) -> &'static [(&'static str, Pick)] {
    if cx.can_select_mixed_files_and_dirs() {
        &[("Open recording…", Pick::File)]
    } else {
        &[("Open file…", Pick::File), ("Open folder…", Pick::Folder)]
    }
}

/// Paths chosen for an action that has only one entry point: on Linux it first asks whether to
/// look for a file or a folder. `None` when cancelled or when the dialog fails.
pub fn ask_and_pick(title: &'static str, multiple: bool, window: &mut Window, cx: &mut App) -> Task<Option<Vec<PathBuf>>> {
    let answer = (!cx.can_select_mixed_files_and_dirs())
        .then(|| window.prompt(PromptLevel::Info, title, Some("Files: .bin, .cbin. Folders: .zarr, .nwb.zarr."), &["File", "Folder", "Cancel"], cx));
    cx.spawn(async move |cx| {
        let pick = match answer {
            None => Pick::File,
            Some(answer) => match answer.await.ok()? {
                0 => Pick::File,
                1 => Pick::Folder,
                _ => return None,
            },
        };
        let paths = cx.update(|cx| cx.prompt_for_paths(pick.options(title, multiple, cx)));
        paths.await.ok()?.ok()?
    })
}
