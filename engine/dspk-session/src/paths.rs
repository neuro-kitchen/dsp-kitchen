//! Paths as the interface shows them: never the whole path (it names the user and their folders
//! in screenshots and shared logs), only the last two parts.

use std::path::{Component, Path};

/// `…/sessions/run-12.bin` for `/home/<user>/data/sessions/run-12.bin`; a relative path of two
/// parts or fewer as it is. A server path (`dspk://lab/data/x.bin`) keeps its last two parts too.
pub fn short_path(path: &Path) -> String {
    let parts: Vec<String> = path.components().filter(|c| matches!(c, Component::Normal(_))).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    if parts.len() <= 2 && !path.is_absolute() {
        return parts.join("/");
    }
    let tail = &parts[parts.len().saturating_sub(2)..];
    format!("…/{}", tail.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_last_two_parts_show() {
        assert_eq!(short_path(Path::new("/home/someone/data/sessions/run-12.bin")), "…/sessions/run-12.bin");
        assert_eq!(short_path(Path::new("/recording.bin")), "…/recording.bin");
        assert_eq!(short_path(Path::new("sessions/run.bin")), "sessions/run.bin");
        assert_eq!(short_path(Path::new("run.bin")), "run.bin");
        assert_eq!(short_path(Path::new("a/b/c/d.zarr")), "…/c/d.zarr");
        assert_eq!(short_path(Path::new("dspk://lab/data/sessions/x.bin")), "…/sessions/x.bin");
    }
}
