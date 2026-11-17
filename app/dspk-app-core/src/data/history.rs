//! Who works: the people ([`kitchen_shell::people`]), read from `<config>/dspk-app/people.json`.
//! What is saved names them (a curation's edits); a guest is named nowhere. The app's activity log
//! is the session's own (in memory, shown in Analytics ▸ This session) and is written nowhere.

/// Reads the people. Once, at start (a page in a browser keeps them for its life only).
pub fn install() {
    #[cfg(not(target_family = "wasm"))]
    if let Some(path) = crate::data::preferences::Session::people_path()
        && let Some(notice) = kitchen_shell::people::load(path)
    {
        tracing::warn!("people: {notice}");
    }
}
