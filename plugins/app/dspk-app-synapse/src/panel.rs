//! The Processing panel's Sorting section for a view: the linked `.sorting.zarr` (attach,
//! detach), its units as a legend with toggles (shown on the traces or not), and Sort (a sorter
//! and a probe, run by `dspk-cli-sorter`; the result is linked to the view).

use std::path::PathBuf;

use dspk_app_core::data::jobs::processed_path;
use dspk_app_core::data::store::Store;
use dspk_session::command::SessionCommand;
use dspk_session::session::ViewDoc;
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::widgets::{MenuSelect, Section};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, PathPromptOptions, SharedString, StatefulInteractiveElement as _, Styled as _, Window};

use crate::sort::{self, SortRequest, PROBES, SORTERS};
use crate::{sorting_of, state, ID};

/// The section for `view`.
pub fn section(store: &Entity<Store>, view: &ViewDoc, _window: &mut Window, cx: &mut App) -> AnyElement {
    let muted = cx.theme().muted_foreground;
    let Some(state) = state(cx) else { return div().into_any_element() };
    let linked = sorting_of(view);
    let loaded = state.read(cx).active_sorting(cx);
    let running = store.read(cx).job.is_some();
    let mut section = Section::new("Linked sorting");

    // The linked sorting and its units
    match (&linked, &loaded) {
        (Some(path), Some(sorting)) => {
            let palette = store.read(cx).palette();
            let mut order: Vec<usize> = sorting.alive().collect();
            order.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
            let rows = order.into_iter().map(|u| {
                let unit = &sorting.units[u];
                let hidden = state.read(cx).is_hidden(path, u);
                let (state, path) = (state.clone(), path.clone());
                h_flex()
                    .id(SharedString::from(format!("unit-{u}")))
                    .h(px(22.))
                    .px_1()
                    .gap_2()
                    .cursor_pointer()
                    .rounded_sm()
                    .child(div().size(px(9.)).rounded_full().when(!hidden, |d| d.bg(color(palette.unit(u)))).border_1().border_color(color(palette.unit(u))))
                    .child(div().w(px(40.)).text_xs().when(hidden, |d| d.text_color(muted)).child(unit.info.unit_id.to_string()))
                    .child(div().flex_1().min_w_0().truncate().text_xs().text_color(muted).child(unit.info.quality_label.clone()))
                    .child(div().text_xs().text_color(muted).child(unit.info.num_spikes.to_string()))
                    .on_click(move |_, _, cx| state.update(cx, |s, cx| s.toggle_unit(&path, u, cx)))
            });
            let (s1, s2, p1, p2) = (state.clone(), state.clone(), path.clone(), path.clone());
            section = section
                .child(div().text_xs().child(sorting.summary()))
                .child(div().text_xs().text_color(muted).truncate().child(path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Button::new("units-all").ghost().xsmall().label("Show all").on_click(move |_, _, cx| s1.update(cx, |s, cx| s.show_all(&p1, true, cx))))
                        .child(Button::new("units-none").ghost().xsmall().label("Hide all").on_click(move |_, _, cx| s2.update(cx, |s, cx| s.show_all(&p2, false, cx))))
                        .child(div().flex_1())
                        .child(detach(store, view)),
                )
                .child(v_flex().id("unit-legend").max_h(px(220.)).overflow_y_scroll().children(rows))
                .child(div().text_xs().text_color(muted).child("Click a unit to show or hide it (windows ≤ 30 s)."));
        }
        (Some(path), None) => {
            section = section.child(div().text_xs().text_color(muted).child(format!("Loading {}…", dspk_session::paths::short_path(&path)))).child(detach(store, view));
        }
        (None, _) => {
            section = section.child(div().text_xs().text_color(muted).child("No sorting linked to this view."));
        }
    }
    section = section.child(div().text_xs().text_color(muted).child("Add or change it in the Timeline panel: + → Sorting…"));

    // Sort
    let (sorter, probe) = (state.read(cx).sorter, state.read(cx).probe);
    let s1 = state.clone();
    let sorter_pick = MenuSelect::new("sorter", SORTERS.iter().map(|(_, n)| SharedString::from(*n)).collect(), Some(sorter), move |i, _, cx| {
        s1.update(cx, |s, cx| {
            s.sorter = i;
            s.choices_changed(cx);
        })
    });
    let s2 = state.clone();
    let probe_pick = MenuSelect::new("probe", PROBES.iter().map(|(_, n)| SharedString::from(*n)).collect(), Some(probe), move |i, _, cx| {
        s2.update(cx, |s, cx| {
            s.probe = i;
            s.choices_changed(cx);
        })
    });
    let processed = view.ops.iter().any(|o| o.enabled);
    let input_note = if processed {
        "Sorts the written processed recording (Write it first)."
    } else {
        "Sorts the recording as stored."
    };
    let (store_c, view_c) = (store.clone(), view.clone());
    let run = Button::new("sort-view").small().primary().icon(IconName::Play).label("Sort").disabled(running).on_click(move |_, _, cx| start_sort(&store_c, &view_c, sorter, probe, cx));
    v_flex()
        .gap_6()
        .child(section)
        .child(
            Section::new("Sort")
                .child(dspk_app_core::ui::widgets::row("Sorter", sorter_pick))
                .child(dspk_app_core::ui::widgets::row("Probe", probe_pick))
                .child(run)
                .child(div().text_xs().text_color(muted).child(input_note)),
        )
        .into_any_element()
}

fn detach(store: &Entity<Store>, view: &ViewDoc) -> impl IntoElement {
    let (store, view) = (store.clone(), view.clone());
    Button::new("detach-sorting").ghost().xsmall().label("Detach").on_click(move |_, _, cx| {
        let data = crate::plugin_data(&view, None, None);
        let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetPluginData { view: view.id.clone(), plugin: ID.into(), data }, cx));
    })
}

/// Picks a `.sorting.zarr` folder, or a Phy / Kilosort folder (imported as a `.sorting.zarr`
/// next to it, once), links it to view `view` and opens it in Curation.
pub fn pick_and_attach(store: &Entity<Store>, view: String, cx: &mut App) {
    // A browser page: the folder's files are read into memory, the sorting computed on here
    if dspk_app_core::platform::remote::runs_alone() {
        pick_into_page(store, view, cx);
        return;
    }
    if !dspk_app_core::platform::LOCAL_FILES {
        store.update(cx, |s, cx| s.set_status("Sortings open in the desktop app for now", cx));
        return;
    }
    let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some("Open a sorting (.sorting.zarr or Phy folder)".into()) });
    let store = store.clone();
    cx.spawn(async move |cx| {
        let Ok(Ok(Some(paths))) = rx.await else { return };
        let Some(path) = paths.into_iter().next() else { return };
        let path = if dspk_synapse::io::phy::PhyFolder::is_phy_folder(&path) {
            let folder = path.clone();
            match cx.background_executor().spawn(async move { dspk_synapse::curation::phy::import_phy(&folder) }).await {
                Ok(imported) => {
                    let note = format!("Phy folder imported as {}", dspk_session::paths::short_path(&imported));
                    let _ = store.update(cx, |s, cx| s.set_status(note, cx));
                    imported
                }
                Err(e) => {
                    let _ = store.update(cx, |s, cx| s.set_status(format!("Phy folder not imported: {e:#}"), cx));
                    return;
                }
            }
        } else {
            path
        };
        // A sorting is curated: Curation shows it
        let _ = store.update(cx, |s, cx| {
            link(s, &view, path, cx);
            s.set_workspace(dspk_app_core::navigation::route::Workspace::Curation, cx);
        });
    })
    .detach();
}

/// A browser page: picks a `.sorting.zarr` folder, reads its files into memory, loads the sorting
/// from them (on a worker thread) and links it to `view` under the folder's name; Curation opens.
fn pick_into_page(store: &Entity<Store>, view: String, cx: &mut App) {
    let Some(state) = state(cx) else { return };
    let store = store.clone();
    cx.spawn(async move |cx| {
        let picked = dspk_app_core::platform::remote::pick_folder_bytes().await;
        let (name, files) = match picked {
            Ok(Some(folder)) => folder,
            Ok(None) => return,
            Err(e) => {
                let _ = store.update(cx, |s, cx| s.set_status(format!("Sorting not read: {e:#}"), cx));
                return;
            }
        };
        if !name.to_ascii_lowercase().ends_with(".sorting.zarr") {
            let _ = store.update(cx, |s, cx| s.set_status(format!("{name} is not a .sorting.zarr: pick the sorting's folder (Phy folders: import them in the desktop app)"), cx));
            return;
        }
        let _ = store.update(cx, |s, cx| s.set_status(format!("Reading {name}…"), cx));
        let task_name = name.clone();
        let loaded = cx.background_executor().spawn(async move { dspk_synapse::curation::sorting::SortingData::load_files(&task_name, files) }).await;
        let path = PathBuf::from(&name);
        match loaded {
            Ok(data) => {
                let _ = state.update(cx, |st, cx| st.loaded(path.clone(), data, Ok(Default::default()), cx));
                let _ = store.update(cx, |s, cx| {
                    link(s, &view, path, cx);
                    s.set_workspace(dspk_app_core::navigation::route::Workspace::Curation, cx);
                });
            }
            Err(e) => {
                let _ = store.update(cx, |s, cx| s.set_status(format!("{name}: {e:#}"), cx));
            }
        }
    })
    .detach();
}

/// A page served by a server: links the sorting at `path` (under the server's data root) to the
/// active view and opens Curation; the plugin's state reads it from the server.
pub fn open_server_sorting(store: &Entity<Store>, path: String, cx: &mut App) {
    if store.read(cx).active_view().is_none() {
        store.update(cx, |s, cx| s.set_status("Open a recording first: a sorting belongs to one", cx));
        return;
    }
    let path = PathBuf::from(path);
    if let Some(st) = state(cx) {
        st.update(cx, |st, cx| st.load(path.clone(), cx));
    }
    store.update(cx, |s, cx| {
        if let Some(view) = s.active_view().map(|v| v.id.clone()) {
            link(s, &view, path, cx);
        }
        s.set_workspace(dspk_app_core::navigation::route::Workspace::Curation, cx);
    });
}

/// The Timeline panel's Sortings section: the active view's sorting (open it in Curation, or
/// detach it), or how to add one.
pub fn timeline_section(store: &Entity<Store>, cx: &mut App) -> AnyElement {
    let muted = cx.theme().muted_foreground;
    let Some(view) = store.read(cx).active_view().cloned() else {
        return div().px_1().text_xs().text_color(muted).child("Open a recording first").into_any_element();
    };
    let Some(path) = sorting_of(&view) else {
        return div().px_1().text_xs().text_color(muted).child("None yet: + → Sorting…").into_any_element();
    };
    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    let store2 = store.clone();
    h_flex()
        .id("timeline-sorting")
        .gap_1()
        .px_1()
        .items_center()
        .child(gpui_kit::component::Icon::new(gpui_kit::assets::IconName::Microscope).xsmall().text_color(muted))
        .child(div().flex_1().min_w_0().truncate().text_sm().child(name))
        .child(Button::new("timeline-curate").ghost().xsmall().label("Curate").tooltip("Open it in Curation").on_click(move |_, _, cx| {
            store2.update(cx, |s, cx| s.set_workspace(dspk_app_core::navigation::route::Workspace::Curation, cx));
        }))
        .child(detach(store, &view))
        .into_any_element()
}

/// Links `path` to view `view` (its runs kept).
pub fn link(s: &mut Store, view: &str, path: PathBuf, cx: &mut gpui_kit::Context<Store>) {
    let Some(v) = s.doc.view(view).cloned() else { return };
    let data = crate::plugin_data(&v, Some(&path), None);
    let _ = s.execute(SessionCommand::SetPluginData { view: view.to_string(), plugin: ID.into(), data }, cx);
}

/// Sorts `view` with sorter `sorter` and probe preset `probe` ([`SORTERS`], [`PROBES`]).
fn start_sort(store: &Entity<Store>, view: &ViewDoc, sorter: usize, probe: usize, cx: &mut App) {
    let (key, name) = SORTERS[sorter];
    let processed = view.ops.iter().any(|o| o.enabled);
    let (recording, source) = if processed {
        let written = processed_path(&view.recording.path, &view.id);
        if !written.exists() {
            store.update(cx, |s, cx| s.set_status(format!("Write the processed recording of {} first (Output → Write): sorters read files", view.name), cx));
            return;
        }
        (written, None)
    } else {
        (view.recording.path.clone(), view.recording.source.clone())
    };
    let output = sort::sorting_path(&recording, key);
    let req = SortRequest { sorter: key.to_string(), recording, source, probe: PROBES[probe].0.map(str::to_string), output, params: Vec::new(), start_sec: None, duration_sec: None };
    let id = view.id.clone();
    store.update(cx, |s, cx| {
        s.start_job(format!("{name} on {}", view.name), move |send| sort::sort(&req, move |m| send(m)), move |s, path, cx| link(s, &id, path, cx), cx);
    });
}
