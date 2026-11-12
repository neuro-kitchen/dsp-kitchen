//! dsp-kitchen's commands, declared once (kitchen-shell's [`Registry`]): each has an id, a title,
//! an icon, a line saying what it does, a group, a key, when it can run (and why not) and what it
//! does. The menus, the keys, the search (Ctrl+K) and an agent's tools all come from this list.
//!
//! Ids are dotted by area: `file.*`, `edit.*`, `view.*`, `timeline.*`, `process.*`, `help.*`,
//! `remote.*`. Entries that follow what the session holds (recent files, recipes) are menu
//! providers running a command with arguments. The Pipeline's own commands live with it
//! ([`crate::screens::pipeline::PipelineView::commands`]).

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::base::dock::DockPlacement;
use gpui_kit::{App, AppContext as _, Entity, SharedString, WeakEntity, Window};
use kitchen_shell::{CommandSpec, MenuItem, MenuSpec, Registry, Shell};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::data::store::{file_label, Store};
use crate::navigation::route::Workspace;
use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::ExploreView;
use crate::screens::pipeline::catalog::catalog;
use crate::screens::pipeline::PipelineView;
use crate::ui::recording_picker::Pick;

/// The synthetic recording of the start screen and the File menu: channels, sample rate, seconds.
pub const SYNTHETIC: (usize, f64, f64) = (32, 30_000.0, 300.0);

/// What the commands act on.
#[derive(Clone)]
pub struct Ctx {
    pub store: Entity<Store>,
    /// The frame (set once it exists: commands only run after).
    pub shell: Rc<std::cell::OnceCell<WeakEntity<Shell>>>,
    pub explore: WeakEntity<ExploreView>,
    pub pipeline: WeakEntity<PipelineView>,
    /// Shows or hides the Keyboard & mouse sheet, and About.
    pub toggle_help: Rc<dyn Fn(&mut Window, &mut App)>,
    pub toggle_about: Rc<dyn Fn(&mut Window, &mut App)>,
    /// Opens "Open from server" over the frame.
    pub open_browser: Rc<dyn Fn(&mut Window, &mut App)>,
}

impl Ctx {
    /// Runs `f` on the frame.
    pub fn shell(&self, cx: &mut App, f: impl FnOnce(&mut Shell, &mut gpui_kit::Context<Shell>)) {
        if let Some(s) = self.shell.get().and_then(|w| w.upgrade()) {
            s.update(cx, f);
        }
    }
}

/// The command id of an operation of the catalog (`process.op.bandpass`).
pub fn op_id(label: &str) -> String {
    let slug: String = label.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("process.op.{}", slug.split('_').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("_"))
}

/// The workspace ids the frame uses (`shell.workspace.<id>`).
pub fn workspace_id(w: Workspace) -> SharedString {
    w.title().to_lowercase().into()
}

/// Asks for a recording (a file, or a folder) and opens it.
pub fn prompt_open(store: &Entity<Store>, pick: Pick, cx: &mut App) {
    let rx = cx.prompt_for_paths(pick.options("Open recording", false, cx));
    let store = store.clone();
    cx.spawn(async move |cx| match rx.await {
        Ok(Ok(Some(paths))) => {
            if let Some(path) = paths.into_iter().next() {
                store.update(cx, |s, cx| s.open(path, cx));
            }
        }
        Ok(Err(e)) => store.update(cx, |s, cx| s.set_status(format!("File dialog failed: {e}"), cx)),
        _ => {}
    })
    .detach();
}

/// Opens the synthetic recording.
pub fn open_synthetic(store: &Entity<Store>, cx: &mut App) {
    let (ch, rate, dur) = SYNTHETIC;
    store.update(cx, |s, cx| s.open_synthetic(ch, rate, dur, cx));
}

/// Tunes this device in the background, so first runs and previews don't stop to benchmark.
fn tune_device(store: &Entity<Store>, cx: &mut App) {
    store.update(cx, |s, cx| s.set_status("Tuning this device in the background (about a minute on a GPU)…", cx));
    let store = store.clone();
    cx.spawn(async move |cx| {
        let tuned = cx
            .background_spawn(async move {
                let client = dspk_core::ComputeTarget::from_env().map_err(|e| e.to_string())?.client().map_err(|e| e.to_string())?;
                dspk_ops::autotune::autotune(&client, &Default::default(), |_, _| {}).map_err(|e| e.to_string())
            })
            .await;
        let text = match tuned {
            Ok(done) => format!("This device is tuned: {} sizes in {:.0} s", done.sizes, done.seconds),
            Err(e) => format!("Could not tune this device: {e}"),
        };
        let _ = store.update(cx, |s, cx| s.set_status(text, cx));
    })
    .detach();
}

fn needs_recording(store: &Entity<Store>) -> impl Fn(&App) -> Result<(), String> + 'static {
    let store = store.clone();
    move |cx| if store.read(cx).recording.is_some() { Ok(()) } else { Err("Open a recording first".into()) }
}

/// A file to open.
#[derive(Deserialize, JsonSchema)]
struct PathArgs {
    /// A recording (a file or a folder).
    path: String,
}

/// A recipe.
#[derive(Deserialize, JsonSchema)]
struct RecipeArgs {
    /// Its name (built in, or of the session).
    name: String,
}

/// Registers dsp-kitchen's commands and menu providers (the Pipeline registers its own).
pub fn register(registry: &Registry, c: &Ctx) {
    let s = &c.store;
    let st = |f: fn(&mut Store, &mut gpui_kit::Context<Store>)| {
        let store = s.clone();
        move |_: &mut Window, cx: &mut App| store.update(cx, f)
    };
    let (o1, o2, o3, o4, o5) = (s.clone(), s.clone(), s.clone(), s.clone(), s.clone());
    let (help, about, browser) = (c.toggle_help.clone(), c.toggle_about.clone(), c.open_browser.clone());
    let recent = s.clone();
    let mut list = vec![
        // File
        CommandSpec::new("file.open", "Open recording…", IconName::FolderOpen, move |_, cx| prompt_open(&o1, Pick::File, cx)).group("File").hint("A recording from this computer: .bin, .cbin, .zarr, .nwb.zarr").key("secondary-o"),
        CommandSpec::new("file.open_folder", "Open folder…", IconName::FolderOpen, move |_, cx| prompt_open(&o2, Pick::Folder, cx)).group("File").hint("A recording stored as a folder (Zarr)").key("secondary-shift-o"),
        CommandSpec::with_args::<PathArgs>("file.open_path", "Open a file", IconName::FolderOpen, move |a, _, cx| {
            o3.update(cx, |s, cx| s.open(a.path.into(), cx));
            Ok(Value::Null)
        })
        .group("File")
        .hint("Open a recording by its path"),
        CommandSpec::new("file.open_server", "Open from server…", IconName::Server, move |w, cx| browser(w, cx)).group("File").hint("Browse the recordings a dspk server offers"),
        CommandSpec::new("file.synthetic", "Synthetic recording", IconName::AudioWaveform, move |_, cx| open_synthetic(&o4, cx)).group("File").hint("32 channels, 5 minutes: to try things out"),
        CommandSpec::new("file.tune", "Tune this device", IconName::Gauge, move |_, cx| tune_device(&o5, cx)).group("File").hint("Benchmark this GPU once, in the background"),
        CommandSpec::new("file.clear_recent", "Clear recent", IconName::Trash, st(|s, cx| s.forget_recent(None, cx))).group("File").hint("Forget the recent lists (the files stay)").enabled(move |cx| {
            let s = &recent.read(cx).session;
            if s.recent.is_empty() && s.recent_sortings.is_empty() { Err("Nothing to clear".into()) } else { Ok(()) }
        }),
        CommandSpec::new("file.close_recording", "Close recording", IconName::X, st(|s, cx| s.close_recording(None, cx))).group("File").hint("Close the recording shown").enabled(needs_recording(s)),
        CommandSpec::new("app.quit", "Quit", IconName::Power, |_, cx| cx.quit()).group("File").hint("Close dsp-kitchen").key("secondary-q"),
        // Edit
        CommandSpec::new("edit.undo", "Undo", IconName::Undo2, st(|s, cx| s.undo(false, cx))).group("Edit").hint("Undo the last change of the session").key("secondary-z"),
        CommandSpec::new("edit.redo", "Redo", IconName::Redo2, st(|s, cx| s.undo(true, cx))).group("Edit").hint("Redo what was undone").key("secondary-shift-z"),
    ];
    // View: the views of Explore and its docks
    let explore = c.explore.clone();
    let add = |kind: TimeViewKind| {
        let (store, explore) = (s.clone(), explore.clone());
        move |_: &mut Window, cx: &mut App| {
            store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx));
            let _ = explore.update(cx, |e, cx| e.add_view(kind, cx));
        }
    };
    let dock = |placement: DockPlacement| {
        let (store, explore) = (s.clone(), explore.clone());
        move |window: &mut Window, cx: &mut App| {
            store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx));
            let _ = explore.update(cx, |e, cx| e.toggle_dock(placement, window, cx));
        }
    };
    let (d1, d2, f1, f2) = (s.clone(), s.clone(), s.clone(), s.clone());
    list.extend([
        CommandSpec::new("view.add_traces", "Add a traces view", IconName::AudioWaveform, add(TimeViewKind::Traces)).group("View").hint("In Explore, next to the views open").key("secondary-t").enabled(needs_recording(s)),
        CommandSpec::new("view.add_heatmap", "Add a heatmap view", IconName::LayoutGrid, add(TimeViewKind::Heatmap)).group("View").hint("In Explore, next to the views open").key("secondary-h").enabled(needs_recording(s)),
        CommandSpec::new("view.add_raster", "Add a raster view", IconName::ChartColumn, add(TimeViewKind::Raster)).group("View").hint("Spikes by unit (needs a sorting's units)").enabled(needs_recording(s)),
        CommandSpec::new("view.channels", "Channels", IconName::PanelLeft, dock(DockPlacement::Left)).group("View").hint("Show or hide the channel list").key("secondary-b").enabled(needs_recording(s)),
        CommandSpec::new("view.inspector", "Inspector", IconName::PanelRight, dock(DockPlacement::Right)).group("View").hint("Show or hide the inspector").key("secondary-alt-b").enabled(needs_recording(s)),
        CommandSpec::new("view.timeline", "Timeline", IconName::PanelBottom, dock(DockPlacement::Bottom)).group("View").hint("Show or hide the timeline").key("secondary-j").enabled(needs_recording(s)),
        CommandSpec::new("view.dark_plots", "Dark plots in the light theme", IconName::Moon, move |_, cx| {
            let on = d1.read(cx).session.dark_plots;
            d1.update(cx, |s, cx| s.set_dark_plots(!on, cx));
        })
        .group("View")
        .hint("Traces, heatmaps and charts on a dark background, whatever the theme")
        .checked(move |cx| d2.read(cx).session.dark_plots),
        CommandSpec::new("view.follow_system", "Follow the system theme", IconName::SunMoon, move |window, cx| {
            let follow = f1.read(cx).session.dark.is_none();
            let choice = if follow { Some(f1.read(cx).is_dark()) } else { None };
            f1.update(cx, |s, cx| s.set_theme_choice(choice, cx));
            crate::app::DspApp::apply_theme(&f1, choice, window, cx);
        })
        .group("View")
        .hint("Dark or light as the system is set, while running")
        .checked(move |cx| f2.read(cx).session.dark.is_none()),
    ]);
    let ep = s.clone();
    list.push(CommandSpec::new("timeline.new_epoch", "New epoch from the visible window", IconName::Clock, move |_, cx| crate::screens::explore::components::timeline_tab::new_epoch(&ep, cx)).group("Timeline").hint("Marks the span shown as an epoch").enabled(needs_recording(s)));
    // Process: every operation, a node on the Pipeline canvas
    for cat in catalog() {
        for it in cat.items {
            let (store, pipeline, template) = (s.clone(), c.pipeline.clone(), it.template.clone());
            list.push(
                CommandSpec::new(op_id(it.label), it.label, cat.icon, move |window, cx| {
                    store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
                    let _ = pipeline.update(cx, |p, cx| p.add(&template, window, cx));
                })
                .group(cat.name)
                .hint(format!("{} · a node on the Pipeline canvas", it.hint)),
            );
        }
    }
    let (r1, rp) = (s.clone(), c.pipeline.clone());
    list.push(
        CommandSpec::with_args::<RecipeArgs>("process.recipe", "Add a recipe", IconName::Workflow, move |a, _, cx| {
            let recipe = crate::domain::recipe::builtin().into_iter().chain(r1.read(cx).doc.recipes.values().cloned()).find(|r| r.name == a.name).ok_or_else(|| format!("no recipe {:?}", a.name))?;
            r1.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
            let _ = rp.update(cx, |p, cx| p.add_recipe(recipe, cx));
            Ok(Value::Null)
        })
        .group("Recipe")
        .hint("A recipe: a chain of operations as one node"),
    );
    let fc = c.clone();
    list.push(
        CommandSpec::new("process.find", "Find an operation…", IconName::Search, move |window, cx| {
            let filter: kitchen_shell::search::Filter = Rc::new(|c: &CommandSpec| c.id.starts_with("process.op."));
            fc.shell(cx, |sh, cx| sh.open_search_with(Some(("Find an operation".into(), filter)), window, cx));
        })
        .group("Process")
        .hint("Search the operations only")
        .key("f3")
        .read_only(),
    );
    // Help, where the app runs
    list.extend([
        CommandSpec::new("help.keys", "Keyboard & mouse", IconName::Keyboard, move |w, cx| help(w, cx)).group("Help").hint("Every key and gesture").key("f1").read_only(),
        CommandSpec::new("help.about", "About DSP App", IconName::Info, move |w, cx| about(w, cx)).group("Help").hint("Version, runtimes").read_only(),
        CommandSpec::new("remote.this", "This computer", IconName::Monitor, |_, _| {}).group("App").hint("dsp-kitchen runs on this computer").checked(|_| true).read_only(),
        CommandSpec::new("remote.connect", "Connect to a dsp-kitchen server…", IconName::Server, |_, _| {})
            .group("App")
            .hint("Run on a server")
            .enabled(|_| Err("Servers you added in Settings open recordings from File › Open from server".into())),
    ]);
    registry.extend(list);
    providers(registry, s);
}

/// The menu entries that follow what the session holds.
fn providers(registry: &Registry, s: &Entity<Store>) {
    let store = s.clone();
    registry.provide("file.recent", move |cx| {
        let recent = store.read(cx).session.recent.clone();
        if recent.is_empty() {
            return vec![];
        }
        let mut out = vec![MenuItem::Separator, MenuItem::label("Recent")];
        out.extend(recent.iter().take(8).map(|p| MenuItem::run(file_label(p), "file.open_path", json!({"path": p}))));
        out.push(MenuItem::command("file.clear_recent"));
        out
    });
    let store = s.clone();
    registry.provide("process.recipes", move |cx| {
        let recipes: Vec<String> = crate::domain::recipe::builtin().into_iter().chain(store.read(cx).doc.recipes.values().cloned()).map(|r| r.name).collect();
        if recipes.is_empty() {
            return vec![];
        }
        vec![MenuItem::Separator, MenuItem::submenu_with_icon("Recipes", IconName::Workflow, recipes.into_iter().map(|n| MenuItem::run(n.clone(), "process.recipe", json!({"name": n})).icon(IconName::Workflow)).collect())]
    });
    registry.provide("process.families", |_| {
        catalog().into_iter().map(|c| MenuItem::submenu_with_icon(c.name, c.icon, c.items.iter().map(|i| MenuItem::command(op_id(i.label))).collect())).collect()
    });
}

/// The app menu (named after the app): recordings; Settings comes last (the frame's).
pub fn app_menu(cx: &App) -> Vec<MenuItem> {
    let mut items: Vec<MenuItem> = crate::ui::recording_picker::open_actions(cx).iter().map(|(_, pick)| MenuItem::command(if *pick == Pick::File { "file.open" } else { "file.open_folder" })).collect();
    items.extend([
        MenuItem::command("file.open_server"),
        MenuItem::command("file.synthetic"),
        MenuItem::command("file.tune"),
        MenuItem::provided("file.recent"),
        MenuItem::Separator,
        MenuItem::command("file.close_recording"),
        MenuItem::Separator,
        MenuItem::command("app.quit"),
    ]);
    items
}

/// The Process menu (Explore and the plugin workspaces): find an operation, a submenu per family,
/// the recipes.
fn process_menu() -> MenuSpec {
    MenuSpec::new("Process", vec![MenuItem::command("process.find"), MenuItem::Separator, MenuItem::label("A node on the Pipeline canvas"), MenuItem::provided("process.families"), MenuItem::provided("process.recipes")])
}

/// Explore's menu row: Edit, View, Timeline, Process.
pub fn explore_menus() -> Vec<MenuSpec> {
    let c = |id: &str| MenuItem::command(id.to_string());
    vec![
        MenuSpec::new("Edit", vec![c("edit.undo"), c("edit.redo")]),
        MenuSpec::new(
            "View",
            vec![
                c("view.add_traces"),
                c("view.add_heatmap"),
                c("view.add_raster"),
                MenuItem::Separator,
                c("view.channels"),
                c("view.inspector"),
                c("view.timeline"),
                MenuItem::Separator,
                c("view.dark_plots"),
                c("view.follow_system"),
                c("shell.theme"),
            ],
        ),
        MenuSpec::new("Timeline", vec![c("timeline.new_epoch")]),
        process_menu(),
    ]
}

/// The menu row of the workspaces plugins provide (Sorting, Curation) and Analytics: Edit, View,
/// Process.
pub fn other_menus() -> Vec<MenuSpec> {
    let c = |id: &str| MenuItem::command(id.to_string());
    vec![MenuSpec::new("Edit", vec![c("edit.undo"), c("edit.redo")]), MenuSpec::new("View", vec![c("view.dark_plots"), c("view.follow_system"), c("shell.theme")]), process_menu()]
}

/// Opens `content` over the frame in a card ("Open from server").
pub fn modal(content: gpui_kit::AnyView, close: Rc<dyn Fn(&mut Window, &mut App)>, cx: &mut App) -> Entity<crate::app::Modal> {
    cx.new(|_| crate::app::Modal { content, close })
}
