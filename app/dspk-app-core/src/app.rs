//! The root view: a title bar with the menus, the workspaces (Explore · Sorting · Pipeline ·
//! Curation) and the open recording; the workspace shown; a status bar. Without a recording the
//! body is a start screen (open a file, a recent one, or a synthetic recording).

use std::path::PathBuf;

use gpui_kit::base::dock::DockPlacement;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::status_bar::StatusBar;
use gpui_kit::assets::IconName;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, Selectable as _, Sizable as _, Theme, ThemeMode, TitleBar};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, ExternalPaths, FocusHandle, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::navigation::actions::*;
use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::data::store::{file_label, AppEvent, Store};
use crate::screens::explore::ExploreView;
use crate::ui::recording_picker::{open_actions, Pick};
use crate::ui::widgets::EmptyState;
use crate::navigation::route::Workspace;

/// The synthetic recording of the start screen and the File menu.
pub const SYNTHETIC: (usize, f64, f64) = (32, 30_000.0, 300.0);

pub struct DspApp {
    store: Entity<Store>,
    pub explore: Entity<ExploreView>,
    pipeline: Entity<crate::screens::pipeline::PipelineView>,
    analytics: Entity<crate::screens::analytics::AnalyticsWorkspace>,
    floating: Entity<crate::screens::floating::FloatingLayer>,
    palette: Option<(Entity<crate::screens::palette::CommandPalette>, Subscription)>,
    show_help: bool,
    show_about: bool,
    /// Settings, while open.
    settings: Option<Entity<crate::screens::settings::SettingsScreen>>,
    /// Open from server, while open.
    browser: Option<(Entity<crate::screens::server_browser::ServerBrowser>, Subscription)>,
    /// Workspaces plugins provide (built the first time they are shown).
    plugin_views: std::collections::HashMap<Workspace, gpui_kit::AnyView>,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl DspApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let choice = store.read(cx).session.dark;
        Self::apply_theme(&store, choice, window, cx);
        store.update(cx, |s, cx| s.suggest_tuning_once(cx));
        let pipeline = cx.new(|cx| crate::screens::pipeline::PipelineView::new(store.clone(), window, cx));
        let explore = cx.new(|cx| ExploreView::new(store.clone(), pipeline.clone(), window, cx));
        let analytics = cx.new(|cx| crate::screens::analytics::AnalyticsWorkspace::new(store.clone(), window, cx));
        let floating = cx.new(|cx| crate::screens::floating::FloatingLayer::new(store.clone(), window, cx));
        let subs = vec![
            // Follow the system's light / dark until the user picks one
            cx.observe_window_appearance(window, |this, window, cx| {
                if this.store.read(cx).session.dark.is_none() {
                    Self::apply_theme(&this.store, None, window, cx);
                }
            }),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::RecordingChanged | AppEvent::WorkspaceChanged | AppEvent::Status) {
                    cx.notify();
                }
            }),
            cx.observe(&store, |_, _, cx| cx.notify()),
        ];
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        Self { store, explore, pipeline, analytics, floating, palette: None, show_help: false, show_about: false, settings: None, browser: None, plugin_views: Default::default(), focus, _subs: subs }
    }

    /// Light, dark, or the system's (`None`); the store learns which, for the plot colours.
    pub(crate) fn apply_theme(store: &Entity<Store>, dark: Option<bool>, window: &mut Window, cx: &mut App) {
        match dark {
            Some(true) => Theme::change(ThemeMode::Dark, Some(window), cx),
            Some(false) => Theme::change(ThemeMode::Light, Some(window), cx),
            None => Theme::sync_system_appearance(Some(window), cx),
        }
        let is_dark = cx.theme().is_dark();
        store.update(cx, |s, cx| s.set_dark(is_dark, cx));
    }

    fn toggle_theme(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let dark = !cx.theme().is_dark();
        self.store.update(cx, |s, cx| s.set_theme_choice(Some(dark), cx));
        Self::apply_theme(&self.store, Some(dark), window, cx);
        cx.notify();
    }

    /// Opens the recording at `path` in Explore. (Sorting folders open in Curation, parked in
    /// `.refactoring/dspk-app/`.)
    pub fn open_any(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.open(path, cx));
    }

    fn prompt_open(&mut self, pick: Pick, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(pick.options("Open recording", false, cx));
        let store = self.store.clone();
        cx.spawn(async move |_, cx| match rx.await {
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

    /// Tunes this device in the background, so first runs and previews don't stop to benchmark.
    fn tune_device(&mut self, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.set_status("Tuning this device in the background (about a minute on a GPU)…", cx));
        let store = self.store.clone();
        cx.spawn(async move |_, cx| {
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

    fn open_path(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.open(path, cx));
    }

    fn open_synthetic(&mut self, cx: &mut Context<Self>) {
        let (ch, rate, dur) = SYNTHETIC;
        self.store.update(cx, |s, cx| s.open_synthetic(ch, rate, dur, cx));
    }

    fn add_view(&mut self, kind: TimeViewKind, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx));
        self.explore.update(cx, |e, cx| e.add_view(kind, cx));
    }

    /// Every command the search offers (`ops_only`: the operations and recipes, as Process ▸
    /// "Find an operation…" lists them).
    fn commands(&self, ops_only: bool, cx: &mut Context<Self>) -> Vec<crate::screens::palette::Command> {
        use crate::screens::palette::Command;
        use std::rc::Rc;
        let me = cx.entity().downgrade();
        let open_actions = open_actions(cx);
        let open = self.store.read(cx).recording.is_some();
        let mut out = Vec::new();
        let mut add = |group: &'static str, icon: IconName, name: String, hint: String, f: Rc<dyn Fn(&mut DspApp, &mut Window, &mut Context<DspApp>)>| {
            let me = me.clone();
            out.push(Command {
                group,
                name,
                hint,
                icon,
                run: Rc::new(move |w, cx| {
                    let f = f.clone();
                    let _ = me.update(cx, |a, cx| f(a, w, cx));
                }),
            });
        };
        if !ops_only {
            for ws in Workspace::ALL {
                add("Go to", ws.icon(), format!("{} workspace", ws.title()), ws.hint().into(), Rc::new(move |a, _, cx| a.store.update(cx, |s, cx| s.set_workspace(ws, cx))));
            }
            for &(label, pick) in open_actions {
                add("File", IconName::FolderOpen, label.into(), "Open a recording from this computer".into(), Rc::new(move |a, _, cx| a.prompt_open(pick, cx)));
            }
            add("File", IconName::FolderOpen, "Open from server…".into(), "Browse the recordings a dspk server offers".into(), Rc::new(|a, _, cx| a.toggle_browser(cx)));
            add("File", IconName::AudioWaveform, "Synthetic recording".into(), "32 channels, 5 minutes: to try things out".into(), Rc::new(|a, _, cx| a.open_synthetic(cx)));
            if open {
                add("File", IconName::X, "Close recording".into(), "Close the recording shown".into(), Rc::new(|a, _, cx| a.store.update(cx, |s, cx| s.close_recording(None, cx))));
            }
            add("File", IconName::Settings, "Settings".into(), "Compute, plugins, appearance".into(), Rc::new(|a, w, cx| {
                if a.settings.is_none() {
                    a.toggle_settings(w, cx);
                }
            }));
            add("File", IconName::Settings, "Tune this device".into(), "Benchmark this GPU once, in the background".into(), Rc::new(|a, _, cx| a.tune_device(cx)));
            add("Edit", IconName::Undo2, "Undo".into(), "Ctrl+Z".into(), Rc::new(|a, _, cx| a.store.update(cx, |s, cx| s.undo(false, cx))));
            add("Edit", IconName::Redo2, "Redo".into(), "Ctrl+Shift+Z".into(), Rc::new(|a, _, cx| a.store.update(cx, |s, cx| s.undo(true, cx))));
            if open {
                for (name, kind, icon) in [("traces", TimeViewKind::Traces, IconName::AudioWaveform), ("heatmap", TimeViewKind::Heatmap, IconName::LayoutGrid), ("raster", TimeViewKind::Raster, IconName::ChartColumn)] {
                    add("View", icon, format!("Add a {name} view"), "In Explore, next to the views open".into(), Rc::new(move |a, _, cx| a.add_view(kind, cx)));
                }
                for (name, placement, hint) in [("Channels", DockPlacement::Left, "Show or hide the channel list"), ("Inspector", DockPlacement::Right, "Show or hide the inspector"), ("Timeline", DockPlacement::Bottom, "Show or hide the timeline")] {
                    add("View", IconName::PanelRight, name.into(), hint.into(), Rc::new(move |a, w, cx| {
                        a.store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx));
                        a.toggle_dock(placement, w, cx);
                    }));
                }
                add("Timeline", IconName::Clock, "New epoch from the visible window".into(), "Marks the span shown as an epoch".into(), Rc::new(|a, _, cx| crate::screens::explore::components::timeline_tab::new_epoch(&a.store, cx)));
            }
            add("View", IconName::Sun, "Light / dark theme".into(), "Switch the theme".into(), Rc::new(|a, w, cx| a.toggle_theme(w, cx)));
            add("Help", IconName::Info, "Keyboard & mouse".into(), "Every key and gesture".into(), Rc::new(|a, _, cx| {
                a.show_help = true;
                cx.notify();
            }));
            add("Help", IconName::Info, "About DSP App".into(), "Version, runtimes".into(), Rc::new(|a, _, cx| {
                a.show_about = true;
                cx.notify();
            }));
        }
        // Operations under their family, as Process ▸ shows them
        for c in crate::screens::pipeline::catalog::catalog() {
            for it in c.items {
                let template = it.template.clone();
                add(c.name, c.icon, it.label.to_string(), format!("{} · a node on the Pipeline canvas", it.hint), Rc::new(move |a, w, cx| {
                    let template = template.clone();
                    a.store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
                    a.pipeline.update(cx, |p, cx| p.add(&template, w, cx));
                }));
            }
        }
        for r in crate::domain::recipe::builtin().into_iter().chain(self.store.read(cx).doc.recipes.values().cloned()) {
            let name = r.name.clone();
            add("Recipe", IconName::Workflow, name, "A recipe: a chain of operations as one node".into(), Rc::new(move |a, _, cx| {
                let r = r.clone();
                a.store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
                a.pipeline.update(cx, |p, cx| p.add_recipe(r, cx));
            }));
        }
        out
    }

    fn toggle_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open_search(false, window, cx);
    }

    /// The search (Ctrl+K: everything; Process ▸ Find an operation: operations only); closes it
    /// when open.
    pub fn open_search(&mut self, ops_only: bool, window: &mut Window, cx: &mut Context<Self>) {
        use crate::screens::palette::{CommandPalette, PaletteEvent};
        if self.palette.take().is_some() {
            window.focus(&self.focus, cx);
            cx.notify();
            return;
        }
        let commands = self.commands(ops_only, cx);
        let (title, placeholder) = if ops_only { ("Find an operation", "Bandpass, notch, CMR, spikes…") } else { ("Search", "An operation, a view or an action…") };
        let palette = cx.new(|cx| CommandPalette::new(commands, title, placeholder, window, cx));
        let sub = cx.subscribe_in(&palette, window, |this, _, e: &PaletteEvent, window, cx| {
            this.palette = None;
            window.focus(&this.focus, cx);
            cx.notify();
            // The chosen command runs once the search is gone, on the window with no view
            // borrowed (it may update the app itself: Settings, a workspace)
            if let PaletteEvent::Run(run) = e {
                let run = run.clone();
                window.defer(cx, move |window, cx| run(window, cx));
            }
        });
        self.palette = Some((palette, sub));
        cx.notify();
    }

    fn toggle_dock(&mut self, placement: DockPlacement, window: &mut Window, cx: &mut Context<Self>) {
        self.explore.update(cx, |e, cx| e.toggle_dock(placement, window, cx));
    }

    // ------------------------------------------------------------------------
    // Title bar
    // ------------------------------------------------------------------------

    fn menus(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let this = cx.entity();
        let recent = self.store.read(cx).session.recent.clone();
        let open = self.store.read(cx).recording.is_some();
        let (dark_plots, follow) = (self.store.read(cx).session.dark_plots, self.store.read(cx).session.dark.is_none());
        let open_actions = open_actions(cx);
        let file = {
            let this = this.clone();
            Button::new("menu-file").ghost().small().label("File").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut menu = menu;
                let item = |label: &str, icon: IconName| PopupMenuItem::new(label.to_string()).icon(Icon::new(icon));
                for &(label, pick) in open_actions {
                    let t = this.clone();
                    menu = menu.item(item(label, IconName::FolderOpen).on_click(move |_, _, cx| t.update(cx, |a, cx| a.prompt_open(pick, cx))));
                }
                let t = this.clone();
                menu = menu.item(item("Open from server…", IconName::FolderOpen).on_click(move |_, _, cx| t.update(cx, |a, cx| a.toggle_browser(cx))));
                let t = this.clone();
                menu = menu.item(item("Synthetic recording (32 ch, 5 min)", IconName::AudioWaveform).on_click(move |_, _, cx| t.update(cx, |a, cx| a.open_synthetic(cx))));
                let t = this.clone();
                menu = menu.item(item("Tune this device", IconName::Settings).on_click(move |_, _, cx| t.update(cx, |a, cx| a.tune_device(cx))));
                if !recent.is_empty() {
                    menu = menu.separator().label("Recent");
                    for p in &recent {
                        let (t, p) = (this.clone(), p.clone());
                        menu = menu.item(PopupMenuItem::new(file_label(&p)).on_click(move |_, _, cx| t.update(cx, |a, cx| a.open_path(p.clone(), cx))));
                    }
                    let t = this.clone();
                    menu = menu.item(PopupMenuItem::new("Clear recent").on_click(move |_, _, cx| {
                        let store = t.read(cx).store.clone();
                        store.update(cx, |s, cx| s.forget_recent(None, cx));
                    }));
                }
                if open {
                    let t = this.clone();
                    menu = menu.separator().item(item("Close recording", IconName::X).on_click(move |_, _, cx| {
                        let store = t.read(cx).store.clone();
                        store.update(cx, |s, cx| s.close_recording(None, cx));
                    }));
                }
                let t = this.clone();
                menu.separator()
                    .item(item("Settings…", IconName::Settings).on_click(move |_, window, cx| t.update(cx, |a, cx| a.toggle_settings(window, cx))))
                    .separator()
                    .item(item("Quit  Ctrl+Q", IconName::X).on_click(|_, _, cx| cx.quit()))
            })
        };
        let view = {
            let this = this.clone();
            Button::new("menu-view").ghost().small().label("View").dropdown_menu(move |menu: PopupMenu, _, _| {
                let item = |label: &'static str, icon: IconName, f: fn(&mut DspApp, &mut Window, &mut Context<DspApp>)| {
                    let t = this.clone();
                    PopupMenuItem::new(label).icon(Icon::new(icon)).disabled(!open).on_click(move |_, window, cx| t.update(cx, |a, cx| f(a, window, cx)))
                };
                menu.item(item("Add traces view", IconName::AudioWaveform, |a, _, cx| a.add_view(TimeViewKind::Traces, cx)))
                    .item(item("Add heatmap view", IconName::LayoutGrid, |a, _, cx| a.add_view(TimeViewKind::Heatmap, cx)))
                    .item(item("Add raster view", IconName::ChartColumn, |a, _, cx| a.add_view(TimeViewKind::Raster, cx)))
                    .separator()
                    .item(item("Channels", IconName::PanelRight, |a, w, cx| a.toggle_dock(DockPlacement::Left, w, cx)))
                    .item(item("Inspector", IconName::PanelRight, |a, w, cx| a.toggle_dock(DockPlacement::Right, w, cx)))
                    .item(item("Timeline", IconName::PanelRight, |a, w, cx| a.toggle_dock(DockPlacement::Bottom, w, cx)))
                    .separator()
                    .item({
                        let t = this.clone();
                        PopupMenuItem::new("Dark plots in the light theme").checked(dark_plots).on_click(move |_, _, cx| {
                            t.update(cx, |a, cx| a.store.update(cx, |s, cx| s.set_dark_plots(!dark_plots, cx)))
                        })
                    })
                    .item({
                        let t = this.clone();
                        PopupMenuItem::new("Follow the system theme").checked(follow).on_click(move |_, window, cx| {
                            t.update(cx, |a, cx| {
                                let choice = if follow { Some(cx.theme().is_dark()) } else { None };
                                a.store.update(cx, |s, cx| s.set_theme_choice(choice, cx));
                                Self::apply_theme(&a.store, choice, window, cx);
                            })
                        })
                    })
            })
        };
        let help = {
            let this = this.clone();
            Button::new("menu-help").ghost().small().label("Help").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (t1, t2) = (this.clone(), this.clone());
                menu.item(PopupMenuItem::new("Search…  Ctrl+K").icon(Icon::new(IconName::Search)).on_click({
                    let t = this.clone();
                    move |_, window, cx| t.update(cx, |a, cx| a.open_search(false, window, cx))
                }))
                .item(PopupMenuItem::new("Keyboard & mouse").icon(Icon::new(IconName::Info)).on_click(move |_, _, cx| {
                    t1.update(cx, |a, cx| {
                        a.show_help = true;
                        cx.notify();
                    })
                }))
                .separator()
                .item(PopupMenuItem::new("About DSP App").icon(Icon::new(IconName::Info)).on_click(move |_, _, cx| {
                    t2.update(cx, |a, cx| {
                        a.show_about = true;
                        cx.notify();
                    })
                }))
            })
        };
        let edit = {
            let this = this.clone();
            Button::new("menu-edit").ghost().small().label("Edit").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b) = (this.clone(), this.clone());
                menu.item(PopupMenuItem::new("Undo  Ctrl+Z").icon(Icon::new(IconName::Undo2)).on_click(move |_, _, cx| a.update(cx, |a, cx| a.store.update(cx, |s, cx| s.undo(false, cx)))))
                    .item(PopupMenuItem::new("Redo  Ctrl+Shift+Z").icon(Icon::new(IconName::Redo2)).on_click(move |_, _, cx| b.update(cx, |a, cx| a.store.update(cx, |s, cx| s.undo(true, cx)))))
            })
        };
        // Process: a node on the Pipeline canvas, one submenu per family (as the operations
        // tree), then the recipes
        let process = {
            let this = this.clone();
            let recipes: Vec<crate::domain::recipe::RecipeDoc> = crate::domain::recipe::builtin().into_iter().chain(self.store.read(cx).doc.recipes.values().cloned()).collect();
            Button::new("menu-process").ghost().small().label("Process").dropdown_menu(move |menu: PopupMenu, window, cx| {
                let find = this.clone();
                let mut m = menu
                    .item(PopupMenuItem::new("Find an operation…  Ctrl+K").icon(Icon::new(IconName::Search)).on_click(move |_, window, cx| find.update(cx, |a, cx| a.open_search(true, window, cx))))
                    .separator()
                    .label("A node on the Pipeline canvas");
                for c in crate::screens::pipeline::catalog::catalog() {
                    let (this, icon, items) = (this.clone(), c.icon, c.items);
                    m = m.submenu_with_icon(Some(Icon::new(icon)), c.name, window, cx, move |mut sub, _, _| {
                        for it in &items {
                            let (t, template) = (this.clone(), it.template.clone());
                            sub = sub.item(PopupMenuItem::new(format!("{}   {}", it.label, it.hint)).icon(Icon::new(icon)).on_click(move |_, window, cx| {
                                t.update(cx, |a, cx| {
                                    a.store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
                                    a.pipeline.update(cx, |p, cx| p.add(&template, window, cx));
                                })
                            }));
                        }
                        sub
                    });
                }
                if !recipes.is_empty() {
                    let (this, recipes) = (this.clone(), recipes.clone());
                    m = m.separator().submenu_with_icon(Some(Icon::new(IconName::Workflow)), "Recipes", window, cx, move |mut sub, _, _| {
                        for r in &recipes {
                            let (t, r) = (this.clone(), r.clone());
                            sub = sub.item(PopupMenuItem::new(r.name.clone()).icon(Icon::new(IconName::Workflow)).on_click(move |_, _, cx| {
                                let r = r.clone();
                                t.update(cx, |a, cx| {
                                    a.store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx));
                                    a.pipeline.update(cx, |p, cx| p.add_recipe(r, cx));
                                })
                            }));
                        }
                        sub
                    });
                }
                m
            })
        };
        h_flex().gap_0p5().child(file).child(edit).child(view).child(process).child(help)
    }

    fn workspaces(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let current = self.store.read(cx).workspace;
        h_flex().gap_0p5().children(Workspace::ALL.into_iter().map(|w| {
            let store = self.store.clone();
            Button::new(SharedString::from(format!("workspace-{}", w.title().to_lowercase())))
                .small()
                .ghost()
                .icon(w.icon())
                .label(w.title())
                .tooltip(w.hint())
                .selected(w == current)
                .on_click(move |_, _, cx| store.update(cx, |s, cx| s.set_workspace(w, cx)))
        }))
    }

    fn title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let t = cx.theme();
        // The file, then the source the active view reads (what is drawn and edited)
        let recording = s.recording.as_ref().map(|r| {
            let source = s.active_view().map(|v| s.view_source(v)).unwrap_or_else(|| r.sources.default_entry().id.clone());
            let several = r.sources.entries().len() > 1;
            h_flex()
                .gap_2()
                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(t.foreground).child(r.name.clone()))
                .when(several, |d| d.child(div().text_sm().text_color(t.foreground).child(r.sources.entry(&source).name.clone())))
                .child(div().text_xs().text_color(t.muted_foreground).child(r.summary_of(&source)))
        });
        let opening = s.opening.clone().map(|n| div().text_xs().text_color(t.muted_foreground).child(format!("Opening {n}…")));
        TitleBar::new().child(
            h_flex()
                .w_full()
                .gap_3()
                .pr_2()
                .child(self.menus(cx))
                .child(div().h(px(18.)).w(px(1.)).bg(cx.theme().border))
                .child(self.workspaces(cx))
                .child(div().flex_1())
                .children(opening)
                .children(recording)
                .child({
                    let dark = cx.theme().is_dark();
                    Button::new("theme")
                        .ghost()
                        .small()
                        .icon(if dark { IconName::Sun } else { IconName::Moon })
                        .tooltip(if dark { "Light theme" } else { "Dark theme" })
                        .on_click(cx.listener(|this, _, window, cx| this.toggle_theme(window, cx)))
                }),
        )
    }

    // ------------------------------------------------------------------------
    // Body
    // ------------------------------------------------------------------------

    fn start_screen(&self, cx: &mut Context<Self>) -> AnyElement {
        let recent = self.store.read(cx).session.recent.clone();
        let muted = cx.theme().muted_foreground;
        let mut screen = EmptyState::new(
            gpui_kit::assets::IconName::AudioWaveform,
            "Open a recording",
            ".bin · .cbin · .zarr · .nwb.zarr",
        )
        .child(
            h_flex()
                .gap_2()
                .children(open_actions(cx).iter().enumerate().map(|(i, &(label, pick))| {
                    let button = Button::new(("start-open", i)).icon(if pick == Pick::File { IconName::File } else { IconName::FolderOpen }).label(label);
                    let button = if i == 0 { button.primary() } else { button.outline() };
                    button.on_click(cx.listener(move |this, _, _, cx| this.prompt_open(pick, cx)))
                }))
                .child(Button::new("start-synthetic").ghost().label("Synthetic recording").on_click(cx.listener(|this, _, _, cx| this.open_synthetic(cx)))),
        );
        if !recent.is_empty() {
            screen = screen.child(
                v_flex()
                    .mt_4()
                    .w(px(420.))
                    .gap_1()
                    .child(div().text_xs().text_color(muted).child("Recent"))
                    .children(recent.into_iter().take(6).enumerate().map(|(i, p)| {
                        let label = file_label(&p);
                        let dir = p.parent().map(|d| d.display().to_string()).unwrap_or_default();
                        let forget = p.clone();
                        h_flex()
                            .w_full()
                            .gap_1()
                            .child(
                                Button::new(("recent", i))
                                    .ghost()
                                    .flex_1()
                                    .child(h_flex().w_full().gap_2().child(div().text_sm().child(label)).child(div().flex_1().text_xs().text_color(muted).truncate().child(dir)))
                                    .on_click(cx.listener(move |this, _, _, cx| this.open_path(p.clone(), cx))),
                            )
                            .child(
                                crate::ui::widgets::icon_button(("recent-forget", i), IconName::Close, "Remove from the list (the file stays)")
                                    .xsmall()
                                    .on_click(cx.listener(move |this, _, _, cx| this.store.update(cx, |s, cx| s.forget_recent(Some(&forget), cx)))),
                            )
                    })),
            );
        }
        div().id("start-screen").test_support().size_full().child(screen).into_any_element()
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let workspace = self.store.read(cx).workspace;
        // A workspace a plugin provides (e.g. Sorting), built once
        if workspace != Workspace::Explore {
            if !self.plugin_views.contains_key(&workspace) {
                let store = self.store.clone();
                let built = crate::plugin::plugins().iter().filter(|p| p.provides(workspace)).find_map(|p| p.workspace(workspace, &store, window, cx));
                if let Some(view) = built {
                    self.plugin_views.insert(workspace, view);
                }
            }
            if let Some(view) = self.plugin_views.get(&workspace) {
                return view.clone().into_any_element();
            }
        }
        if workspace == Workspace::Pipeline {
            return self.pipeline.clone().into_any_element();
        }
        if workspace == Workspace::Analytics {
            return self.analytics.clone().into_any_element();
        }
        let s = self.store.read(cx);
        if let Some(text) = workspace.planned() {
            return div()
                .id("planned-workspace")
                .test_support()
                .size_full()
                .child(EmptyState::new(workspace.icon(), format!("{} is on its way", workspace.title()), text))
                .into_any_element();
        }
        if s.recording.is_none() {
            return self.start_screen(cx);
        }
        self.explore.clone().into_any_element()
    }

    fn status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let muted = cx.theme().muted_foreground;
        StatusBar::new().left(div().text_xs().child(s.status.clone())).right(div().text_xs().text_color(muted).child(s.render_info.clone()))
    }

    fn help(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, bg, border) = (t.muted_foreground, t.popover, t.border);
        div()
            .id("help-overlay")
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.5))
            .on_click(cx.listener(|this, _, _, cx| {
                this.show_help = false;
                cx.notify();
            }))
            .child(
                v_flex()
                    .id("help")
                    .w(px(560.))
                    .p_4()
                    .gap_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Keyboard & mouse"))
                    .children(HELP.iter().map(|(group, rows)| {
                        v_flex().gap_1().child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(*group)).children(
                            rows.iter().map(|(keys, what)| h_flex().gap_3().child(div().w(px(230.)).text_sm().font_weight(FontWeight::MEDIUM).child(*keys)).child(div().text_sm().child(*what))),
                        )
                    }))
                    .child(h_flex().justify_end().child(Button::new("help-close").outline().small().label("Close").on_click(cx.listener(|this, _, _, cx| {
                        this.show_help = false;
                        cx.notify();
                    })))),
            )
    }

    /// Opens Settings, or closes it when open.
    fn toggle_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.settings = match self.settings.take() {
            Some(_) => None,
            None => Some(cx.new(|cx| crate::screens::settings::SettingsScreen::new(self.store.clone(), window, cx))),
        };
        cx.notify();
    }

    /// Opens Open from server, or closes it when open.
    fn toggle_browser(&mut self, cx: &mut Context<Self>) {
        if self.browser.take().is_none() {
            let browser = cx.new(|cx| crate::screens::server_browser::ServerBrowser::new(self.store.clone(), cx));
            let sub = cx.subscribe(&browser, |this, _, e: &crate::screens::server_browser::BrowserEvent, cx| match e {
                crate::screens::server_browser::BrowserEvent::Opened => {
                    this.browser = None;
                    this.store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx));
                    cx.notify();
                }
            });
            self.browser = Some((browser, sub));
        }
        cx.notify();
    }

    /// `content` over everything, in a card; a click outside it runs `close`.
    fn modal(&self, id: &'static str, content: AnyElement, close: fn(&mut Self), cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (bg, border) = (t.popover, t.border);
        div()
            .id(SharedString::from(format!("{id}-overlay")))
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.5))
            .on_click(cx.listener(move |this, _, _, cx| {
                close(this);
                cx.notify();
            }))
            .child(
                div()
                    .id(SharedString::from(format!("{id}-card")))
                    .p_4()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(content),
            )
    }

    fn about(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, bg, border) = (t.muted_foreground, t.popover, t.border);
        div()
            .id("about-overlay")
            .test_support()
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.5))
            .on_click(cx.listener(|this, _, _, cx| {
                this.show_about = false;
                cx.notify();
            }))
            .child(
                v_flex()
                    .id("about")
                    .w(px(440.))
                    .p_4()
                    .gap_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            .child(gpui_kit::img(crate::ui::assets::BRAND_LOGO).size(px(56.)).flex_none())
                            .child(
                                v_flex()
                                    .gap_1()
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("DSP App"))
                                            .child(div().text_xs().text_color(muted).child(format!("v{}", env!("CARGO_PKG_VERSION")))),
                                    )
                                    .child(div().text_xs().text_color(muted).child("by neuro-kitchen")),
                            ),
                    )
                    .child(div().text_sm().text_color(muted).child(
                        "Electrophysiology workbench built with GPUI and gpui-kit. Explore multi-channel recordings across docked time views and curate spike-sorting results.",
                    ))
                    .child(
                        h_flex()
                            .justify_end()
                            .child(Button::new("about-close").outline().small().label("Close").on_click(cx.listener(|this, _, _, cx| {
                                this.show_about = false;
                                cx.notify();
                            }))),
                    ),
            )
    }
}

impl Render for DspApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("dspk-app")
            .track_focus(&self.focus)
            .size_full()
            .relative()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .on_action(cx.listener(|this, _: &OpenRecording, _, cx| this.prompt_open(Pick::File, cx)))
            .on_action(cx.listener(|this, _: &OpenRecordingFolder, _, cx| this.prompt_open(Pick::Folder, cx)))
            .on_action(cx.listener(|this, _: &OpenSynthetic, _, cx| this.open_synthetic(cx)))
            .on_action(cx.listener(|this, _: &AddTraces, _, cx| this.add_view(TimeViewKind::Traces, cx)))
            .on_action(cx.listener(|this, _: &AddHeatmap, _, cx| this.add_view(TimeViewKind::Heatmap, cx)))
            .on_action(cx.listener(|this, _: &ToggleChannels, w, cx| this.toggle_dock(DockPlacement::Left, w, cx)))
            .on_action(cx.listener(|this, _: &ToggleSettings, w, cx| this.toggle_dock(DockPlacement::Right, w, cx)))
            .on_action(cx.listener(|this, _: &ToggleTimeline, w, cx| this.toggle_dock(DockPlacement::Bottom, w, cx)))
            .on_action(cx.listener(|this, _: &ShowHelp, _, cx| {
                this.show_help = !this.show_help;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ShowPreferences, window, cx| this.toggle_settings(window, cx)))
            .on_action(cx.listener(|this, _: &ShowAbout, _, cx| {
                this.show_about = !this.show_about;
                cx.notify();
            }))
            .on_action(|_: &Quit, _, cx: &mut App| cx.quit())
            .on_action(cx.listener(|this, _: &Undo, _, cx| this.store.update(cx, |s, cx| s.undo(false, cx))))
            .on_action(cx.listener(|this, _: &Redo, _, cx| this.store.update(cx, |s, cx| s.undo(true, cx))))
            .on_action(cx.listener(|this, _: &CommandPalette, w, cx| this.toggle_palette(w, cx)))
            // A file or folder dropped anywhere opens it
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first().cloned() {
                    this.open_any(p, cx);
                }
            }))
            .child(self.title_bar(cx))
            .child(div().flex_1().min_h_0().child(self.body(window, cx)))
            .child(self.status_bar(cx))
            .child(self.floating.clone())
            .children(self.palette.as_ref().map(|(p, _)| p.clone()))
            .when(self.show_help, |d| d.child(self.help(cx)))
            .when(self.show_about, |d| d.child(self.about(cx)))
            .children(self.settings.clone().map(|s| self.modal("settings", s.into_any_element(), |a| a.settings = None, cx)))
            .children(self.browser.as_ref().map(|(b, _)| b.clone()).map(|b| self.modal("server-browser", b.into_any_element(), |a| a.browser = None, cx)))
    }
}

/// Headless UI tests: real windows, layout and hit testing, no pixels; rendering runs on the real
/// render threads.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use gpui_kit::base::dock::DockPlacement;
    use gpui_kit::test::TestWindowExt as _;
    use gpui_kit::{px, size, AnyWindowHandle, AppContext as _, Bounds, Entity, Point, TestAppContext, WindowBounds, WindowOptions};

    use super::DspApp;
    use crate::data::recording::{Dataset, SourceSet};
    use crate::screens::explore::trace::renderer::TimeViewKind;
    use crate::data::preferences::Session;
    use crate::data::store::{Recording, Store};
    use crate::screens::explore::{ExploreVm, TraceVm};
    use crate::navigation::route::Workspace;

    fn open(cx: &mut TestAppContext) -> (AnyWindowHandle, Entity<Store>, Entity<DspApp>) {
        // Frames come back from real render threads: allow their wake-ups from the start
        cx.executor().allow_parking();
        cx.update(|cx| {
            gpui_kit::init(cx);
            crate::navigation::actions::bind_keys(cx);
            crate::data::services::Services::install(cx);
            let store = cx.new(|_| Store::new(Session::new(), None));
            let bounds = Bounds { origin: Point::default(), size: size(px(1400.), px(900.)) };
            let options = WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
            let s = store.clone();
            let (window, app) = gpui_kit::open_window(options, cx, move |window, cx| cx.new(|cx| DspApp::new(s, window, cx))).expect("open test window");
            (window, store, app)
        })
    }

    /// Waits (real time) for work on the render threads to come back to the UI.
    fn wait_until(cx: &mut TestAppContext, window: AnyWindowHandle, mut done: impl FnMut(&mut TestAppContext) -> bool) {
        cx.executor().allow_parking();
        for _ in 0..500 {
            cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
            cx.run_until_parked();
            if done(cx) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out");
    }

    fn explore(cx: &mut TestAppContext, app: &Entity<DspApp>) -> Entity<ExploreVm> {
        cx.update(|cx| app.read(cx).explore.read(cx).vm.clone())
    }

    fn views(cx: &mut TestAppContext, app: &Entity<DspApp>) -> Vec<Entity<TraceVm>> {
        let ex = explore(cx, app);
        cx.update(|cx| ex.read(cx).views.clone())
    }

    /// Installs a synthetic 16-channel, 10 s recording.
    fn install(cx: &mut TestAppContext, store: &Entity<Store>) {
        cx.update(|cx| {
            let ds = Dataset::generate_synthetic(16, 10_000.0, 10.0);
            store.update(cx, |s, cx| s.install(Recording::new(SourceSet::single(ds), None), cx));
        });
        cx.run_until_parked();
    }

    /// Every view shows a frame drawn for the current window.
    fn all_drawn_for_window(cx: &mut TestAppContext, store: &Entity<Store>, app: &Entity<DspApp>) -> bool {
        let vs = views(cx, app);
        cx.update(|cx| {
            let start = store.read(cx).timeline.window_start_sec;
            vs.iter().all(|v| v.read(cx).shown.as_ref().is_some_and(|s| (s.start - start).abs() < 1e-9))
        })
    }

    /// Recordings load once each; closing the active one switches to another, then to none, and
    /// Explore is left with no views (its empty state), every view closable.
    #[gpui_kit::test]
    fn recordings_load_once_and_close(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        let rec = |name: &str| Recording::new(SourceSet::single(Dataset::generate_synthetic(4, 1_000.0, 2.0)), Some(std::path::PathBuf::from(format!("/nonexistent/{name}.zarr"))));
        cx.update(|cx| store.update(cx, |s, cx| s.install(rec("a"), cx)));
        cx.update(|cx| store.update(cx, |s, cx| s.install(rec("b"), cx)));
        cx.update(|cx| store.update(cx, |s, cx| s.install(rec("a"), cx)));
        cx.run_until_parked();
        cx.update(|cx| {
            let names: Vec<String> = store.read(cx).recordings().map(|r| r.name.clone()).collect();
            assert_eq!(names, ["a.zarr", "b.zarr"], "each file once, the active first");
        });
        cx.update(|cx| store.update(cx, |s, cx| s.close_recording(None, cx)));
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).recording.as_ref().map(|r| r.name.clone()).as_deref(), Some("b.zarr")));
        cx.update(|cx| store.update(cx, |s, cx| s.close_recording(None, cx)));
        cx.run_until_parked();
        cx.update(|cx| assert!(store.read(cx).recording.is_none() && store.read(cx).loaded.is_empty()));
        assert!(views(cx, &app).is_empty(), "no recording, no views");
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
    }

    /// Every view closes, the last one too (Explore then offers "Add view…"), and adding one
    /// after that lays out the dock again.
    #[gpui_kit::test]
    fn every_view_closes(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
        let ex = explore(cx, &app);
        for vm in views(cx, &app) {
            let id = cx.update(|cx| vm.read(cx).id());
            cx.update(|cx| ex.update(cx, |ex, cx| ex.close(id, cx)));
            cx.run_until_parked();
        }
        assert!(views(cx, &app).is_empty(), "the last view closes too");
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("explore-empty-add-view").is_some(), "the empty state offers Add view");
        })
        .unwrap();
        let source = cx.update(|cx| store.read(cx).sources().unwrap().default_entry().id.clone());
        cx.update(|cx| ex.update(cx, |ex, cx| {
            ex.add_on(TimeViewKind::Traces, &source, cx);
        }));
        cx.run_until_parked();
        assert_eq!(views(cx, &app).len(), 1);
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
    }

    /// Choosing a view's tab selects that view: the Inspector follows the selected view.
    #[gpui_kit::test]
    fn showing_a_tab_selects_its_view(cx: &mut TestAppContext) {
        use gpui_kit::base::dock::Panel as _;
        let (window, store, app) = open(cx);
        install(cx, &store);
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
        let ex = explore(cx, &app);
        let source = cx.update(|cx| store.read(cx).sources().unwrap().default_entry().id.clone());
        cx.update(|cx| ex.update(cx, |ex, cx| {
            ex.add_on(TimeViewKind::Traces, &source, cx);
        }));
        cx.run_until_parked();
        let first = views(cx, &app)[0].clone();
        cx.update(|cx| assert_ne!(ex.read(cx).focused.as_ref(), Some(&first), "the view just added is the focused one"));
        let id = cx.update(|cx| first.read(cx).id());
        cx.update_window(window, |_, window, cx| {
            let panel = app.read(cx).explore.read(cx).panels[&id].clone();
            panel.update(cx, |p, cx| {
                p.set_active(false, window, cx);
                p.set_active(true, window, cx);
            });
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(ex.read(cx).focused.as_ref(), Some(&first)));
    }

    /// "Add view or container": a view kind adds a view (Raster only with units), a container
    /// arranges the views, Close and Esc close it, the arrow keys and Enter run a row.
    #[gpui_kit::test]
    fn add_view_or_container(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        let before = views(cx, &app).len();
        let explore = cx.update(|cx| app.read(cx).explore.clone());
        let dialog_open = |cx: &mut TestAppContext| cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.try_find("add-view-dialog").is_some()
        }).unwrap();
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("explore-add-view", cx);
        }).unwrap();
        cx.run_until_parked();
        assert!(dialog_open(cx), "Add view… opens the dialog");
        cx.update_window(window, |_, window, cx| window.click("add-view-kind-raster", cx)).unwrap();
        cx.run_until_parked();
        assert_eq!(views(cx, &app).len(), before, "no units: Raster adds nothing");
        cx.update_window(window, |_, window, cx| window.click("add-view-kind-heatmap", cx)).unwrap();
        cx.run_until_parked();
        assert_eq!(views(cx, &app).len(), before + 1, "a heatmap view");
        assert!(!dialog_open(cx), "adding closes the dialog");
        cx.update(|cx| explore.update(cx, |e, cx| e.set_add_dialog(true, cx)));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("add-container-grid", cx);
        }).unwrap();
        cx.run_until_parked();
        assert!(!dialog_open(cx));
        assert_eq!(views(cx, &app).len(), before + 1, "arranging keeps every view");
        cx.update(|cx| explore.update(cx, |e, cx| e.set_add_dialog(true, cx)));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("add-view-close", cx);
        }).unwrap();
        cx.run_until_parked();
        assert!(!dialog_open(cx), "Close closes it");
        // The keyboard: Esc closes; from the first row, ↑ wraps to the last it can run (Heatmap:
        // Raster has no units) and Enter adds it
        let open_with_click = |cx: &mut TestAppContext| {
            cx.update_window(window, |_, window, cx| {
                window.render_frame(cx);
                window.click("explore-add-view", cx);
            }).unwrap();
            cx.run_until_parked();
        };
        open_with_click(cx);
        cx.simulate_keystrokes(window, "escape");
        assert!(!dialog_open(cx), "Esc closes it");
        open_with_click(cx);
        let n = views(cx, &app).len();
        cx.simulate_keystrokes(window, "up enter");
        cx.run_until_parked();
        assert!(!dialog_open(cx));
        assert_eq!(views(cx, &app).len(), n + 1, "↑ then Enter added a heatmap");
    }

    /// A remap at the head of a view's chain is what Explore shows: its order, without the
    /// channels it drops.
    #[gpui_kit::test]
    fn explore_follows_a_leading_remap(cx: &mut TestAppContext) {
        let (window, store, _app) = open(cx);
        cx.update(|cx| {
            let ds = Dataset::generate_synthetic(8, 10_000.0, 2.0);
            store.update(cx, |s, cx| s.install(Recording::new(SourceSet::single(ds), Some("/nonexistent/remap.zarr".into())), cx));
        });
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
        cx.run_until_parked();
        let source = cx.update(|cx| store.read(cx).sources().unwrap().default_entry().id.clone());
        let view = cx.update(|cx| store.update(cx, |s, cx| s.view_for_source(&source, cx))).expect("a view of the source");
        cx.update(|cx| store.update(cx, |s, cx| s.execute(crate::domain::command::SessionCommand::SelectView { view: view.clone() }, cx)).unwrap());
        cx.run_until_parked();
        let node = crate::screens::pipeline::channel_table::Rows { base: 0, inputs: vec![3, 1, 2], names: vec![None; 3] }.node();
        cx.update(|cx| store.update(cx, |s, cx| s.execute(crate::domain::command::SessionCommand::AddOp { view, node, index: Some(0), id: None }, cx)).unwrap());
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).selection(&source), &[3, 1, 2]));
    }

    /// Remaps anywhere in a chain preview as the pipeline runs them: between processing steps a
    /// reorder (each channel still its own data), after the last one the channels shown (may
    /// drop); between steps, dropping is refused.
    #[gpui_kit::test]
    fn explore_follows_remaps_anywhere(cx: &mut TestAppContext) {
        use crate::domain::command::SessionCommand;
        use crate::screens::pipeline::channel_table::Rows;
        let (window, store, _app) = open(cx);
        cx.update(|cx| {
            let ds = Dataset::generate_synthetic(8, 10_000.0, 2.0);
            store.update(cx, |s, cx| s.install(Recording::new(SourceSet::single(ds), Some("/nonexistent/remaps.zarr".into())), cx));
        });
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
        cx.run_until_parked();
        let source = cx.update(|cx| store.read(cx).sources().unwrap().default_entry().id.clone());
        let view = cx.update(|cx| store.update(cx, |s, cx| s.view_for_source(&source, cx))).expect("a view");
        let run = |cx: &mut TestAppContext, cmd: SessionCommand| cx.update(|cx| store.update(cx, |s, cx| s.execute(cmd, cx)));
        run(cx, SessionCommand::SelectView { view: view.clone() }).unwrap();
        let scale = |a: f32| serde_json::from_value::<dspk_pipeline::Node>(serde_json::json!({"op": "scale", "alpha": a, "beta": 0.0})).unwrap();
        let reverse = Rows { base: 0, inputs: (0..8).rev().collect(), names: vec![None; 8] }.node();
        for node in [scale(2.0), reverse, scale(1.0)] {
            run(cx, SessionCommand::AddOp { view: view.clone(), node, index: None, id: None }).unwrap();
        }
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).selection(&source), &[7, 6, 5, 4, 3, 2, 1, 0], "the reorder is shown"));
        let (preview, raw) = cx.update(|cx| (store.read(cx).preview_of(&view).expect("a preview"), store.read(cx).sources().unwrap().get(&source).signal().clone()));
        for c in [0usize, 3, 7] {
            let (mut p, mut r) = (vec![0.0f32; 500], vec![0.0f32; 500]);
            preview.read(&[c], 5_000..5_500, &mut p).unwrap();
            raw.read(&[c], 5_000..5_500, &mut r).unwrap();
            assert!(p.iter().zip(&r).all(|(p, r)| (p - 2.0 * r).abs() < 1e-3 * r.abs().max(1.0)), "channel {c} carries its own data");
        }
        // After the last processing step: keep the first two shown
        let keep = Rows { base: 0, inputs: vec![0, 1], names: vec![None; 2] }.node();
        run(cx, SessionCommand::AddOp { view: view.clone(), node: keep, index: None, id: None }).unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).selection(&source), &[7, 6]));
        // Between processing steps, dropping is refused (no preview, the reason in the status)
        let drop = Rows { base: 0, inputs: vec![0, 1], names: vec![None; 2] }.node();
        run(cx, SessionCommand::AddOp { view: view.clone(), node: drop, index: Some(1), id: None }).unwrap();
        cx.run_until_parked();
        cx.update(|cx| {
            let s = store.read(cx);
            assert!(s.preview_of(&view).is_none());
            assert!(s.status.contains("keeps every channel"), "{}", s.status);
        });
    }

    /// A whitening op previews fitted on the view's signal (channels of about unit variance), the
    /// fit in the background; an edit after it reuses the fit.
    #[gpui_kit::test]
    fn whitening_previews_fitted(cx: &mut TestAppContext) {
        use crate::domain::command::SessionCommand;
        let (window, store, _app) = open(cx);
        cx.update(|cx| {
            let ds = Dataset::generate_synthetic(6, 10_000.0, 4.0);
            store.update(cx, |s, cx| s.install(Recording::new(SourceSet::single(ds), Some("/nonexistent/whiten.zarr".into())), cx));
        });
        cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
        cx.run_until_parked();
        let source = cx.update(|cx| store.read(cx).sources().unwrap().default_entry().id.clone());
        let view = cx.update(|cx| store.update(cx, |s, cx| s.view_for_source(&source, cx))).expect("a view");
        let add = |cx: &mut TestAppContext, op: serde_json::Value| {
            let node: dspk_pipeline::Node = serde_json::from_value(op).unwrap();
            cx.update(|cx| store.update(cx, |s, cx| s.execute(SessionCommand::AddOp { view: view.clone(), node, index: None, id: None }, cx))).unwrap();
            cx.run_until_parked();
        };
        let read = |cx: &mut TestAppContext| {
            // The fit runs in the background: the preview comes when it is done
            wait_long(cx, window, |cx| cx.update(|cx| store.read(cx).preview_of(&view).is_some()));
            let preview = cx.update(|cx| store.read(cx).preview_of(&view)).expect("a preview");
            let mut x = vec![0.0f32; 6 * 20_000];
            preview.read(&[0, 1, 2, 3, 4, 5], 10_000..30_000, &mut x).unwrap();
            x
        };
        add(cx, serde_json::json!({"op": "iir", "band": "highpass", "cutoff": [100.0]}));
        add(cx, serde_json::json!({"op": "whiten"}));
        let x = read(cx);
        for c in 0..6 {
            let row = &x[c * 20_000..(c + 1) * 20_000];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / row.len() as f64;
            assert!((var - 1.0).abs() < 0.3, "channel {c}: variance {var:.3}");
        }
        add(cx, serde_json::json!({"op": "scale", "alpha": 2.0, "beta": 0.0}));
        let y = read(cx);
        assert!(x.iter().zip(&y).all(|(a, b)| (2.0 * a - b).abs() < 1e-4 * a.abs().max(1.0)), "the same fit after an edit downstream");
    }

    #[gpui_kit::test]
    fn start_screen_then_workspaces(cx: &mut TestAppContext) {
        let (window, store, _) = open(cx);
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("start-screen").is_some(), "no recording: the start screen");
            window.click("workspace-curation", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).workspace, Workspace::Curation));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("planned-workspace").is_some(), "a workspace still to come says what it will hold");
            assert!(window.try_find("start-screen").is_none());
            window.click("workspace-explore", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).workspace, Workspace::Explore));
    }

    /// A job's graph nodes: running while it runs, then done (or failed); a job refused because
    /// another runs marks nothing.
    #[gpui_kit::test]
    fn jobs_mark_their_nodes(cx: &mut TestAppContext) {
        use crate::data::jobs::{Cancel, JobMessage};
        use crate::data::store::NodeRun;
        let (_window, store, _app) = open(cx);
        cx.update(|cx| {
            store.update(cx, |s, cx| {
                let (tx, rx) = std::sync::mpsc::channel::<Box<dyn Fn(JobMessage) + Send + Sync>>();
                assert!(s.start_job_on(["read".to_string(), "iir".to_string()], "test", move |send| {
                    let _ = tx.send(send);
                    Cancel::default()
                }, |_, _, _| {}, cx));
                assert!(matches!(s.node_status.get("iir"), Some(NodeRun::Running { .. })));
                // Another job while this one runs: refused, nothing marked
                assert!(!s.start_job_on(["other".to_string()], "second", |_| Cancel::default(), |_, _, _| {}, cx));
                let send = rx.recv().unwrap();
                send(JobMessage::Progress("Writing 40 %".into(), Some(0.4)));
                send(JobMessage::Done(std::path::PathBuf::from("/tmp/out.zarr")));
            })
        });
        cx.run_until_parked();
        cx.update(|cx| {
            let s = store.read(cx);
            assert!(matches!(s.node_status.get("read"), Some(NodeRun::Done { .. })), "{:?}", s.node_status);
            assert!(matches!(s.node_status.get("iir"), Some(NodeRun::Done { .. })));
            assert!(!s.node_status.contains_key("other"));
        });
        // A failure marks the job's nodes failed
        cx.update(|cx| {
            store.update(cx, |s, cx| {
                assert!(s.start_job_on(["iir".to_string()], "failing", |send| {
                    send(JobMessage::Failed("disk full".into()));
                    Cancel::default()
                }, |_, _, _| {}, cx));
            })
        });
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).node_status.get("iir"), Some(&NodeRun::Failed("disk full".into()))));
    }

    #[gpui_kit::test]
    fn pipeline_canvas_shows_and_edits_the_graph(cx: &mut TestAppContext) {
        use crate::domain::command::SessionCommand;
        use crate::domain::session::RecordingRef;
        let (window, store, app) = open(cx);
        cx.update(|cx| {
            store.update(cx, |s, cx| {
                let recording = RecordingRef { path: "/data/rec.bin".into(), source: None, channels: None };
                s.execute(SessionCommand::OpenView { recording, name: None }, cx).unwrap();
                s.set_workspace(Workspace::Pipeline, cx);
            })
        });
        cx.run_until_parked();
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("graph-canvas").is_some(), "the Pipeline workspace is a canvas, recording open or not");
            assert!(window.try_find("node-s1").is_some() && window.try_find("node-v1").is_some(), "the recording and its view are nodes");
            assert!(window.try_find("layer-s1").is_some(), "the recording is a layer");
            assert!(window.try_find("pipeline-toolbar").is_some(), "the floating toolbar");
        })
        .unwrap();
        // An operation from the catalogue (what the toolbar's menu does) goes into the graph
        cx.update_window(window, |_, window, cx| {
            let pipeline = app.read(cx).pipeline.clone();
            let template = crate::screens::pipeline::catalog::catalog().into_iter().find(|c| c.name == "Filter").expect("a Filter family").items[0].template.clone();
            pipeline.update(cx, |p, cx| p.add(&template, window, cx));
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| {
            let g = &store.read(cx).doc.graph;
            assert_eq!(g.nodes.len(), 3);
            assert!(g.node("iir").is_some(), "the band-pass went in as op `iir`");
        });
        // A recipe (a group of ops) is one card with its tab
        cx.update(|cx| {
            let pipeline = app.read(cx).pipeline.clone();
            let recipe = crate::domain::recipe::builtin().into_iter().next().expect("a built-in recipe");
            pipeline.update(cx, |p, cx| p.add_recipe(recipe, cx));
        });
        cx.run_until_parked();
        let recipe_node = cx.update(|cx| {
            store.read(cx).doc.graph.nodes.iter().find(|n| matches!(n.kind, crate::domain::graph::NodeKind::Recipe { .. })).map(|n| n.id.clone())
        });
        let recipe_node = recipe_node.expect("the recipe went into the graph as one node");
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find(gpui_kit::SharedString::from(format!("node-{recipe_node}"))).is_some(), "drawn as one card");
        })
        .unwrap();
    }

    #[gpui_kit::test]
    fn views_render_and_follow_one_timeline(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        assert_eq!(views(cx, &app).len(), 2, "default layout: traces over heatmap");
        let vs = views(cx, &app);
        let kinds: Vec<TimeViewKind> = cx.update(|cx| vs.iter().map(|v| v.read(cx).view.kind).collect());
        assert_eq!(kinds, vec![TimeViewKind::Traces, TimeViewKind::Heatmap]);
        // Both are laid out side by side in the dock and get a frame
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
        cx.update_window(window, |_, window, _| {
            for id in [1usize, 2] {
                let b = window.find(("time-view", id)).bounds().size;
                assert!(b.width > px(300.) && b.height > px(100.), "view {id}: {b:?}");
            }
            // One panel per side: Channels, the Inspector (the default right panel), the timeline
            for id in ["channels-panel", "inspector-panel", "timeline-panel"] {
                assert!(window.try_find(id).is_some(), "{id} on screen");
            }
        })
        .unwrap();

        // Moving the shared window redraws every view for it
        cx.update(|cx| store.update(cx, |s, cx| s.pan_time(1.0, cx)));
        cx.update(|cx| assert!((store.read(cx).timeline.window_start_sec - 1.0).abs() < 1e-9));
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));

        // Zooming keeps the time under the anchor
        cx.update(|cx| store.update(cx, |s, cx| s.zoom_at(0.5, 0.5, cx)));
        cx.update(|cx| {
            let tl = &store.read(cx).timeline;
            assert!((tl.visible_window_sec - 0.05).abs() < 1e-9 && (tl.window_start_sec - 1.025).abs() < 1e-9, "{tl:?}");
        });
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
    }

    #[gpui_kit::test]
    fn add_view_selection_and_docks(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));

        // A third view joins the dock beside the focused one and renders too
        cx.update(|cx| app.update(cx, |a, cx| a.add_view(TimeViewKind::Traces, cx)));
        cx.run_until_parked();
        assert_eq!(views(cx, &app).len(), 3);
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
        cx.update_window(window, |_, window, _| assert!(window.try_find(("time-view", 3usize)).is_some())).unwrap();

        // One selection per source: every view of it follows
        cx.update(|cx| store.update(cx, |s, cx| s.select_ranges("main", "2-5, 9", cx).unwrap()));
        cx.run_until_parked();
        let vs = views(cx, &app);
        cx.update(|cx| {
            for v in &vs {
                assert_eq!(v.read(cx).view.selection, vec![2, 3, 4, 5, 9]);
            }
        });
        assert!(cx.update(|cx| store.update(cx, |s, cx| s.select_ranges("main", "40", cx))).is_err(), "out of range");

        // The sides open and close: the icon bar at the left edge, and the panel's hide button
        let explore = cx.update(|cx| app.read(cx).explore.clone());
        let open = |cx: &mut TestAppContext| cx.update(|cx| explore.read(cx).is_side_open(DockPlacement::Left));
        cx.update_window(window, |_, window, cx| app.update(cx, |a, cx| a.toggle_dock(DockPlacement::Left, window, cx))).unwrap();
        cx.run_until_parked();
        assert!(!open(cx));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("channels-panel").is_none(), "a hidden side hides its panel");
            let rail = window.find("rail-channels").bounds();
            assert!(rail.origin.x < px(40.), "icon bar at the window's left edge: {rail:?}");
            window.click("rail-channels", cx);
        })
        .unwrap();
        cx.run_until_parked();
        assert!(open(cx));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("dspk-app.channels.toggle", cx);
        })
        .unwrap();
        cx.run_until_parked();
        assert!(!open(cx));
        // The right bar: the Inspector shows by default; its icon hides the side, and shows it again
        let right = |cx: &mut TestAppContext| cx.update(|cx| explore.read(cx).is_side_open(DockPlacement::Right));
        assert!(right(cx));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("rail-inspector", cx);
        })
        .unwrap();
        cx.run_until_parked();
        assert!(!right(cx));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("inspector-panel").is_none());
            window.click("rail-inspector", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("inspector-panel").is_some(), "the inspector: one panel for view, display and processing");
        })
        .unwrap();
    }

    /// Second pixel of the top row of the traces view's frame (BGRA): background (the first column
    /// can hold the grid line of the 0 s tick; a heatmap's pixels are all data).
    fn corner_pixels(cx: &mut TestAppContext, app: &Entity<DspApp>) -> Vec<Option<[u8; 4]>> {
        let vs = views(cx, app);
        cx.update(|cx| vs.iter().take(1).map(|v| v.read(cx).shown.as_ref().and_then(|s| s.image.as_bytes(0)).map(|b| [b[4], b[5], b[6], b[7]])).collect())
    }

    #[gpui_kit::test]
    fn theme_switch_redraws_plots(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        // Start dark, whatever the test platform's appearance
        cx.update_window(window, |_, window, cx| app.update(cx, |a, cx| DspApp::apply_theme(&a.store, Some(true), window, cx))).unwrap();
        let dark = crate::ui::theme::palette::Palette::DARK.background;
        wait_until(cx, window, |cx| corner_pixels(cx, &app).iter().all(|p| *p == Some([dark.b, dark.g, dark.r, 255])));

        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            window.click("theme", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| {
            assert_eq!(store.read(cx).session.dark, Some(false), "the choice is saved");
            assert!(!store.read(cx).palette().dark);
        });
        wait_until(cx, window, |cx| corner_pixels(cx, &app).iter().all(|p| *p == Some([255, 255, 255, 255])));

        // Dark plots in the light theme
        cx.update(|cx| store.update(cx, |s, cx| s.set_dark_plots(true, cx)));
        wait_until(cx, window, |cx| corner_pixels(cx, &app).iter().all(|p| *p == Some([dark.b, dark.g, dark.r, 255])));
    }

    #[gpui_kit::test]
    fn zoomed_out_views_fill_from_one_background_pyramid(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
        // The whole 10 s recording in one window: far more samples than pixels
        cx.update(|cx| store.update(cx, |s, cx| s.set_window(0.0, 10.0, cx)));
        wait_until(cx, window, |cx| cx.update(|cx| store.read(cx).pyramids.get("main").is_some_and(|&(done, total)| done == total)));
        cx.update(|cx| {
            assert_eq!(store.read(cx).pyramids.len(), 1, "one pyramid for the source both views show");
            assert!(store.read(cx).pyramid_label().is_none(), "nothing left to build");
            let ds = store.read(cx).sources().unwrap().get("main");
            let p = ds.signal().progress();
            assert_eq!(p.done, p.total);
        });
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
    }

    /// Test formats: `*.twosource` (in memory: `array`, 16 channels, and `emg`, 2 channels, at
    /// 10 kHz for 60 s) and the formats of dspk-synapse (NWB, SpikeGLX…), as the app's plugin.
    struct TestFormats;

    impl TestFormats {
        fn memory(id: &str) -> dspk_core::MemoryRecording {
            let (channels, rate, samples) = (if id == "emg" { 2 } else { 16 }, 10_000.0, 600_000);
            let data: Vec<f32> = (0..channels * samples).map(|i| (((i % samples) as f32) * 0.05 * (1 + i / samples) as f32).sin() * 100.0).collect();
            dspk_core::MemoryRecording::new(id, data, channels, rate).unwrap()
        }
    }

    impl crate::plugin::SourceOpener for TestFormats {
        fn reads(&self, path: &std::path::Path) -> bool {
            path.extension().is_some_and(|e| e == "twosource") || dspk_synapse::io::detect(path).is_ok()
        }
        fn sources(&self, path: &std::path::Path) -> dspk_core::DspResult<Vec<dspk_io::SourceEntry>> {
            if path.extension().is_some_and(|e| e == "twosource") {
                return Ok(["array", "emg"].iter().map(|id| {
                    let mut e = dspk_io::core::single_source(&Self::memory(id));
                    e.id = id.to_string();
                    e.name = id.to_uppercase();
                    e
                }).collect());
            }
            dspk_synapse::io::sources(path)
        }
        fn open(&self, path: &std::path::Path, id: &str) -> dspk_core::DspResult<Box<dyn dspk_core::RecordingSource>> {
            if path.extension().is_some_and(|e| e == "twosource") {
                return Ok(Box::new(Self::memory(id)));
            }
            dspk_synapse::io::open_source(path, id)
        }
    }

    impl crate::plugin::AppPlugin for TestFormats {
        fn id(&self) -> &'static str {
            "test-formats"
        }
        fn name(&self) -> &'static str {
            "Test formats"
        }
        fn opener(&self) -> Option<&dyn crate::plugin::SourceOpener> {
            Some(self)
        }
    }

    /// Registers [`TestFormats`] (once per test binary).
    fn test_formats() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::plugin::register(vec![std::sync::Arc::new(TestFormats)]));
    }

    /// As [`wait_until`], for a real file (up to two minutes).
    fn wait_long(cx: &mut TestAppContext, window: AnyWindowHandle, mut done: impl FnMut(&mut TestAppContext) -> bool) {
        cx.executor().allow_parking();
        for _ in 0..12_000 {
            cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
            cx.run_until_parked();
            if done(cx) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out");
    }

    /// A view of a source that is closed, or switched to another source, leaves nothing behind in
    /// the session (Pipeline) unless it holds work: a view with ops stays.
    #[gpui_kit::test]
    fn closed_views_leave_no_trace_unless_they_hold_work(cx: &mut TestAppContext) {
        use crate::domain::command::SessionCommand;
        test_formats();
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/dspk-app-core-tests"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("release.twosource");
        std::fs::write(&path, b"").unwrap();
        let (window, store, app) = open(cx);
        cx.update(|cx| store.update(cx, |s, cx| s.open(path.clone(), cx)));
        wait_long(cx, window, |cx| cx.update(|cx| store.read(cx).recording.is_some()));
        let ex = explore(cx, &app);
        let sessions = |cx: &mut TestAppContext| cx.update(|cx| store.read(cx).doc.views.iter().map(|v| v.recording.source.clone().unwrap_or_default()).collect::<Vec<_>>());
        let count = |cx: &mut TestAppContext, src: &str| sessions(cx).iter().filter(|s| *s == src).count();
        // An untouched view of EMG, closed: gone from the session
        let vm = cx.update(|cx| ex.update(cx, |ex, cx| ex.add_on(TimeViewKind::Traces, "emg", cx)));
        wait_long(cx, window, |cx| cx.update(|cx| !vm.read(cx).view.doc_view.is_empty()));
        assert_eq!(count(cx, "emg"), 1);
        let id = cx.update(|cx| vm.read(cx).id());
        cx.update(|cx| ex.update(cx, |ex, cx| ex.removed(id, cx)));
        cx.run_until_parked();
        assert_eq!(count(cx, "emg"), 0, "closing an untouched view removes it: {:?}", sessions(cx));
        // Switched to another source: the old one goes the same way
        let vm = cx.update(|cx| ex.update(cx, |ex, cx| ex.add_on(TimeViewKind::Traces, "emg", cx)));
        wait_long(cx, window, |cx| cx.update(|cx| !vm.read(cx).view.doc_view.is_empty()));
        cx.update(|cx| vm.update(cx, |vm, cx| vm.set_source("array", cx)));
        cx.run_until_parked();
        assert_eq!(count(cx, "emg"), 0, "switching away removes it");
        // A view with work stays when closed
        cx.update(|cx| vm.update(cx, |vm, cx| vm.set_source("emg", cx)));
        wait_long(cx, window, |cx| cx.update(|cx| !vm.read(cx).view.doc_view.is_empty()));
        let doc_view = cx.update(|cx| vm.read(cx).view.doc_view.clone());
        let node: dspk_pipeline::Node = serde_json::from_value(serde_json::json!({"op": "rectify"})).unwrap();
        cx.update(|cx| store.update(cx, |s, cx| s.execute(SessionCommand::AddOp { view: doc_view, node, index: None, id: None }, cx))).unwrap();
        // Drawn with the op (the device open) before the view closes: a process must not exit
        // while a render thread is still opening the device (the driver's libraries unload under it)
        cx.update(|cx| vm.update(cx, |vm, _| vm.shown = None));
        cx.update(|cx| store.update(cx, |s, cx| s.redraw(cx)));
        wait_long(cx, window, |cx| cx.update(|cx| vm.read(cx).shown.is_some()));
        let id = cx.update(|cx| vm.read(cx).id());
        cx.update(|cx| ex.update(cx, |ex, cx| ex.removed(id, cx)));
        cx.run_until_parked();
        assert_eq!(count(cx, "emg"), 1, "its processing is kept");
    }

    /// The ops of the bug report: a band-pass, CMR, Teager–Kaiser energy, a 20 Hz low-pass.
    fn report_chain() -> Vec<dspk_pipeline::Node> {
        serde_json::from_str(
            r#"[
            {"op": "iir", "band": "bandpass", "cutoff": [300.0, 3000.0], "order": 5, "design": "butterworth", "mode": "forward_backward", "start": "rest"},
            {"op": "cmr"},
            {"op": "teager_kaiser", "edge": "reflect"},
            {"op": "iir", "band": "lowpass", "cutoff": [20.0], "order": 5, "design": "butterworth", "mode": "forward_backward", "start": "rest"}]"#,
        )
        .unwrap()
    }

    /// Opens `path`, shows its `source` (not the default one) in a new view, focuses it and adds
    /// the report's ops one by one: after each, the view's preview exists, a window of `window_s`
    /// seconds reads processed (not raw) within `limit_s`, and the view draws.
    fn ops_on_a_second_source(cx: &mut TestAppContext, path: std::path::PathBuf, source: &str, window_s: f64, limit_s: f64) {
        use crate::domain::command::SessionCommand;
        test_formats();
        let (window, store, app) = open(cx);
        cx.update(|cx| store.update(cx, |s, cx| s.open(path.clone(), cx)));
        wait_long(cx, window, |cx| cx.update(|cx| store.read(cx).opening.is_none()));
        cx.update(|cx| assert!(store.read(cx).recording.is_some(), "{}", store.read(cx).status));
        cx.update(|cx| assert_ne!(store.read(cx).sources().unwrap().default_entry().id, source, "a second source"));
        let ex = explore(cx, &app);
        let vm = cx.update(|cx| ex.update(cx, |ex, cx| ex.add_on(TimeViewKind::Traces, source, cx)));
        wait_long(cx, window, |cx| cx.update(|cx| !vm.read(cx).view.doc_view.is_empty()));
        let view = cx.update(|cx| vm.read(cx).view.doc_view.clone());
        cx.update(|cx| ex.update(cx, |ex, cx| ex.focus(vm.clone(), cx)));
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).doc.active_view.as_deref(), Some(view.as_str()), "the focused view is the one edited"));
        for node in report_chain() {
            let name = format!("{node:?}");
            cx.update(|cx| store.update(cx, |s, cx| s.execute(SessionCommand::AddOp { view: view.clone(), node, index: None, id: None }, cx)).unwrap());
            cx.run_until_parked();
            let (preview, status, raw) = cx.update(|cx| {
                let s = store.read(cx);
                (s.preview_of(&view), s.status.clone(), s.sources().unwrap().get(source).signal().clone())
            });
            let preview = preview.unwrap_or_else(|| panic!("no preview after {name}; status: {status}"));
            let rate = preview.info().sample_rate_hz();
            let n = ((window_s * rate) as u64).min(preview.info().samples);
            let v = dspk_view::View { channels: vec![0, 1], start: 0, end: n, width: 2000 };
            let (mut a, mut b) = (dspk_view::Envelope::Samples(Vec::new()), dspk_view::Envelope::Samples(Vec::new()));
            let t0 = std::time::Instant::now();
            preview.view(&v, &mut a).unwrap();
            let took = t0.elapsed().as_secs_f64();
            eprintln!("{name}: {window_s} s window in {:.0} ms", took * 1e3);
            assert!(took < limit_s, "{name}: the preview took {took:.1} s");
            raw.view(&v, &mut b).unwrap();
            assert_ne!(a, b, "{name}: the view shows the processed signal");
            // The view draws the processed signal
            cx.update(|cx| vm.update(cx, |vm, _| vm.shown = None));
            cx.update(|cx| store.update(cx, |s, cx| s.redraw(cx)));
            wait_long(cx, window, |cx| cx.update(|cx| vm.read(cx).shown.is_some()));
        }
    }

    /// Ops edited on a view of a file's second source preview there, at once, after every edit.
    #[gpui_kit::test]
    fn ops_preview_on_a_second_source(cx: &mut TestAppContext) {
        // The sources are in memory; the file only has to exist (the views' caches key on it)
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/dspk-app-core-tests"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two.twosource");
        std::fs::write(&path, b"").unwrap();
        ops_on_a_second_source(cx, path, "emg", 30.0, 60.0);
    }

    /// The file of the bug report (local only): its EMG source, the report's 4 min 35 s window.
    #[gpui_kit::test]
    #[ignore = "needs .datasets/nwb/15-25-33_meps.nwb.zarr"]
    fn ops_preview_on_the_report_file(cx: &mut TestAppContext) {
        let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../.datasets/nwb/15-25-33_meps.nwb.zarr"));
        ops_on_a_second_source(cx, path, "/acquisition/EMG", 275.0, 15.0);
    }
}
