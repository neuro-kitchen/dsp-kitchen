//! The root view: a title bar with the menus, the workspaces (Explore · Sorting · Pipeline ·
//! Curation) and the open recording; the workspace shown; a status bar. Without a recording the
//! body is a start screen (open a file, a recent one, or a synthetic recording).

use std::path::PathBuf;

use gpui_kit::base::dock::DockPlacement;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::status_bar::StatusBar;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Selectable as _, Sizable as _, Theme, ThemeMode, TitleBar};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, ExternalPaths, FocusHandle, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _,
    PathPromptOptions, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::actions::*;
use crate::engine::time::renderer::TimeViewKind;
use crate::store::{file_label, AppEvent, Store};
use crate::views::{CurationView, ExploreView};
use crate::widgets::EmptyState;
use crate::workspace::Workspace;

/// The synthetic recording of the start screen and the File menu.
pub const SYNTHETIC: (usize, f64, f64) = (32, 30_000.0, 300.0);

pub struct DspApp {
    store: Entity<Store>,
    pub explore: Entity<ExploreView>,
    pub curation: Entity<CurationView>,
    show_help: bool,
    show_about: bool,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl DspApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let choice = store.read(cx).session.dark;
        Self::apply_theme(&store, choice, window, cx);
        let explore = cx.new(|cx| ExploreView::new(store.clone(), window, cx));
        let curation = cx.new(|cx| CurationView::new(store.clone(), window, cx));
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
        Self { store, explore, curation, show_help: false, show_about: false, focus, _subs: subs }
    }

    /// Light, dark, or the system's (`None`); the store learns which, for the plot colours.
    fn apply_theme(store: &Entity<Store>, dark: Option<bool>, window: &mut Window, cx: &mut App) {
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

    /// Asks for a sorting folder and opens it in Curation.
    fn open_sorting(&mut self, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.set_workspace(Workspace::Curation, cx));
        self.curation.update(cx, |c, cx| c.prompt_open(cx));
    }

    /// Opens `path`: a sorting folder in Curation, else a recording in Explore.
    pub fn open_any(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if crate::viewmodels::curation::is_sorting(&path) {
            let vm = self.curation.read(cx).vm.clone();
            vm.update(cx, |vm, cx| vm.open(path, cx));
        } else {
            self.store.update(cx, |s, cx| s.open(path, cx));
        }
    }

    fn prompt_open(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: true, multiple: false, prompt: Some("Open recording".into()) });
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
        let file = {
            let this = this.clone();
            Button::new("menu-file").ghost().small().label("File").dropdown_menu(move |menu: PopupMenu, _, _| {
                let t = this.clone();
                let mut menu = menu.item(PopupMenuItem::new("Open recording…").on_click(move |_, _, cx| t.update(cx, |a, cx| a.prompt_open(cx))));
                let t = this.clone();
                menu = menu.item(PopupMenuItem::new("Open sorting…").on_click(move |_, _, cx| t.update(cx, |a, cx| a.open_sorting(cx))));
                let t = this.clone();
                menu = menu.item(PopupMenuItem::new("Synthetic recording (32 ch, 5 min)").on_click(move |_, _, cx| t.update(cx, |a, cx| a.open_synthetic(cx))));
                if !recent.is_empty() {
                    menu = menu.separator().label("Recent");
                    for p in &recent {
                        let (t, p) = (this.clone(), p.clone());
                        menu = menu.item(PopupMenuItem::new(file_label(&p)).on_click(move |_, _, cx| t.update(cx, |a, cx| a.open_path(p.clone(), cx))));
                    }
                }
                let t = this.clone();
                menu.separator()
                    .item(PopupMenuItem::new("Build min/max caches").disabled(!open).on_click(move |_, _, cx| t.update(cx, |a, cx| a.store.update(cx, |s, cx| s.build_caches(cx)))))
                    .separator()
                    .item(PopupMenuItem::new("Quit").on_click(|_, _, cx| cx.quit()))
            })
        };
        let view = {
            let this = this.clone();
            Button::new("menu-view").ghost().small().label("View").dropdown_menu(move |menu: PopupMenu, _, _| {
                let item = |label: &'static str, f: fn(&mut DspApp, &mut Window, &mut Context<DspApp>)| {
                    let t = this.clone();
                    PopupMenuItem::new(label).disabled(!open).on_click(move |_, window, cx| t.update(cx, |a, cx| f(a, window, cx)))
                };
                menu.item(item("Add traces view", |a, _, cx| a.add_view(TimeViewKind::Traces, cx)))
                    .item(item("Add heatmap view", |a, _, cx| a.add_view(TimeViewKind::Heatmap, cx)))
                    .separator()
                    .item(item("Channels", |a, w, cx| a.toggle_dock(DockPlacement::Left, w, cx)))
                    .item(item("View settings", |a, w, cx| a.toggle_dock(DockPlacement::Right, w, cx)))
                    .item(item("Timeline", |a, w, cx| a.toggle_dock(DockPlacement::Bottom, w, cx)))
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
                menu.item(PopupMenuItem::new("Keyboard & mouse").on_click(move |_, _, cx| {
                    t1.update(cx, |a, cx| {
                        a.show_help = true;
                        cx.notify();
                    })
                }))
                .separator()
                .item(PopupMenuItem::new("About DSP App").on_click(move |_, _, cx| {
                    t2.update(cx, |a, cx| {
                        a.show_about = true;
                        cx.notify();
                    })
                }))
            })
        };
        h_flex().gap_0p5().child(file).child(view).child(help)
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
        let recording = s.recording.as_ref().map(|r| {
            h_flex()
                .gap_2()
                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(t.foreground).child(r.name.clone()))
                .child(div().text_xs().text_color(t.muted_foreground).child(r.summary()))
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
            "SpikeGLX (.bin / .cbin), raw binary with a JSON .meta, or an NWB Zarr store. Or try a synthetic recording: no file needed.",
        )
        .child(
            h_flex()
                .gap_2()
                .child(Button::new("start-open").primary().icon(IconName::FolderOpen).label("Open recording…").on_click(cx.listener(|this, _, _, cx| this.prompt_open(cx))))
                .child(Button::new("start-synthetic").outline().label("Synthetic recording").on_click(cx.listener(|this, _, _, cx| this.open_synthetic(cx)))),
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
                        Button::new(("recent", i))
                            .ghost()
                            .w_full()
                            .child(h_flex().w_full().gap_2().child(div().text_sm().child(label)).child(div().flex_1().text_xs().text_color(muted).truncate().child(dir)))
                            .on_click(cx.listener(move |this, _, _, cx| this.open_path(p.clone(), cx)))
                    })),
            );
        }
        div().id("start-screen").test_support().size_full().child(screen).into_any_element()
    }

    fn body(&self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let workspace = s.workspace;
        if let Some(text) = workspace.planned() {
            return div()
                .id("planned-workspace")
                .test_support()
                .size_full()
                .child(EmptyState::new(workspace.icon(), format!("{} is on its way", workspace.title()), text))
                .into_any_element();
        }
        if workspace == Workspace::Curation {
            return self.curation.clone().into_any_element();
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
                            .gap_2()
                            .items_center()
                            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("DSP App"))
                            .child(div().text_xs().text_color(muted).child(format!("v{}", env!("CARGO_PKG_VERSION")))),
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
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("dsp-app")
            .track_focus(&self.focus)
            .size_full()
            .relative()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .on_action(cx.listener(|this, _: &OpenRecording, _, cx| this.prompt_open(cx)))
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
            .on_action(cx.listener(|this, _: &ShowAbout, _, cx| {
                this.show_about = !this.show_about;
                cx.notify();
            }))
            .on_action(|_: &Quit, _, cx: &mut App| cx.quit())
            // A file or folder dropped anywhere opens it
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first().cloned() {
                    this.open_any(p, cx);
                }
            }))
            .child(self.title_bar(cx))
            .child(div().flex_1().min_h_0().child(self.body(cx)))
            .child(self.status_bar(cx))
            .when(self.show_help, |d| d.child(self.help(cx)))
            .when(self.show_about, |d| d.child(self.about(cx)))
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
    use crate::engine::data::{Dataset, SourceSet};
    use crate::engine::time::renderer::TimeViewKind;
    use crate::session::Session;
    use crate::store::{Recording, Store};
    use crate::viewmodels::{ExploreVm, TraceVm};
    use crate::workspace::Workspace;

    fn open(cx: &mut TestAppContext) -> (AnyWindowHandle, Entity<Store>, Entity<DspApp>) {
        // Frames come back from real render threads: allow their wake-ups from the start
        cx.executor().allow_parking();
        cx.update(|cx| {
            gpui_kit::init(cx);
            crate::actions::bind_keys(cx);
            crate::viewmodels::Services::install(cx);
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

    #[gpui_kit::test]
    fn start_screen_then_workspaces(cx: &mut TestAppContext) {
        let (window, store, _) = open(cx);
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("start-screen").is_some(), "no recording: the start screen");
            window.click("workspace-pipeline", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(store.read(cx).workspace, Workspace::Pipeline));
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
            assert!(window.try_find("channels-panel").is_some() && window.try_find("settings-panel").is_some() && window.try_find("timeline-panel").is_some());
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

        // The docks open and close
        let dock = cx.update(|cx| app.read(cx).explore.read(cx).dock.clone());
        cx.update_window(window, |_, window, cx| app.update(cx, |a, cx| a.toggle_dock(DockPlacement::Left, window, cx))).unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert!(!dock.read(cx).is_dock_open(DockPlacement::Left)));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("channels-panel").is_none(), "closed dock hides its panel");
            // The rail at the left edge shows it again
            let rail = window.find("show-channels").bounds();
            assert!(rail.origin.x < px(40.), "rail at the window's left edge: {rail:?}");
            window.click("show-channels", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert!(dock.read(cx).is_dock_open(DockPlacement::Left)));
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            assert!(window.try_find("show-channels").is_none());
            // The hide button sits at the panel's outer (left) edge
            let hide = window.find("dsp-app.channels.toggle").bounds();
            assert!(hide.origin.x < px(40.), "{hide:?}");
            window.click("dsp-app.channels.toggle", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert!(!dock.read(cx).is_dock_open(DockPlacement::Left)));
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
        let dark = crate::engine::palette::Palette::DARK.background;
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
    fn zoomed_out_views_fill_from_one_background_summary(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        install(cx, &store);
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
        // The whole 10 s recording in one window: far more samples than pixels
        cx.update(|cx| store.update(cx, |s, cx| s.set_window(0.0, 10.0, cx)));
        wait_until(cx, window, |cx| cx.update(|cx| store.read(cx).summaries.get("main").is_some_and(|&(done, total)| done == total)));
        cx.update(|cx| {
            assert_eq!(store.read(cx).summaries.len(), 1, "one summary for the source both views show");
            assert!(store.read(cx).summary_label().is_none(), "nothing left to summarize");
            let ds = store.read(cx).sources().unwrap().get("main");
            assert!(ds.summary().covers(0, ds.total_samples as u64));
        });
        wait_until(cx, window, |cx| all_drawn_for_window(cx, &store, &app));
    }

    #[gpui_kit::test]
    fn a_phy_folder_opens_in_curation(cx: &mut TestAppContext) {
        let (window, store, app) = open(cx);
        let dir = crate::engine::curation::tests::synthetic_folder("ui-open");
        // As `dsp-app <path>`: a file inside the folder is enough
        cx.update(|cx| app.update(cx, |a, cx| a.open_any(dir.join("params.py"), cx)));
        let vm = cx.update(|cx| app.read(cx).curation.read(cx).vm.clone());
        wait_until(cx, window, |cx| cx.update(|cx| vm.read(cx).correlograms.is_some() && !vm.read(cx).waveforms.is_empty()));
        cx.update(|cx| {
            assert_eq!(store.read(cx).workspace, Workspace::Curation);
            let v = vm.read(cx);
            assert_eq!(v.rows().len(), 6);
            assert_eq!(v.selected, vec![0], "the first cluster is selected");
            assert_eq!(v.waveforms[0].template.len(), v.waveforms[0].channels.len() * v.waveforms[0].num_samples);
            assert!(v.waveforms[0].sampled.is_empty(), "no recording: no spikes invented");
            assert_eq!(v.similar().len(), 5);
        });
        cx.update_window(window, |_, window, cx| {
            window.render_frame(cx);
            for id in ["clusters-panel", "similar-panel", "waveforms-panel", "correlograms-panel"] {
                assert!(window.try_find(id).is_some(), "{id}");
            }
            // Pick a second cluster from the Similar list: compared with the first
            window.click(("similar", 3usize), cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(vm.read(cx).selected, vec![0, 3]));
        wait_until(cx, window, |cx| cx.update(|cx| vm.read(cx).correlograms.as_ref().is_some_and(|m| m.clusters == vec![0, 3])));
        cx.update(|cx| assert_eq!(vm.read(cx).waveforms.len(), 2));
    }
}
