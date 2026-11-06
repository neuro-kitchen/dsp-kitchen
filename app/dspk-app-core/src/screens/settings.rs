//! Settings (Ctrl+,): the servers the app works with (add, edit, remove, test), where results
//! fetched from them go, the compute runtime, the theme. Every value shows its default.

use std::path::PathBuf;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Sizable as _};
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, PathPromptOptions, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Task, Window,
};

use crate::data::preferences::Session;
use crate::data::remote::{self, ServerProfile, DEFAULT_SERVER_NAME};
use crate::data::store::Store;
use crate::ui::widgets::{row, MenuSelect};

/// Compute runtimes offered (the first: whatever is compiled in first).
const RUNTIMES: [(&str, Option<&str>); 4] = [("Automatic (first compiled in)", None), ("wgpu (Vulkan, Metal, DX12)", Some("wgpu")), ("CUDA", Some("cuda")), ("CPU", Some("cpu"))];
const THEMES: [(&str, Option<bool>); 3] = [("Follow the system", None), ("Dark", Some(true)), ("Light", Some(false))];

/// What the server form is doing.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    Closed,
    New,
    Edit(usize),
}

pub struct SettingsScreen {
    store: Entity<Store>,
    name: Entity<InputState>,
    address: Entity<InputState>,
    server_name: Entity<InputState>,
    identity: Entity<InputState>,
    results: Entity<InputState>,
    form: Form,
    /// The last test or save: ok, and what to say.
    message: Option<(bool, String)>,
    testing: bool,
    _tasks: Vec<Task<()>>,
}

impl SettingsScreen {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut input = |cx: &mut Context<Self>, placeholder: &str| {
            let placeholder = placeholder.to_string();
            cx.new(|cx| InputState::new(window, cx).placeholder(placeholder))
        };
        let (name, address, server_name, identity) = (input(cx, "e.g. lab GPU"), input(cx, "e.g. 192.168.1.20:50051"), input(cx, DEFAULT_SERVER_NAME), input(cx, "the file `dspk clients enroll` wrote"));
        let default_results = Session::default_results_dir().display().to_string();
        let current = store.read(cx).session.results_dir.clone().map(|p| p.display().to_string()).unwrap_or_default();
        let results = cx.new(|cx| InputState::new(window, cx).placeholder(default_results).default_value(current));
        Self { store, name, address, server_name, identity, results, form: Form::Closed, message: None, testing: false, _tasks: Vec::new() }
    }

    fn text(input: &Entity<InputState>, cx: &App) -> String {
        input.read(cx).value().trim().to_string()
    }

    /// The profile in the form (an empty certificate name means the default).
    fn profile(&self, cx: &App) -> ServerProfile {
        let server_name = Self::text(&self.server_name, cx);
        ServerProfile {
            name: Self::text(&self.name, cx),
            address: Self::text(&self.address, cx),
            server_name: if server_name.is_empty() { DEFAULT_SERVER_NAME.into() } else { server_name },
            identity: PathBuf::from(Self::text(&self.identity, cx)),
        }
    }

    fn open_form(&mut self, form: Form, window: &mut Window, cx: &mut Context<Self>) {
        let p = match form {
            Form::Edit(i) => self.store.read(cx).session.servers.get(i).cloned(),
            _ => None,
        };
        let (n, a, s, id) = match &p {
            Some(p) => (p.name.clone(), p.address.clone(), p.server_name.clone(), p.identity.display().to_string()),
            None => Default::default(),
        };
        self.name.update(cx, |i, cx| i.set_value(n, window, cx));
        self.address.update(cx, |i, cx| i.set_value(a, window, cx));
        self.server_name.update(cx, |i, cx| i.set_value(s, window, cx));
        self.identity.update(cx, |i, cx| i.set_value(id, window, cx));
        self.form = form;
        self.message = None;
        cx.notify();
    }

    fn save_form(&mut self, cx: &mut Context<Self>) {
        let profile = self.profile(cx);
        if let Err(e) = profile.check() {
            self.message = Some((false, e));
            cx.notify();
            return;
        }
        let mut servers = self.store.read(cx).session.servers.clone();
        if servers.iter().enumerate().any(|(i, s)| s.name == profile.name && self.form != Form::Edit(i)) {
            self.message = Some((false, format!("a server named {:?} exists already", profile.name)));
            cx.notify();
            return;
        }
        match self.form {
            Form::Edit(i) if i < servers.len() => servers[i] = profile.clone(),
            _ => servers.push(profile.clone()),
        }
        self.store.update(cx, |s, cx| s.set_servers(servers, cx));
        self.form = Form::Closed;
        self.message = Some((true, format!("Saved {}", profile.name)));
        cx.notify();
    }

    fn remove(&mut self, index: usize, cx: &mut Context<Self>) {
        let mut servers = self.store.read(cx).session.servers.clone();
        if index < servers.len() {
            let gone = servers.remove(index);
            self.store.update(cx, |s, cx| s.set_servers(servers, cx));
            self.message = Some((true, format!("Removed {} (its identity file is untouched)", gone.name)));
            self.form = Form::Closed;
            cx.notify();
        }
    }

    /// Connects with `profile` and says who the server is and who this app is to it.
    fn test(&mut self, profile: ServerProfile, cx: &mut Context<Self>) {
        if let Err(e) = profile.check() {
            self.message = Some((false, e));
            cx.notify();
            return;
        }
        self.testing = true;
        self.message = Some((true, format!("Connecting to {}…", profile.address)));
        let handle = remote::runtime().spawn(remote::test(profile));
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = handle.await;
            let _ = this.update(cx, |s, cx| {
                s.testing = false;
                s.message = Some(match result {
                    Ok(Ok(info)) => {
                        let permission = if info.permission == dspk_stream::protocol::wire::Permission::Control as i32 { "control" } else { "read" };
                        let jobs = if info.runtime.is_empty() { String::new() } else { format!(", jobs on {}", info.runtime) };
                        (true, format!("Connected: {} (dspk {}), you are {} ({permission}){}{jobs}", info.server_name, info.server_version, info.peer_name, if info.has_files { ", files served" } else { "" }))
                    }
                    Ok(Err(e)) => (false, format!("{e:#}")),
                    Err(e) => (false, e.to_string()),
                });
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// A file or folder picked for `input`.
    fn browse(&mut self, input: Entity<InputState>, folder: bool, prompt: &str, window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: !folder, directories: folder, multiple: false, prompt: Some(prompt.to_string().into()) });
        self._tasks.push(cx.spawn_in(window, async move |_, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(path) = paths.into_iter().next()
            {
                let _ = cx.update(|window, cx| input.update(cx, |i, cx| i.set_value(path.display().to_string(), window, cx)));
            }
        }));
    }

    fn save_results(&mut self, cx: &mut Context<Self>) {
        let text = Self::text(&self.results, cx);
        let dir = (!text.is_empty()).then(|| PathBuf::from(text));
        self.store.update(cx, |s, cx| s.set_results_dir(dir, cx));
        self.message = Some((true, "Results folder saved".into()));
        cx.notify();
    }

    fn section(title: &str, cx: &App) -> impl IntoElement + use<> {
        div().pt_3().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(cx.theme().foreground).child(title.to_string())
    }
}

impl Render for SettingsScreen {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, danger, ok, border) = (t.muted_foreground, t.danger, t.success, t.border);
        let me = cx.entity().downgrade();
        let (servers, results_dir, runtime, theme_choice, dark_plots) = {
            let s = &self.store.read(cx).session;
            (s.servers.clone(), s.results_dir(), s.runtime.clone(), s.dark, s.dark_plots)
        };

        // Servers: one row each, then the form
        let mut list = v_flex().gap_1();
        if servers.is_empty() {
            list = list.child(div().text_xs().text_color(muted).child("No servers yet. Add one with the identity file its administrator gave you (`dspk clients enroll`)."));
        }
        for (i, p) in servers.iter().enumerate() {
            let (m1, m2, m3) = (me.clone(), me.clone(), me.clone());
            let profile = p.clone();
            list = list.child(
                h_flex()
                    .id(SharedString::from(format!("server-{i}")))
                    .w_full()
                    .min_w_0()
                    .gap_2()
                    .py_1()
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .child(div().text_sm().font_weight(FontWeight::MEDIUM).truncate().child(p.name.clone()))
                            .child(div().text_xs().text_color(muted).truncate().child(format!("{} · certificate name {}", p.address, p.server_name))),
                    )
                    .child(Button::new(SharedString::from(format!("server-test-{i}"))).outline().xsmall().label("Test").disabled(self.testing).on_click(move |_, _, cx| {
                        let _ = m1.update(cx, |s, cx| s.test(profile.clone(), cx));
                    }))
                    .child(Button::new(SharedString::from(format!("server-edit-{i}"))).ghost().xsmall().label("Edit").on_click(move |_, window, cx| {
                        let _ = m2.update(cx, |s, cx| s.open_form(Form::Edit(i), window, cx));
                    }))
                    .child(Button::new(SharedString::from(format!("server-remove-{i}"))).ghost().xsmall().label("Remove").on_click(move |_, _, cx| {
                        let _ = m3.update(cx, |s, cx| s.remove(i, cx));
                    })),
            );
        }
        let form = (self.form != Form::Closed).then(|| {
            let (m1, m2, m3, m4) = (me.clone(), me.clone(), me.clone(), me.clone());
            let identity = self.identity.clone();
            v_flex()
                .gap_2()
                .p_3()
                .rounded_md()
                .border_1()
                .border_color(border)
                .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).child(if self.form == Form::New { "New server" } else { "Edit server" }))
                .child(row("Name", div().w_full().child(Input::new(&self.name).small()), cx))
                .child(row("Address", div().w_full().child(Input::new(&self.address).small()), cx))
                .child(row("Certificate name", div().w_full().child(Input::new(&self.server_name).small()), cx))
                .child(row(
                    "Identity file",
                    h_flex()
                        .w_full()
                        .gap_1()
                        .child(div().flex_1().min_w_0().child(Input::new(&self.identity).small()))
                        .child(Button::new("server-identity-browse").outline().xsmall().label("Browse…").on_click(move |_, window, cx| {
                            let _ = m1.update(cx, |s, cx| s.browse(identity.clone(), false, "Identity file", window, cx));
                        })),
                    cx,
                ))
                .child(div().text_xs().text_color(muted).child(format!("Certificate name: default {DEFAULT_SERVER_NAME}; one of the names the server was started with (--name).")))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Button::new("server-save").primary().small().label("Save").on_click(move |_, _, cx| {
                            let _ = m2.update(cx, |s, cx| s.save_form(cx));
                        }))
                        .child(Button::new("server-test-form").outline().small().label("Test").disabled(self.testing).on_click(move |_, _, cx| {
                            let _ = m3.update(cx, |s, cx| {
                                let p = s.profile(cx);
                                s.test(p, cx)
                            });
                        }))
                        .child(Button::new("server-cancel").ghost().small().label("Cancel").on_click(move |_, _, cx| {
                            let _ = m4.update(cx, |s, cx| {
                                s.form = Form::Closed;
                                cx.notify();
                            });
                        })),
                )
        });
        let add = (self.form == Form::Closed).then(|| {
            let m = me.clone();
            Button::new("server-add").outline().small().label("Add server").on_click(move |_, window, cx| {
                let _ = m.update(cx, |s, cx| s.open_form(Form::New, window, cx));
            })
        });

        // Results
        let (m_rb, m_rs) = (me.clone(), me.clone());
        let results_input = self.results.clone();
        let results = row(
            "Fetched results",
            h_flex()
                .w_full()
                .gap_1()
                .child(div().flex_1().min_w_0().child(Input::new(&self.results).small()))
                .child(Button::new("results-browse").outline().xsmall().label("Browse…").on_click(move |_, window, cx| {
                    let _ = m_rb.update(cx, |s, cx| s.browse(results_input.clone(), true, "Folder for fetched results", window, cx));
                }))
                .child(Button::new("results-save").outline().xsmall().label("Save").on_click(move |_, _, cx| {
                    let _ = m_rs.update(cx, |s, cx| s.save_results(cx));
                })),
            cx,
        );

        // Compute and appearance
        let store = self.store.clone();
        let runtime_names: Vec<SharedString> = RUNTIMES.iter().map(|(n, _)| SharedString::from(*n)).collect();
        let runtime_sel = RUNTIMES.iter().position(|(_, v)| v.map(str::to_string) == runtime);
        let s1 = store.clone();
        let runtime_menu = MenuSelect::new("settings-runtime", runtime_names, runtime_sel, move |i, _, cx| {
            let choice = RUNTIMES[i].1.map(str::to_string);
            s1.update(cx, |s, cx| s.set_runtime(choice, cx));
        });
        let now = std::env::var("DSPK_RUNTIME").unwrap_or_else(|_| "automatic".into());
        let theme_names: Vec<SharedString> = THEMES.iter().map(|(n, _)| SharedString::from(*n)).collect();
        let theme_sel = THEMES.iter().position(|(_, v)| *v == theme_choice);
        let s2 = store.clone();
        let theme_menu = MenuSelect::new("settings-theme", theme_names, theme_sel, move |i, window, cx| {
            let choice = THEMES[i].1;
            s2.update(cx, |s, cx| s.set_theme_choice(choice, cx));
            crate::app::DspApp::apply_theme(&s2, choice, window, cx);
        });
        let s3 = store.clone();
        let plots = Checkbox::new("settings-dark-plots").checked(dark_plots).on_click(move |on: &bool, _, cx| {
            let on = *on;
            s3.update(cx, |s, cx| s.set_dark_plots(on, cx));
        });

        v_flex()
            .id("settings")
            .w(px(560.))
            .max_h(gpui_kit::relative(0.85))
            .overflow_y_scroll()
            .gap_2()
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Settings"))
            .child(Self::section("Servers", cx))
            .child(list)
            .children(form)
            .children(add)
            .children(self.message.clone().map(|(good, text)| div().text_xs().text_color(if good { ok } else { danger }).child(text)))
            .child(Self::section("Results", cx))
            .child(results)
            .child(div().text_xs().text_color(muted).child(format!("Now: {} (default {}); each server's results in a folder of its own.", results_dir.display(), Session::default_results_dir().display())))
            .child(Self::section("Compute", cx))
            .child(row("Runtime", runtime_menu, cx))
            .child(div().text_xs().text_color(muted).child(format!("Takes effect at the next start (now: {now}). DSPK_RUNTIME in the environment overrides it.")))
            .child(Self::section("Appearance", cx))
            .child(row("Theme", theme_menu, cx))
            .child(row("Dark plots", plots, cx))
    }
}

/// Headless: the server form saves a checked profile, refuses bad ones, removes.
#[cfg(test)]
mod tests {
    use gpui_kit::{px, size, Bounds, Point, TestAppContext, WindowBounds, WindowOptions};

    use super::*;

    #[gpui_kit::test]
    fn servers_are_saved_checked_and_removed(cx: &mut TestAppContext) {
        let identity = std::env::temp_dir().join(format!("dspk-app-settings-{}.dspk-identity.pem", std::process::id()));
        std::fs::write(&identity, "test").unwrap();
        let (window, store, screen) = cx.update(|cx| {
            gpui_kit::init(cx);
            let store = cx.new(|_| Store::new(Session::new(), None));
            let options = WindowOptions { window_bounds: Some(WindowBounds::Windowed(Bounds { origin: Point::default(), size: size(px(800.), px(700.)) })), ..Default::default() };
            let s = store.clone();
            let (window, screen) = gpui_kit::open_window(options, cx, move |window, cx| cx.new(|cx| SettingsScreen::new(s, window, cx))).expect("open test window");
            (window, store, screen)
        });
        let fill = |cx: &mut TestAppContext, name: &str, address: &str| {
            let (name, address, id) = (name.to_string(), address.to_string(), identity.display().to_string());
            cx.update_window(window, |_, window, cx| {
                screen.update(cx, |s, cx| {
                    s.open_form(Form::New, window, cx);
                    s.name.update(cx, |i, cx| i.set_value(name, window, cx));
                    s.address.update(cx, |i, cx| i.set_value(address, window, cx));
                    s.identity.update(cx, |i, cx| i.set_value(id, window, cx));
                    s.save_form(cx);
                })
            })
            .unwrap();
        };

        fill(cx, "lab", "192.168.1.20:50051");
        let servers = cx.update(|cx| store.read(cx).session.servers.clone());
        assert_eq!(servers.len(), 1);
        assert_eq!((servers[0].name.as_str(), servers[0].server_name.as_str()), ("lab", DEFAULT_SERVER_NAME), "empty certificate name → the default");

        fill(cx, "other", "lab-gpu");
        assert!(cx.update(|cx| screen.read(cx).message.clone()).is_some_and(|(ok, m)| !ok && m.contains("address")));
        fill(cx, "lab", "10.0.0.2:50051");
        assert!(cx.update(|cx| screen.read(cx).message.clone()).is_some_and(|(ok, m)| !ok && m.contains("exists")));
        assert_eq!(cx.update(|cx| store.read(cx).session.servers.len()), 1, "refused ones are not saved");

        cx.update(|cx| screen.update(cx, |s, cx| s.remove(0, cx)));
        assert!(cx.update(|cx| store.read(cx).session.servers.is_empty()));
        let _ = std::fs::remove_file(&identity);
    }
}
