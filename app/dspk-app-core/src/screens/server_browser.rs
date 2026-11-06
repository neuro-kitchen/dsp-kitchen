//! Open from server (File menu): pick a server, walk its data root, open a recording in Explore
//! (read on the server) or fetch a file or folder (a sorting) into the results folder.

use dspk_stream::protocol::wire;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Sizable as _};
use gpui_kit::{
    div, px, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _,
    Task, Window,
};

use crate::data::remote::{self, ServerProfile};
use crate::data::store::Store;
use crate::ui::widgets::{row, MenuSelect};

pub struct ServerBrowser {
    store: Entity<Store>,
    server: Option<usize>,
    /// The directory shown, relative to the server's data root ("" = the root).
    dir: String,
    list: Option<wire::FileList>,
    loading: bool,
    message: Option<(bool, String)>,
    _tasks: Vec<Task<()>>,
}

/// What the browser asks the owner to do.
pub enum BrowserEvent {
    /// A recording was opened: close the browser.
    Opened,
}

impl gpui_kit::EventEmitter<BrowserEvent> for ServerBrowser {}

impl ServerBrowser {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let first = (!store.read(cx).session.servers.is_empty()).then_some(0);
        let mut b = Self { store, server: first, dir: String::new(), list: None, loading: false, message: None, _tasks: Vec::new() };
        if first.is_some() {
            b.load(String::new(), cx);
        }
        b
    }

    fn profile(&self, cx: &Context<Self>) -> Option<ServerProfile> {
        self.server.and_then(|i| self.store.read(cx).session.servers.get(i).cloned())
    }

    /// Lists `dir` on the chosen server.
    fn load(&mut self, dir: String, cx: &mut Context<Self>) {
        let Some(profile) = self.profile(cx) else { return };
        self.loading = true;
        self.message = None;
        let handle = remote::runtime().spawn(remote::list(profile, dir.clone()));
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = handle.await;
            let _ = this.update(cx, |b, cx| {
                b.loading = false;
                match result {
                    Ok(Ok(list)) => {
                        b.dir = list.dir.clone();
                        b.list = Some(list);
                    }
                    Ok(Err(e)) => b.message = Some((false, format!("{e:#}"))),
                    Err(e) => b.message = Some((false, e.to_string())),
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    fn path_of(&self, name: &str) -> String {
        if self.dir.is_empty() { name.to_string() } else { format!("{}/{name}", self.dir) }
    }

    fn open(&mut self, name: &str, cx: &mut Context<Self>) {
        let Some(profile) = self.profile(cx) else { return };
        let path = self.path_of(name);
        self.store.update(cx, |s, cx| s.open_remote(profile, path, cx));
        cx.emit(BrowserEvent::Opened);
    }

    fn fetch(&mut self, name: &str, cx: &mut Context<Self>) {
        let Some(profile) = self.profile(cx) else { return };
        let (path, results) = (self.path_of(name), self.store.read(cx).session.results_dir());
        self.message = Some((true, format!("Fetching {path}…")));
        let handle = remote::runtime().spawn(remote::fetch(profile, path.clone(), results));
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = handle.await;
            let _ = this.update(cx, |b, cx| {
                b.message = Some(match result {
                    Ok(Ok(local)) => (true, format!("Fetched {path} → {}", local.display())),
                    Ok(Err(e)) => (false, format!("{e:#}")),
                    Err(e) => (false, e.to_string()),
                });
                cx.notify();
            });
        }));
        cx.notify();
    }
}

impl Render for ServerBrowser {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, danger, ok, border) = (t.muted_foreground, t.danger, t.success, t.border);
        let me = cx.entity().downgrade();
        let servers = self.store.read(cx).session.servers.clone();
        let names: Vec<SharedString> = servers.iter().map(|s| SharedString::from(format!("{} ({})", s.name, s.address))).collect();
        let m = me.clone();
        let picker = MenuSelect::new("browser-server", names, self.server, move |i, _, cx| {
            let _ = m.update(cx, |b, cx| {
                b.server = Some(i);
                b.list = None;
                b.load(String::new(), cx);
            });
        });

        let mut entries = v_flex().gap_0p5();
        if let Some(list) = &self.list {
            if !self.dir.is_empty() {
                let m = me.clone();
                let up = self.dir.rsplit_once('/').map_or(String::new(), |(p, _)| p.to_string());
                entries = entries.child(Button::new("browser-up").ghost().xsmall().label("↑ Up").on_click(move |_, _, cx| {
                    let _ = m.update(cx, |b, cx| b.load(up.clone(), cx));
                }));
            }
            if list.entries.is_empty() {
                entries = entries.child(div().text_xs().text_color(muted).child("Nothing here."));
            }
            for (i, e) in list.entries.iter().enumerate() {
                let kind = wire::FileKind::try_from(e.kind).unwrap_or(wire::FileKind::Other);
                let label = match kind {
                    wire::FileKind::Recording => "recording",
                    wire::FileKind::Sorting => "sorting",
                    wire::FileKind::Directory => "folder",
                    _ => "",
                };
                let size = if e.size > 0 { format!("{:.1} MB", e.size as f64 / 1e6) } else { String::new() };
                let name = e.name.clone();
                let mut actions = h_flex().gap_1().flex_none();
                if kind == wire::FileKind::Recording {
                    let (m, n) = (me.clone(), name.clone());
                    actions = actions.child(Button::new(SharedString::from(format!("browser-open-{i}"))).primary().xsmall().label("Open").on_click(move |_, _, cx| {
                        let _ = m.update(cx, |b, cx| b.open(&n, cx));
                    }));
                }
                if kind == wire::FileKind::Directory {
                    let (m, n) = (me.clone(), name.clone());
                    actions = actions.child(Button::new(SharedString::from(format!("browser-cd-{i}"))).outline().xsmall().label("Enter").on_click(move |_, _, cx| {
                        let _ = m.update(cx, |b, cx| {
                            let p = b.path_of(&n);
                            b.load(p, cx)
                        });
                    }));
                } else {
                    let (m, n) = (me.clone(), name.clone());
                    actions = actions.child(Button::new(SharedString::from(format!("browser-fetch-{i}"))).outline().xsmall().label("Fetch").on_click(move |_, _, cx| {
                        let _ = m.update(cx, |b, cx| b.fetch(&n, cx));
                    }));
                }
                entries = entries.child(
                    h_flex()
                        .id(SharedString::from(format!("browser-entry-{i}")))
                        .w_full()
                        .min_w_0()
                        .gap_2()
                        .py_0p5()
                        .child(div().w(px(72.)).flex_none().text_xs().text_color(muted).child(label))
                        .child(div().flex_1().min_w_0().text_sm().truncate().child(name))
                        .child(div().w(px(72.)).flex_none().text_xs().text_color(muted).child(size))
                        .child(actions),
                );
            }
        }

        let refresh = {
            let m = me.clone();
            let dir = self.dir.clone();
            Button::new("browser-refresh").ghost().xsmall().label("Refresh").disabled(self.loading || self.server.is_none()).on_click(move |_, _, cx| {
                let _ = m.update(cx, |b, cx| b.load(dir.clone(), cx));
            })
        };
        v_flex()
            .id("server-browser")
            .w(px(620.))
            .max_h(gpui_kit::relative(0.85))
            .overflow_y_scroll()
            .gap_2()
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Open from server"))
            .child(if servers.is_empty() {
                div().text_sm().text_color(muted).child("No servers yet: add one in Settings (Ctrl+,).").into_any_element()
            } else {
                row("Server", picker, cx).into_any_element()
            })
            .child(
                h_flex()
                    .gap_2()
                    .border_b_1()
                    .border_color(border)
                    .pb_1()
                    .child(div().flex_1().min_w_0().text_xs().text_color(muted).truncate().child(format!("/{}", self.dir)))
                    .child(div().text_xs().text_color(muted).child(if self.loading { "Loading…" } else { "" }))
                    .child(refresh),
            )
            .child(entries)
            .children(self.message.clone().map(|(good, text)| div().text_xs().text_color(if good { ok } else { danger }).child(text)))
            .child(div().text_xs().text_color(muted).child("Open reads a recording on the server (views come from its pyramid); Fetch copies into the results folder (Settings)."))
    }
}
