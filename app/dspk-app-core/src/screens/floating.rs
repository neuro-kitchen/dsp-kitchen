//! Floating windows over every workspace: a recording's metadata (and where it sits on the
//! session's time), and the epoch editor (name, type, start, end, notes; save or delete). Each
//! is a card with a title bar to drag it by and a close button; the store keeps which are open.

use std::collections::HashMap;
use std::path::PathBuf;

use dspk_io::SourceEntry;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, DragMoveEvent, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Task, Window,
};

use crate::data::store::{duration, file_label, AppEvent, Floating, Store};
use crate::domain::command::SessionCommand;
use crate::domain::epoch::{EpochDoc, EPOCH_KINDS};
use crate::ui::widgets::{icon_button, row, MenuSelect};

/// Card width (logical px).
const CARD_W: f32 = 380.;

/// A card being dragged by its title bar: which, and where in the bar it was grabbed.
#[derive(Debug, Clone)]
pub struct CardDrag(usize);

impl Render for CardDrag {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

pub struct FloatingLayer {
    store: Entity<Store>,
    /// Card positions (window px) by index, and the grab offset of a drag.
    positions: HashMap<usize, (f32, f32)>,
    grab: (f32, f32),
    /// The epoch editor's inputs and the epoch they hold.
    name: Entity<InputState>,
    start: Entity<InputState>,
    end: Entity<InputState>,
    notes: Entity<InputState>,
    kind: usize,
    epoch_for: Option<EpochDoc>,
    error: Option<String>,
    /// Sources of recordings that are not open (read in the background), and the start input.
    sources: HashMap<PathBuf, Result<Vec<SourceEntry>, String>>,
    place: Entity<InputState>,
    place_for: Option<PathBuf>,
    _work: Vec<Task<()>>,
    _subs: Vec<Subscription>,
}

impl FloatingLayer {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut input = |cx: &mut Context<Self>, ph: &str| {
            let ph = ph.to_string();
            cx.new(|cx| InputState::new(window, cx).placeholder(ph))
        };
        let (name, start, end, notes, place) = (input(cx, "e.g. tone 1"), input(cx, "s"), input(cx, "s"), input(cx, "Anything to remember"), input(cx, "s"));
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Floating | AppEvent::Document(_) | AppEvent::RecordingChanged) {
                cx.notify();
            }
        })];
        Self {
            store,
            positions: HashMap::new(),
            grab: (0.0, 0.0),
            name,
            start,
            end,
            notes,
            kind: 0,
            epoch_for: None,
            error: None,
            sources: HashMap::new(),
            place,
            place_for: None,
            _work: Vec::new(),
            _subs: subs,
        }
    }

    fn close(&mut self, index: usize, cx: &mut Context<Self>) {
        self.positions.remove(&index);
        self.store.update(cx, |s, cx| s.close_floating(index, cx));
    }

    /// The editor's values as an epoch (`base`'s recording and id).
    fn epoch_from_inputs(&self, base: &EpochDoc, cx: &App) -> Result<EpochDoc, String> {
        let num = |i: &Entity<InputState>, what: &str| i.read(cx).value().trim().parse::<f64>().map_err(|_| format!("{what}: not a number of seconds"));
        Ok(EpochDoc {
            name: self.name.read(cx).value().trim().to_string(),
            kind: EPOCH_KINDS[self.kind].0.to_string(),
            start_sec: num(&self.start, "Start")?,
            end_sec: num(&self.end, "End")?,
            notes: self.notes.read(cx).value().trim().to_string(),
            ..base.clone()
        })
    }

    fn save_epoch(&mut self, index: usize, base: &EpochDoc, cx: &mut Context<Self>) {
        match self.epoch_from_inputs(base, cx) {
            Ok(epoch) => match self.store.update(cx, |s, cx| s.execute(SessionCommand::SetEpoch { epoch }, cx)) {
                Ok(_) => {
                    self.error = None;
                    self.epoch_for = None;
                    self.close(index, cx);
                }
                Err(e) => self.error = Some(e.to_string()),
            },
            Err(e) => self.error = Some(e),
        }
        cx.notify();
    }

    fn epoch_card(&mut self, index: usize, epoch: &EpochDoc, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        if self.epoch_for.as_ref() != Some(epoch) {
            self.epoch_for = Some(epoch.clone());
            self.error = None;
            self.kind = EPOCH_KINDS.iter().position(|(k, _)| *k == epoch.kind).unwrap_or(EPOCH_KINDS.len() - 1);
            let (n, a, b, notes) = (epoch.name.clone(), format!("{:.3}", epoch.start_sec), format!("{:.3}", epoch.end_sec), epoch.notes.clone());
            self.name.update(cx, |i, cx| i.set_value(n, window, cx));
            self.start.update(cx, |i, cx| i.set_value(a, window, cx));
            self.end.update(cx, |i, cx| i.set_value(b, window, cx));
            self.notes.update(cx, |i, cx| i.set_value(notes, window, cx));
        }
        let muted = cx.theme().muted_foreground;
        let me = cx.entity().downgrade();
        let kinds = MenuSelect::new("epoch-kind", EPOCH_KINDS.iter().map(|(k, _)| SharedString::from(*k)).collect(), Some(self.kind), {
            let me = me.clone();
            move |i, _, cx| {
                let _ = me.update(cx, |l, cx| {
                    l.kind = i;
                    cx.notify();
                });
            }
        });
        let (base, base2) = (epoch.clone(), epoch.clone());
        let saved = self.store.read(cx).doc.epochs.iter().any(|e| e.id == epoch.id && !epoch.id.is_empty());
        v_flex()
            .gap_2()
            .child(div().text_xs().text_color(muted).truncate().child(file_label(&epoch.recording)))
            .child(row("Name", div().w_full().child(Input::new(&self.name).small()), cx))
            .child(row("Type", kinds, cx))
            .child(row("Start (s)", div().w_full().child(Input::new(&self.start).small()), cx))
            .child(row("End (s)", div().w_full().child(Input::new(&self.end).small()), cx))
            .child(row("Notes", div().w_full().child(Input::new(&self.notes).small()), cx))
            .children(self.error.clone().map(|e| div().text_xs().text_color(cx.theme().danger).child(e)))
            .child(div().text_xs().text_color(muted).child("Saved in the session and in the recording's .dspk.zarr store (epochs group)."))
            .child(
                h_flex()
                    .gap_1()
                    .child({
                        let me = me.clone();
                        Button::new("epoch-save").primary().small().label("Save").on_click(move |_, _, cx| {
                            let _ = me.update(cx, |l, cx| l.save_epoch(index, &base, cx));
                        })
                    })
                    .when(saved, |d| {
                        let (me, id) = (me.clone(), base2.id.clone());
                        d.child(Button::new("epoch-delete").outline().small().label("Delete").on_click(move |_, _, cx| {
                            let _ = me.update(cx, |l, cx| {
                                let _ = l.store.update(cx, |s, cx| s.execute(SessionCommand::RemoveEpoch { id: id.clone() }, cx));
                                l.close(index, cx);
                            });
                        }))
                    }),
            )
            .into_any_element()
    }

    fn metadata_card(&mut self, path: &PathBuf, source: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let muted = cx.theme().muted_foreground;
        let s = self.store.read(cx);
        let open = s.recording.as_ref().is_some_and(|r| r.path.as_deref() == Some(path.as_path()));
        let entries: Result<Vec<SourceEntry>, String> = if open {
            Ok(s.sources().map(|ss| ss.entries().to_vec()).unwrap_or_default())
        } else {
            match self.sources.get(path) {
                Some(r) => r.clone(),
                None => {
                    // Read the file's sources once, off the UI thread
                    self.sources.insert(path.clone(), Err("Reading…".into()));
                    let p = path.clone();
                    let work = cx.background_spawn(async move { crate::plugin::sources(&p).map_err(|e| e.to_string()) });
                    let p = path.clone();
                    self._work.push(cx.spawn(async move |this, cx| {
                        let r = work.await;
                        let _ = this.update(cx, |l, cx| {
                            l.sources.insert(p, r);
                            cx.notify();
                        });
                    }));
                    Err("Reading…".into())
                }
            }
        };
        let layout = s.sources().filter(|_| open).and_then(|ss| ss.get(source).layout.clone());
        let placed = s.doc.placements.iter().find(|p| p.path == *path).map(|p| p.start_sec);
        let epochs = s.doc.epochs.iter().filter(|e| e.recording == *path).count();
        if self.place_for.as_ref() != Some(path) {
            self.place_for = Some(path.clone());
            let v = placed.map(|t| format!("{t:.3}")).unwrap_or_default();
            self.place.update(cx, |i, cx| i.set_value(v, window, cx));
        }
        let size = std::fs::metadata(path).map(|m| m.len()).ok();
        let fact = |k: &'static str, v: String| h_flex().gap_2().child(div().w(px(110.)).text_xs().text_color(muted).child(k)).child(div().flex_1().min_w_0().text_xs().child(v));
        let mut body = v_flex().gap_1().child(fact("File", path.display().to_string()));
        if let Some(b) = size {
            body = body.child(fact("Size", format!("{:.1} MB", b as f64 / 1e6)));
        }
        match &entries {
            Ok(list) => {
                for e in list {
                    let mark = if e.id == source { " (shown)" } else { "" };
                    body = body
                        .child(div().pt_1().text_xs().font_weight(FontWeight::SEMIBOLD).child(format!("{}{mark}", e.name)))
                        .child(fact("Channels", e.channels.to_string()))
                        .child(fact("Rate", format!("{} Hz", e.sample_rate.rate_hz())))
                        .child(fact("Length", format!("{} ({} samples)", duration(e.duration_sec()), e.samples)))
                        .child(fact("Starts at", format!("{:.6} s", e.start_time.as_duration().as_secs_f64())))
                        .child(fact("Stored as", e.format.name().to_string()))
                        .child(fact("Unit", e.unit.symbol().to_string()));
                }
            }
            Err(e) => body = body.child(div().text_xs().text_color(muted).child(e.clone())),
        }
        if let Some(l) = layout {
            body = body.child(fact("Layout", format!("{} {}s · {} labelled channels", l.groups.len(), l.group_kind.to_lowercase(), l.labels.iter().flatten().count())));
        }
        body = body.child(fact("Epochs", epochs.to_string()));
        let (store, p, src) = (self.store.clone(), path.clone(), source.to_string());
        let place = self.place.clone();
        body.child(
            h_flex()
                .pt_2()
                .gap_2()
                .child(div().text_xs().text_color(muted).child("On the session's time at (s)"))
                .child(div().w(px(90.)).child(Input::new(&self.place).small()))
                .child(Button::new("metadata-place").outline().xsmall().label("Place").on_click(move |_, _, cx| {
                    let text = place.read(cx).value().trim().to_string();
                    match text.parse::<f64>() {
                        Ok(t) => {
                            let source = Some(src.clone()).filter(|s| !s.is_empty());
                            let _ = store.update(cx, |s, cx| s.execute(SessionCommand::PlaceRecording { path: p.clone(), source, start_sec: t }, cx));
                        }
                        Err(_) => store.update(cx, |s, cx| s.set_status(format!("{text:?} is not a number of seconds"), cx)),
                    }
                })),
        )
        .into_any_element()
    }
}

impl Render for FloatingLayer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let items = self.store.read(cx).floating.clone();
        let t = cx.theme().clone();
        let mut cards = Vec::new();
        for (i, f) in items.iter().enumerate() {
            let (title, body) = match f {
                Floating::Epoch(e) => (if e.id.is_empty() { "New epoch".to_string() } else { format!("Epoch · {}", e.name) }, self.epoch_card(i, e, window, cx)),
                Floating::Metadata { path, source } => (format!("{} · metadata", file_label(path)), self.metadata_card(path, source, window, cx)),
            };
            let (x, y) = self.positions.get(&i).copied().unwrap_or((window.viewport_size().width.as_f32() / 2.0 - CARD_W / 2.0 + i as f32 * 28.0, 120.0 + i as f32 * 28.0));
            let me = cx.entity().downgrade();
            let close = {
                let me = me.clone();
                icon_button(SharedString::from(format!("floating-close-{i}")), IconName::Close, "Close").xsmall().on_click(move |_, _, cx| {
                    let _ = me.update(cx, |l, cx| l.close(i, cx));
                })
            };
            let grab_me = me.clone();
            cards.push(
                v_flex()
                    .id(SharedString::from(format!("floating-{i}")))
                    .occlude()
                    .absolute()
                    .left(px(x))
                    .top(px(y))
                    .w(px(CARD_W))
                    .max_h(px(560.))
                    .rounded_lg()
                    .border_1()
                    .border_color(t.border)
                    .bg(t.popover)
                    .shadow_lg()
                    .child(
                        h_flex()
                            .id(SharedString::from(format!("floating-title-{i}")))
                            .h(px(34.))
                            .px_3()
                            .gap_2()
                            .cursor_grab()
                            .border_b_1()
                            .border_color(t.border)
                            .child(div().flex_1().min_w_0().truncate().text_sm().font_weight(FontWeight::SEMIBOLD).child(title))
                            .child(close)
                            .on_drag(CardDrag(i), move |d: &CardDrag, offset, _, cx| {
                                let o = (offset.x.as_f32(), offset.y.as_f32());
                                let _ = grab_me.update(cx, |l, _| l.grab = o);
                                cx.new(|_| d.clone())
                            }),
                    )
                    .child(div().id(SharedString::from(format!("floating-body-{i}"))).p_3().overflow_y_scroll().child(body)),
            );
        }
        div()
            .id("floating-layer")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .on_drag_move(cx.listener(|this, e: &DragMoveEvent<CardDrag>, _, cx| {
                let i = e.drag(cx).0;
                let p = e.event.position;
                this.positions.insert(i, (p.x.as_f32() - this.grab.0, (p.y.as_f32() - this.grab.1).max(0.0)));
                cx.notify();
            }))
            .children(cards)
    }
}
