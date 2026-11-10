//! An Epochs view (a session view with the Epochs layout, from the Pipeline's Output menu) shown
//! in Explore: every event's window of the view's signal stacked per channel (faint), with the
//! mean ± SD on top and a line at the event. Its events (the view's `epochs_from`) are the live
//! ones of the visible window (the chain ends with Crossings: they follow the level while it is
//! tuned) or the set a Write of the same chain saved. Its window and source are the view's
//! display settings (saved in the session). Computed off the UI thread; redrawn when the chain,
//! the level, the window or a setting changes.

use std::sync::Arc;

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::Button;
use gpui_kit::component::dock::Panel;
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::events::Event;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::domain::session::EpochsFrom;
use crate::screens::analytics::aligned::{self, aligned_average, AlignTo, ChannelAverage};
use crate::screens::analytics::WINDOWS_MS;
use crate::screens::explore::components::timeline_tab::tick_color;

/// Snippets drawn under the mean, per channel.
const SNIPPETS: usize = 60;
const CARD_W: f32 = 240.;
const CARD_H: f32 = 120.;
/// Tick hue of live events (the ruler's).
const LIVE_HUE: f32 = 45.0;

/// Where the events of one computation come from.
#[derive(Debug, Clone, PartialEq)]
enum Source {
    /// Found now in the visible window by the Crossings op the chain ends with.
    Live,
    /// A saved event set (by id).
    Set(String),
}

/// What the cards show (a finished computation for other settings is dropped).
#[derive(Debug, Clone, PartialEq)]
struct Key {
    source: Source,
    window: usize,
    /// The active view's chain, the window's samples (live) and the signal read.
    chain: String,
    span: (u64, u64),
    signal: usize,
}

struct Card {
    avg: ChannelAverage,
    image: Option<Arc<RenderImage>>,
}

pub struct EpochsPanel {
    store: Entity<Store>,
    /// The session view (Epochs layout) shown.
    pub view: String,
    shown: Option<(Key, Vec<Card>, usize)>,
    computing: Option<Key>,
    note: Option<String>,
    focus: FocusHandle,
    _sub: Subscription,
}

impl EpochsPanel {
    pub fn new(store: Entity<Store>, view: String, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::WindowMoved | AppEvent::RecordingChanged) {
                this.refresh(cx);
            }
        });
        let mut this = Self { store, view, shown: None, computing: None, note: None, focus: cx.focus_handle(), _sub: sub };
        this.refresh(cx);
        this
    }

    /// Where the events come from now: the setting, the saved set falling back to live in Auto.
    fn source(&self, cx: &App) -> Result<Source, String> {
        let s = self.store.read(cx);
        let view = s.doc.view(&self.view).ok_or("This Epochs view is no longer in the session.")?;
        let saved = s.saved_events_of(view).map(|set| set.id.clone());
        match (view.display.epochs_from, saved) {
            (EpochsFrom::Live, _) | (EpochsFrom::Auto, None) => Ok(Source::Live),
            (_, Some(id)) => Ok(Source::Set(id)),
            (EpochsFrom::Saved, None) => Err("No saved events yet: double-click the Write after Crossings in Pipeline (or switch to Live).".into()),
        }
    }

    /// Changes the view's display settings (saved in the session).
    fn set_display(&self, f: impl FnOnce(&mut crate::domain::session::DisplayDoc), cx: &mut App) {
        let Some(mut display) = self.store.read(cx).doc.view(&self.view).map(|v| v.display.clone()) else { return };
        f(&mut display);
        let cmd = SessionCommand::SetDisplay { view: self.view.clone(), display };
        self.store.update(cx, |s, cx| {
            let _ = s.execute(cmd, cx);
        });
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        let source = match self.source(cx) {
            Ok(source) => source,
            Err(e) => {
                (self.note, self.shown) = (Some(e), None);
                cx.notify();
                return;
            }
        };
        let s = self.store.read(cx);
        let Some(view) = s.doc.view(&self.view).cloned() else { return };
        if s.recording.as_ref().and_then(|r| r.path.as_ref()) != Some(&view.recording.path) {
            (self.note, self.shown) = (Some("Its recording isn't the open one: open it to see its epochs.".into()), None);
            cx.notify();
            return;
        }
        let Some(sources) = s.sources() else { return };
        let dataset = sources.get(&s.view_source(&view));
        let signal = s.preview_of(&view.id).unwrap_or_else(|| dataset.signal().clone());
        let rate = signal.info().sample_rate_hz();
        let (t0, win) = (s.timeline.window_start_sec.max(0.0), s.timeline.visible_window_sec);
        let span = ((t0 * rate) as u64, ((t0 + win) * rate).ceil() as u64);
        let window = view.display.epochs_window.min(WINDOWS_MS.len() - 1);
        let (pre_ms, post_ms) = WINDOWS_MS[window];
        let key = Key {
            source: source.clone(),
            window,
            chain: serde_json::to_string(&view.ops).unwrap_or_default(),
            span: if source == Source::Live { span } else { (0, 0) },
            signal: Arc::as_ptr(&signal) as *const () as usize,
        };
        if self.shown.as_ref().is_some_and(|(k, _, _)| *k == key) || self.computing.as_ref() == Some(&key) {
            return;
        }
        // The events: live ones are found with the averaging, in the background
        let (hue, live, saved): (f32, Option<dspk_pipeline::Node>, Option<Arc<Vec<Event>>>) = match &source {
            Source::Live => match crate::data::live::crossings_of(&view) {
                Some(n) => (LIVE_HUE, Some(n.clone()), None),
                None => {
                    self.note = Some("Its chain doesn't end with Crossings: add Crossings before this output in Pipeline.".into());
                    self.shown = None;
                    cx.notify();
                    return;
                }
            },
            Source::Set(id) => match s.doc.event_sets.iter().find(|e| &e.id == id) {
                Some(set) => (set.hue, None, s.events_of(set)),
                None => return,
            },
        };
        let selection: Vec<usize> = s.selection(&s.view_source(&view)).to_vec();
        let palette = s.palette();
        let scale = 2.0;
        let task = cx.background_spawn(async move {
            let events: Arc<Vec<Event>> = match (live, saved) {
                (Some(node), _) => Arc::new(crate::data::live::detect(&signal, &selection, span.0..span.1, &node).map_err(|e| e.to_string())?),
                (None, Some(ev)) => ev,
                _ => return Err("the events could not be read".to_string()),
            };
            let mut channels: Vec<usize> = events.iter().map(|e| e.channel as usize).collect();
            channels.sort_unstable();
            channels.dedup();
            channels.truncate(crate::screens::analytics::MAX_CHANNELS);
            let info = signal.info().clone();
            let read = |ch: usize, r: std::ops::Range<u64>, out: &mut [f32]| signal.read(&[ch], r, out);
            let avg = aligned_average(read, info.samples, info.sample_rate_hz(), &channels, &events, AlignTo::OwnEvents, pre_ms * 1e-3, post_ms * 1e-3, SNIPPETS).map_err(|e| e.to_string())?;
            let rgb = tick_color(hue, 1.0).to_rgb();
            let colour = crate::ui::plot::canvas::Pixel::rgb((rgb.r * 255.0) as u8, (rgb.g * 255.0) as u8, (rgb.b * 255.0) as u8);
            let frac = (pre_ms / (pre_ms + post_ms)) as f32;
            let cards = avg
                .channels
                .into_iter()
                .map(|a| {
                    let frame = aligned::draw(&a, frac, (CARD_W * scale) as u32, (CARD_H * scale) as u32, palette.background, palette.grid, colour, scale);
                    let (w, h) = (frame.width, frame.height);
                    let image = image::RgbaImage::from_raw(w, h, frame.into_bgra()).map(|b| Arc::new(RenderImage::new(vec![image::Frame::new(b)])));
                    Card { avg: a, image }
                })
                .collect::<Vec<_>>();
            Ok((cards, events.len()))
        });
        self.computing = Some(key.clone());
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |p, cx| {
                if p.computing.as_ref() != Some(&key) {
                    return;
                }
                p.computing = None;
                match result {
                    Ok((cards, n)) => {
                        p.note = (n == 0).then(|| "No events here: move the window, or lower the level.".to_string());
                        p.shown = Some((key, cards, n));
                    }
                    Err(e) => {
                        p.note = Some(e);
                        p.shown = None;
                    }
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

}

impl Render for EpochsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, border) = (t.muted_foreground, t.border);
        let s = self.store.read(cx);
        let display = s.doc.view(&self.view).map(|v| v.display.clone()).unwrap_or_default();
        let saved = s.doc.view(&self.view).and_then(|v| s.saved_events_of(v)).map(|set| set.name.clone());
        let source_label = match (display.epochs_from, &saved) {
            (EpochsFrom::Live, _) | (EpochsFrom::Auto, None) => "Live: this window".to_string(),
            (_, Some(name)) => format!("Saved: {name}"),
            (EpochsFrom::Saved, None) => "Saved (none yet)".to_string(),
        };
        let window = display.epochs_window.min(WINDOWS_MS.len() - 1);
        let (pre, post) = WINDOWS_MS[window];
        let me = cx.entity();
        let me2 = me.clone();
        let status = if self.computing.is_some() {
            "Finding and stacking…".to_string()
        } else {
            self.shown.as_ref().map_or(String::new(), |(_, c, n)| format!("{n} event{} · {} channel{}", if *n == 1 { "" } else { "s" }, c.len(), if c.len() == 1 { "" } else { "s" }))
        };
        let header = h_flex()
            .h(px(36.))
            .px_2()
            .gap_2()
            .border_b_1()
            .border_color(border)
            .child(Button::new("epochs-source").outline().small().label(source_label).dropdown_menu(move |m: PopupMenu, _, _| {
                let pick = |label: &str, from: EpochsFrom| {
                    let me = me.clone();
                    PopupMenuItem::new(label.to_string()).on_click(move |_, _, cx| me.update(cx, |p, cx| p.set_display(|d| d.epochs_from = from, cx)))
                };
                m.item(pick("Auto: saved set, else live", EpochsFrom::Auto)).item(pick("Live: this window", EpochsFrom::Live)).item(pick("Saved: the Write's event set", EpochsFrom::Saved))
            }))
            .child(Button::new("epochs-window").outline().small().label(format!("−{pre:.0} … +{post:.0} ms")).dropdown_menu(move |mut m: PopupMenu, _, _| {
                for (i, (a, b)) in WINDOWS_MS.iter().enumerate() {
                    let me = me2.clone();
                    m = m.item(PopupMenuItem::new(format!("−{a:.0} … +{b:.0} ms")).on_click(move |_, _, cx| me.update(cx, |p, cx| p.set_display(|d| d.epochs_window = i, cx))));
                }
                m
            }))
            .child(div().flex_1())
            .child(div().text_xs().text_color(muted).child(status));
        let unit = s.doc.view(&self.view).and_then(|v| s.sources().map(|ss| ss.entry(&s.view_source(v)).unit.symbol().to_string())).unwrap_or_default();
        let cards = self.shown.as_ref().map(|(_, cards, _)| {
            cards
                .iter()
                .map(|c| {
                    let (lo, hi) = c.avg.mean.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                    v_flex()
                        .w(px(CARD_W + 12.))
                        .p_1p5()
                        .gap_1()
                        .rounded_md()
                        .border_1()
                        .border_color(border)
                        .child(
                            h_flex()
                                .gap_2()
                                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(format!("Ch {}", c.avg.channel)))
                                .child(div().flex_1())
                                .child(div().text_xs().text_color(muted).child(format!("{} events", c.avg.trials))),
                        )
                        .children(c.image.clone().map(|i| img(i).w(px(CARD_W)).h(px(CARD_H))))
                        .child(h_flex().text_xs().text_color(muted).child(format!("−{pre:.0} ms")).child(div().flex_1()).child("0").child(div().flex_1()).child(format!("+{post:.0} ms")))
                        .when(lo.is_finite(), |d| d.child(div().text_xs().text_color(muted).child(format!("mean {lo:.3} … {hi:.3} {unit}"))))
                        .into_any_element()
                })
                .collect::<Vec<_>>()
        });
        v_flex()
            .size_full()
            .track_focus(&self.focus)
            .child(header)
            .child(
                v_flex()
                    .id("epochs-body")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .p_2()
                    .gap_2()
                    .when_some(self.note.clone(), |d, n| d.child(div().text_sm().text_color(muted).child(n)))
                    .child(h_flex().flex_wrap().gap_2().children(cards.unwrap_or_default())),
            )
    }
}

impl EventEmitter<PanelEvent> for EpochsPanel {}

impl Focusable for EpochsPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for EpochsPanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.epochs"
    }
}

impl Panel for EpochsPanel {
    fn title(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.store.read(cx).doc.view(&self.view).map_or_else(|| "Epochs".to_string(), |v| v.name.clone())
    }
}
