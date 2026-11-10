//! The Analytics workspace: things aligned to events. Pick an event set (a crossings output, a
//! selection) and a window; every channel of the active view's signal is shown averaged around the
//! events (mean ± SD, [`aligned`]), and plugins add their own cards for the same events
//! (electrophysiology: each unit's raster and PETH, [`crate::plugin::AppPlugin::aligned`]).

pub mod aligned;

use std::sync::Arc;

use gpui_kit::component::button::Button;
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, img, px, AnyElement, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage, StatefulInteractiveElement as _, Styled as _,
    Subscription, Window,
};

use crate::data::store::{AppEvent, Store};
use crate::navigation::route::Workspace;
use crate::ui::widgets::EmptyState;
use aligned::{aligned_average, AlignTo, ChannelAverage};

/// Windows offered, (before, after) the event in milliseconds.
pub const WINDOWS_MS: [(f64, f64); 5] = [(50.0, 150.0), (100.0, 300.0), (200.0, 800.0), (500.0, 1500.0), (1000.0, 3000.0)];
/// Default: −200 … +800 ms (a contraction and its relaxation).
const DEFAULT_WINDOW: usize = 2;
/// Most channels averaged at once (aligning every channel of a 384-channel probe is a later step).
pub const MAX_CHANNELS: usize = 32;
/// Card plot size (logical pixels).
const CARD_W: f32 = 260.;
const CARD_H: f32 = 110.;

/// What the cards show, so a finished computation for older settings is dropped.
#[derive(Debug, Clone, PartialEq)]
struct Key {
    set: String,
    window: usize,
    align: AlignTo,
    view: String,
}

struct Card {
    avg: ChannelAverage,
    name: String,
    image: Option<Arc<RenderImage>>,
}

pub struct AnalyticsWorkspace {
    store: Entity<Store>,
    set: Option<String>,
    window: usize,
    align: AlignTo,
    /// The settings of the cards shown, and the cards.
    shown: Option<(Key, Vec<Card>)>,
    computing: Option<Key>,
    error: Option<String>,
    /// Unit of the aligned signal's values (for the card text).
    unit: String,
    _store: Subscription,
}

impl AnalyticsWorkspace {
    pub fn new(store: Entity<Store>, _: &mut Window, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_)) {
                this.refresh(cx);
            }
        });
        let mut this = Self { store, set: None, window: DEFAULT_WINDOW, align: AlignTo::OwnEvents, shown: None, computing: None, error: None, unit: String::new(), _store: sub };
        this.refresh(cx);
        this
    }

    /// The settings now (the first event set of the open recording when none is picked).
    fn key(&self, cx: &Context<Self>) -> Option<Key> {
        let s = self.store.read(cx);
        let open = s.recording.as_ref().and_then(|r| r.path.clone())?;
        let set = match &self.set {
            Some(id) if s.doc.event_sets.iter().any(|e| &e.id == id) => id.clone(),
            _ => s.doc.event_sets.iter().find(|e| e.recording == open)?.id.clone(),
        };
        let view = s.doc.active().map(|v| v.id.clone()).unwrap_or_default();
        Some(Key { set, window: self.window, align: self.align, view })
    }

    /// Computes the cards for the settings now, unless they are shown or on their way.
    fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(key) = self.key(cx) else {
            self.shown = None;
            cx.notify();
            return;
        };
        if self.shown.as_ref().is_some_and(|(k, _)| *k == key) || self.computing.as_ref() == Some(&key) {
            return;
        }
        let s = self.store.read(cx);
        let Some(set) = s.doc.event_sets.iter().find(|e| e.id == key.set).cloned() else { return };
        let Some(events) = s.events_of(&set) else {
            self.error = Some(format!("{}: its events could not be read", set.name));
            cx.notify();
            return;
        };
        let open = s.recording.as_ref().and_then(|r| r.path.clone());
        if open.as_ref() != Some(&set.recording) {
            self.error = Some(format!("{} is on another recording: open it to align its signal", set.name));
            cx.notify();
            return;
        }
        let Some(view) = s.doc.active().cloned() else { return };
        let source = s.view_source(&view);
        let Some(sources) = s.sources() else { return };
        let dataset = sources.get(&source);
        let signal = s.preview_of(&view.id).unwrap_or_else(|| dataset.signal().clone());
        let names: Vec<String> = signal.info().channels.iter().map(|c| c.name.clone()).collect();
        let unit = signal.info().channels.first().map(|c| c.unit.symbol().to_string()).unwrap_or_default();
        let mut channels: Vec<usize> = match key.align {
            AlignTo::OwnEvents => events.iter().map(|e| e.channel as usize).collect(),
            AlignTo::AllEvents => s.selection(&source).to_vec(),
        };
        channels.sort_unstable();
        channels.dedup();
        channels.retain(|&c| c < names.len());
        channels.truncate(MAX_CHANNELS);
        let palette = s.palette();
        let (pre_ms, post_ms) = WINDOWS_MS[key.window];
        let scale = 2.0;
        let (hue, align) = (set.hue, key.align);
        let task = cx.background_spawn(async move {
            let info = signal.info().clone();
            let read = |ch: usize, r: std::ops::Range<u64>, out: &mut [f32]| signal.read(&[ch], r, out);
            let avg = aligned_average(read, info.samples, info.sample_rate_hz(), &channels, &events, align, pre_ms * 1e-3, post_ms * 1e-3).map_err(|e| e.to_string())?;
            let rgb = crate::screens::explore::components::timeline_tab::tick_color(hue, 1.0).to_rgb();
            let colour = crate::ui::plot::canvas::Pixel::rgb((rgb.r * 255.0) as u8, (rgb.g * 255.0) as u8, (rgb.b * 255.0) as u8);
            let frac = (pre_ms / (pre_ms + post_ms)) as f32;
            Ok::<_, String>(
                avg.channels
                    .into_iter()
                    .map(|a| {
                        let frame = aligned::draw(&a, frac, (CARD_W * scale) as u32, (CARD_H * scale) as u32, palette.background, palette.grid, colour, scale);
                        let (w, h) = (frame.width, frame.height);
                        let image = image::RgbaImage::from_raw(w, h, frame.into_bgra()).map(|b| Arc::new(RenderImage::new(vec![image::Frame::new(b)])));
                        let name = names.get(a.channel).cloned().unwrap_or_default();
                        Card { avg: a, name, image }
                    })
                    .collect::<Vec<_>>(),
            )
        });
        self.computing = Some(key.clone());
        self.error = None;
        self.unit = unit;
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |w, cx| {
                if w.computing.as_ref() == Some(&key) {
                    w.computing = None;
                    match result {
                        Ok(cards) => w.shown = Some((key, cards)),
                        Err(e) => w.error = Some(e),
                    }
                    cx.notify();
                }
            });
        })
        .detach();
        cx.notify();
    }

    fn controls(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let open = s.recording.as_ref().and_then(|r| r.path.clone());
        let sets: Vec<(String, String)> = s.doc.event_sets.iter().filter(|e| Some(&e.recording) == open.as_ref()).map(|e| (e.id.clone(), e.name.clone())).collect();
        let current = self.key(cx).map(|k| k.set);
        let set_label = sets.iter().find(|(id, _)| Some(id) == current.as_ref()).map_or("Event set".to_string(), |(_, n)| n.clone());
        let this = cx.entity();
        let (this2, this3) = (this.clone(), this.clone());
        let (pre, post) = WINDOWS_MS[self.window];
        let align_label = match self.align {
            AlignTo::OwnEvents => "Each channel to its own events",
            AlignTo::AllEvents => "Selected channels to every event",
        };
        h_flex()
            .h(px(44.))
            .px_3()
            .gap_2()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child("Aligned to"))
            .child(Button::new("analytics-set").outline().small().label(set_label).dropdown_menu(move |mut m: PopupMenu, _, _| {
                for (id, name) in &sets {
                    let (this, id) = (this.clone(), id.clone());
                    m = m.item(PopupMenuItem::new(name.clone()).on_click(move |_, _, cx| {
                        this.update(cx, |w, cx| {
                            w.set = Some(id.clone());
                            w.refresh(cx);
                        })
                    }));
                }
                m
            }))
            .child(Button::new("analytics-window").outline().small().label(format!("−{pre:.0} … +{post:.0} ms")).dropdown_menu(move |mut m: PopupMenu, _, _| {
                for (i, (a, b)) in WINDOWS_MS.iter().enumerate() {
                    let this = this2.clone();
                    m = m.item(PopupMenuItem::new(format!("−{a:.0} … +{b:.0} ms")).on_click(move |_, _, cx| {
                        this.update(cx, |w, cx| {
                            w.window = i;
                            w.refresh(cx);
                        })
                    }));
                }
                m
            }))
            .child(Button::new("analytics-align").outline().small().label(align_label).dropdown_menu(move |m: PopupMenu, _, _| {
                let (a, b) = (this3.clone(), this3.clone());
                m.item(PopupMenuItem::new("Each channel to its own events").on_click(move |_, _, cx| {
                    a.update(cx, |w, cx| {
                        w.align = AlignTo::OwnEvents;
                        w.refresh(cx);
                    })
                }))
                .item(PopupMenuItem::new("Selected channels to every event").on_click(move |_, _, cx| {
                    b.update(cx, |w, cx| {
                        w.align = AlignTo::AllEvents;
                        w.refresh(cx);
                    })
                }))
            }))
            .child(div().flex_1())
            .child(div().text_xs().text_color(cx.theme().muted_foreground).child(if self.computing.is_some() {
                "Averaging…".to_string()
            } else {
                self.shown.as_ref().map_or(String::new(), |(_, c)| format!("{} channel{} · mean ± SD", c.len(), if c.len() == 1 { "" } else { "s" }))
            }))
            .into_any_element()
    }
}

impl Render for AnalyticsWorkspace {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let store = self.store.clone();
        let s = self.store.read(cx);
        if s.recording.is_none() || s.doc.event_sets.is_empty() {
            return div().id("analytics").test_support().size_full().child(EmptyState::new(
                Workspace::Analytics.icon(),
                "Nothing to align yet",
                "Open a recording and find events on it: in Pipeline, a chain ending with Crossings and a Write saves an event set. \
                 Its channels' average around the events, and each unit's raster and PETH, show here.",
            ));
        }
        let (muted, border) = (cx.theme().muted_foreground, cx.theme().border);
        let controls = self.controls(cx);
        let (pre, post) = WINDOWS_MS[self.window];
        let cards = self.shown.as_ref().map(|(_, cards)| {
            cards
                .iter()
                .map(|c| {
                    v_flex()
                        .w(px(CARD_W + 16.))
                        .p_2()
                        .gap_1()
                        .rounded_md()
                        .border_1()
                        .border_color(border)
                        .child(
                            h_flex()
                                .gap_2()
                                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(format!("Ch {}", c.avg.channel)))
                                .child(div().flex_1().min_w_0().truncate().text_xs().text_color(muted).child(c.name.clone()))
                                .child(div().text_xs().text_color(muted).child(format!("{} events", c.avg.trials))),
                        )
                        .children(c.image.clone().map(|i| img(i).w(px(CARD_W)).h(px(CARD_H))))
                        .child(h_flex().text_xs().text_color(muted).child(format!("−{pre:.0} ms")).child(div().flex_1()).child("0").child(div().flex_1()).child(format!("+{post:.0} ms")))
                        .child(div().text_xs().text_color(muted).child({
                            let (lo, hi) = c.avg.mean.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                            if lo.is_finite() { format!("mean {lo:.3} … {hi:.3} {}", self.unit) } else { "no events with room around them".to_string() }
                        }))
                        .into_any_element()
                })
                .collect::<Vec<_>>()
        });
        // What plugins align to the same events (units: raster and PETH)
        let events: Option<Vec<f64>> = self.key(cx).and_then(|k| {
            let s = self.store.read(cx);
            let set = s.doc.event_sets.iter().find(|e| e.id == k.set)?;
            Some(s.events_of(set)?.iter().map(|e| e.onset_s).collect())
        });
        let extra: Vec<AnyElement> = events.map_or_else(Vec::new, |ev| crate::plugin::plugins().iter().filter_map(|p| p.aligned(&store, &ev, pre, post, window, cx)).collect());
        v_flex()
            .id("analytics")
            .test_support()
            .size_full()
            .child(controls)
            .child(
                v_flex()
                    .id("analytics-body")
                    .flex_1()
                    .overflow_y_scroll()
                    .p_3()
                    .gap_3()
                    .when_some(self.error.clone(), |d, e| d.child(div().text_sm().text_color(muted).child(e)))
                    .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child("SIGNAL"))
                    .child(h_flex().flex_wrap().gap_3().children(cards.unwrap_or_default()))
                    .children(extra),
            )
    }
}
