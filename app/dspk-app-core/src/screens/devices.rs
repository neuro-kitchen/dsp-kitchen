//! Devices (File › Devices…): connect to a dspk device (a profile as a server's: its address, its
//! certificate name, this app's identity from `dspk pair`), see its state, take control, send a
//! chain, start a recording or a preview, mark it, stop it, and watch a tap live.
//!
//! The live strip draws the newest min / max columns of the chosen tap (a few channels); following
//! the head in Explore's plots, and scrolling back, build on the same data next.

#![cfg(feature = "device")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dspk_stream::live::wire;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Sizable as _};
use gpui_kit::{div, px, Context, Entity, FontWeight, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Task, Window, InteractiveElement as _};

use crate::data::instrument::{Instrument, LiveColumns};
use crate::data::remote::ServerProfile;
use crate::data::store::Store;
use crate::ui::recording_picker::Pick;
use crate::ui::widgets::{row, MenuSelect};

/// How often the screen redraws while connected (status and the live strip).
const REDRAW: Duration = Duration::from_millis(100);
/// Channels drawn in the live strip, and columns.
const STRIP_CHANNELS: usize = 4;
const STRIP_COLUMNS: usize = 240;
const STRIP_HEIGHT: f32 = 48.0;
/// Columns per second asked of the device for the strip.
const STRIP_COLUMNS_PER_SECOND: f64 = 60.0;

pub struct DevicesPanel {
    store: Entity<Store>,
    profile: Option<usize>,
    instrument: Option<Arc<Instrument>>,
    tap: Option<String>,
    live: Option<Arc<Mutex<LiveColumns>>>,
    message: Option<(bool, String)>,
    busy: bool,
    _tasks: Vec<Task<()>>,
}

impl DevicesPanel {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let first = (!store.read(cx).session.servers.is_empty()).then_some(0);
        Self { store, profile: first, instrument: None, tap: None, live: None, message: None, busy: false, _tasks: Vec::new() }
    }

    fn chosen(&self, cx: &Context<Self>) -> Option<ServerProfile> {
        self.profile.and_then(|i| self.store.read(cx).session.servers.get(i).cloned())
    }

    /// Runs `work` off the UI thread, then shows its result.
    fn run(&mut self, what: &'static str, work: impl FnOnce() -> anyhow::Result<String> + Send + 'static, cx: &mut Context<Self>) {
        self.busy = true;
        self.message = Some((true, format!("{what}…")));
        let task = cx.background_executor().spawn(async move { work() });
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |p, cx| {
                p.busy = false;
                p.message = Some(match result {
                    Ok(done) => (true, done),
                    Err(e) => (false, format!("{what}: {e:#}")),
                });
                cx.notify();
            });
        }));
        cx.notify();
    }

    fn connect(&mut self, cx: &mut Context<Self>) {
        let Some(profile) = self.chosen(cx) else { return };
        self.busy = true;
        self.message = Some((true, format!("Connecting to {}…", profile.name)));
        let task = cx.background_executor().spawn(async move { Instrument::connect(profile, Arc::new(|| {})) });
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |p, cx| {
                p.busy = false;
                match result {
                    Ok(instrument) => {
                        let name = instrument.welcome().device.as_ref().map(|d| d.name.clone()).unwrap_or_default();
                        p.message = Some((true, format!("Connected to {name} as {}", instrument.welcome().client_name)));
                        p.instrument = Some(Arc::new(instrument));
                        p.redraw_while_connected(cx);
                    }
                    Err(e) => p.message = Some((false, format!("{e:#}"))),
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Redraws every [`REDRAW`] while connected: the status and the live strip follow the device.
    fn redraw_while_connected(&mut self, cx: &mut Context<Self>) {
        self._tasks.push(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(REDRAW).await;
                let connected = this.update(cx, |p, cx| {
                    cx.notify();
                    p.instrument.as_ref().is_some_and(|i| i.client().is_open())
                });
                if !matches!(connected, Ok(true)) {
                    break;
                }
            }
        }));
    }

    fn watch(&mut self, tap: String, cx: &mut Context<Self>) {
        let Some(instrument) = self.instrument.clone() else { return };
        let channels: Vec<u32> = (0..STRIP_CHANNELS as u32).collect();
        let t = tap.clone();
        let task = cx.background_executor().spawn(async move {
            let taps = instrument.block_on(instrument.client().list_taps())?;
            let have = taps.iter().find(|i| i.name == t).map_or(0, |i| i.channels.len());
            let picked: Vec<u32> = channels.into_iter().filter(|&c| (c as usize) < have).collect();
            instrument.live_view(&t, 1, &picked, STRIP_COLUMNS_PER_SECOND)
        });
        self._tasks.push(cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |p, cx| {
                match result {
                    Ok(live) => {
                        p.tap = Some(tap);
                        p.live = Some(live);
                    }
                    Err(e) => p.message = Some((false, format!("Watching: {e:#}"))),
                }
                cx.notify();
            });
        }));
    }

    fn configure(&mut self, cx: &mut Context<Self>) {
        let Some(instrument) = self.instrument.clone() else { return };
        let rx = cx.prompt_for_paths(Pick::File.options("Send a chain (a pipeline spec)", false, cx));
        self._tasks.push(cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(path) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |p, cx| {
                p.run(
                    "Sending the chain",
                    move || {
                        let text = std::fs::read_to_string(&path)?;
                        instrument.request_control()?;
                        let refused = instrument.configure(&text)?;
                        if refused.is_empty() {
                            Ok(format!("Chain from {} configured", path.display()))
                        } else {
                            let why: Vec<String> = refused.iter().map(|r| format!("{} ({}): {}", if r.node.is_empty() { "spec" } else { &r.node }, r.op, r.reason)).collect();
                            anyhow::bail!("refused: {}", why.join("; "))
                        }
                    },
                    cx,
                );
            });
        }));
    }

    fn control(&mut self, what: &'static str, cx: &mut Context<Self>, act: impl FnOnce(&Instrument) -> anyhow::Result<()> + Send + 'static) {
        let Some(instrument) = self.instrument.clone() else { return };
        self.run(
            what,
            move || {
                instrument.request_control()?;
                act(&instrument)?;
                Ok(format!("{what}: done"))
            },
            cx,
        );
    }
}

/// `stopped`, `running`, … (the protocol's names without their prefix).
fn state_name(state: i32) -> String {
    wire::DeviceState::try_from(state).map(|s| s.as_str_name().trim_start_matches("DEVICE_STATE_").to_ascii_lowercase()).unwrap_or_else(|_| "?".into())
}

impl Render for DevicesPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, danger, ok, border, accent) = (t.muted_foreground, t.danger, t.success, t.border, t.primary);
        let me = cx.entity().downgrade();
        let profiles = self.store.read(cx).session.servers.clone();
        let names: Vec<SharedString> = profiles.iter().map(|s| SharedString::from(format!("{} ({})", s.name, s.address))).collect();
        let m = me.clone();
        let picker = MenuSelect::new("devices-profile", names, self.profile, move |i, _, cx| {
            let _ = m.update(cx, |p, cx| {
                p.profile = Some(i);
                p.instrument = None;
                p.live = None;
                cx.notify();
            });
        });
        let connect = {
            let m = me.clone();
            Button::new("devices-connect").primary().xsmall().label(if self.instrument.is_some() { "Reconnect" } else { "Connect" }).disabled(self.busy || self.profile.is_none()).on_click(move |_, _, cx| {
                let _ = m.update(cx, |p, cx| p.connect(cx));
            })
        };

        let mut body = v_flex().gap_1();
        if let Some(instrument) = &self.instrument {
            let s = instrument.status();
            let director = if s.director.is_empty() { "none".to_string() } else if instrument.is_director() { format!("{} (you)", s.director) } else { s.director.clone() };
            body = body.child(row("State", div().text_sm().child(format!("{}{}", state_name(s.state), if s.message.is_empty() { String::new() } else { format!(" — {}", s.message) }))));
            body = body.child(row("Director", div().text_sm().child(director)));
            if let Some(run) = &s.run {
                let what = if run.recording { format!("recording {}", run.path) } else { "preview".into() };
                body = body.child(row("Run", div().text_sm().child(format!("{what}: {} samples{}, {} lost", run.head_sample, if run.running { " so far" } else { "" }, run.lost_samples))));
            }
            for src in &s.sources {
                let health = if src.connected { format!("{:.1} Hz", src.measured_rate_hz) } else { "silent".into() };
                body = body.child(row("Source", div().text_sm().child(format!("{}: {health}, {} lost, {} checksum errors", src.name, src.lost_samples, src.checksum_errors))));
            }
            if let Some(h) = &s.health {
                body = body.child(row("Health", div().text_sm().child(format!("block latency {} µs (max {}), ring {:.0} %, {} frames dropped", h.block_latency_median_us, h.block_latency_max_us, h.ring_use * 100.0, h.frames_dropped))));
            }

            let running = s.state == wire::DeviceState::Running as i32;
            let button = |id: &'static str, label: &'static str| Button::new(id).outline().xsmall().label(label).disabled(self.busy);
            let (m1, m2, m3, m4, m5) = (me.clone(), me.clone(), me.clone(), me.clone(), me.clone());
            let actions = h_flex()
                .gap_1()
                .flex_wrap()
                .child(button("devices-configure", "Send chain…").disabled(self.busy || running).on_click(move |_, _, cx| {
                    let _ = m1.update(cx, |p, cx| p.configure(cx));
                }))
                .child(button("devices-record", "Record").disabled(self.busy || running).on_click(move |_, _, cx| {
                    let _ = m2.update(cx, |p, cx| p.control("Starting a recording", cx, |i| i.start(true, "")));
                }))
                .child(button("devices-preview", "Preview").disabled(self.busy || running).on_click(move |_, _, cx| {
                    let _ = m3.update(cx, |p, cx| p.control("Starting a preview", cx, |i| i.start(false, "")));
                }))
                .child(button("devices-mark", "Mark").disabled(self.busy || !running).on_click(move |_, _, cx| {
                    let _ = m4.update(cx, |p, cx| {
                        if let Some(i) = p.instrument.clone() {
                            p.run("Marking", move || i.mark("mark").map(|_| "Marked".to_string()), cx);
                        }
                    });
                }))
                .child(button("devices-stop", "Stop").disabled(self.busy || !running).on_click(move |_, _, cx| {
                    let _ = m5.update(cx, |p, cx| p.control("Stopping", cx, |i| i.stop()));
                }));
            body = body.child(actions);

            // The taps, each watchable in the strip
            let taps: Vec<wire::TapInfo> = s.chain.as_ref().map(|c| c.taps.clone()).unwrap_or_default();
            let mut tap_row = h_flex().gap_1().flex_wrap().child(div().text_xs().text_color(muted).child("Watch:"));
            let raw: Vec<String> = instrument.welcome().device.as_ref().map(|d| d.sources.iter().map(|s| s.name.clone()).collect()).unwrap_or_default();
            let mut names: Vec<String> = raw;
            for tap in &taps {
                if !names.contains(&tap.name) && tap.kind == wire::TapKind::Samples as i32 {
                    names.push(tap.name.clone());
                }
            }
            for (i, name) in names.into_iter().enumerate() {
                let (m, n) = (me.clone(), name.clone());
                let chosen = self.tap.as_deref() == Some(name.as_str());
                let b = Button::new(SharedString::from(format!("devices-tap-{i}"))).xsmall().label(name);
                let b = if chosen { b.primary() } else { b.ghost() };
                tap_row = tap_row.child(b.on_click(move |_, _, cx| {
                    let _ = m.update(cx, |p, cx| p.watch(n.clone(), cx));
                }));
            }
            body = body.child(tap_row);

            // The live strip: the newest columns of up to STRIP_CHANNELS channels
            if let Some(live) = &self.live {
                let live = live.lock().unwrap_or_else(|e| e.into_inner());
                let start = live.columns.len().saturating_sub(STRIP_COLUMNS);
                let shown: Vec<&Vec<(f32, f32)>> = live.columns.iter().skip(start).collect();
                let rows = shown.first().map_or(0, |c| c.len());
                for ch in 0..rows {
                    let (lo, hi) = shown.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), c| (a.min(c[ch].0), b.max(c[ch].1)));
                    let span = (hi - lo).max(f32::EPSILON);
                    let mut strip = h_flex().h(px(STRIP_HEIGHT)).items_start().gap_0().border_b_1().border_color(border);
                    for col in &shown {
                        let (mn, mx) = col[ch];
                        let top = (hi - mx) / span * STRIP_HEIGHT;
                        let height = ((mx - mn) / span * STRIP_HEIGHT).max(1.0);
                        strip = strip.child(v_flex().w(px(2.)).flex_none().child(div().h(px(top))).child(div().h(px(height)).w_full().bg(accent)));
                    }
                    body = body.child(h_flex().gap_2().child(div().w(px(56.)).flex_none().text_xs().text_color(muted).child(format!("ch {ch}  {:.0}…{:.0}", lo, hi))).child(strip));
                }
                body = body.child(div().text_xs().text_color(muted).child(format!(
                    "up to sample {} · {:.1} samples per column{}",
                    live.head_sample(),
                    live.samples_per_column,
                    if live.dropped > 0 { format!(" · {} frames dropped (a slow connection)", live.dropped) } else { String::new() }
                )));
            }
        }

        v_flex()
            .id("devices")
            .w(px(720.))
            .max_h(gpui_kit::relative(0.85))
            .overflow_y_scroll()
            .gap_2()
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Devices"))
            .child(if profiles.is_empty() {
                div().text_sm().text_color(muted).child("No devices yet: add one in Settings › Servers (its address, and the identity `dspk pair` wrote).").into_any_element()
            } else {
                h_flex().gap_2().child(row("Device", picker)).child(connect).into_any_element()
            })
            .child(div().border_b_1().border_color(border))
            .child(body)
            .children(self.message.clone().map(|(good, text)| div().text_xs().text_color(if good { ok } else { danger }).child(text)))
            .child(div().text_xs().text_color(muted).child("The device runs and records on its own: closing this, or losing the connection, never stops it."))
    }
}
