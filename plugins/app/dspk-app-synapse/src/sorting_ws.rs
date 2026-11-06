//! The Sorting workspace (Plexon Offline Sorter style): run the installed sorters on the active
//! view, or on a part of it, and look at what they found, channel by channel.
//!
//! - Left, the control grid: the sorter (the ones `dspk-cli-sorter describe` lists), the part
//!   of the recording (start, duration: a short run first), the probe, and every parameter of
//!   the sorter with its default shown (typed values replace it); Run.
//! - Centre: the view's runs (each sort made here; click one to show it, everywhere), the units
//!   of the selected channel as cards (waveform mean ± sd, ISI, spikes) and their clusters
//!   (PC1 · PC2 of the channel's spikes, unsorted grey).
//! - Right: the channels that have units, with their units and spikes; click one.
//! - Bottom: the strip (the channel's traces or a raster over the shared timeline).
//!
//! Judging and editing units (labels, merges) is the Curation workspace's.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use dspk_app_core::data::jobs::processed_path;
use dspk_app_core::data::store::{file_label, AppEvent, Store};
use dspk_app_core::domain::command::SessionCommand;
use dspk_app_core::navigation::route::Workspace;
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::widgets::{row, EmptyState, MenuSelect, Section};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Task, Window,
};

use crate::sort::{self, Described, Param, SortRequest, PROBES};
use crate::sorting::{isi_histogram, write_start_sec, Snippets, SortingData};
use crate::strip::{view_signal, BottomStrip};
use crate::{plots, plugin_data, runs_of, sorting_of, state, SynapseEvent, ID};

/// Widths of the side columns (logical px), card size (logical and rendered px).
const LEFT_W: f32 = 300.;
const RIGHT_W: f32 = 240.;
const CARD_W: f32 = 200.;
const CARD_IMG_H: f32 = 120.;
const CARD_PX: (u32, u32) = (400, 240);
const CLUSTER_PX: (u32, u32) = (640, 480);
/// Spikes per unit read for the cards and clusters; units per channel shown.
const SPIKES_PER_UNIT: usize = 200;
const MAX_CARDS: usize = 12;
/// ISI of the cards: 0–30 ms in 0.5 ms bins, refractory period marked.
const ISI_MAX_MS: f64 = 30.0;
const ISI_BIN_MS: f64 = 0.5;
const REFRACTORY_MS: f64 = 1.5;
/// Default part of the recording for a first run (s).
const DEFAULT_DURATION_SEC: f64 = 60.0;

pub struct SortingWorkspace {
    store: Entity<Store>,
    strip: Entity<BottomStrip>,
    described: Option<Result<Described, String>>,
    /// Index into the described sorters.
    sorter: usize,
    /// The sorter's parameters and their inputs (empty: the default).
    inputs: Vec<(Param, Entity<InputState>)>,
    inputs_for: String,
    start: Entity<InputState>,
    duration: Entity<InputState>,
    /// The channel whose units show (the sorting's channel numbering).
    channel: Option<usize>,
    cards: Vec<(usize, Arc<RenderImage>)>,
    clusters: Option<Arc<RenderImage>>,
    computing: bool,
    /// The sorting the cards were made for.
    shown: Option<*const SortingData>,
    _describe: Option<Task<()>>,
    _work: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl SortingWorkspace {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let strip = cx.new(|cx| BottomStrip::new(store.clone(), cx));
        let start = cx.new(|cx| InputState::new(window, cx).default_value("0"));
        let duration = cx.new(|cx| InputState::new(window, cx).default_value(format!("{DEFAULT_DURATION_SEC}")).placeholder("to the end"));
        let mut subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| match e {
            AppEvent::Document(_) | AppEvent::RecordingChanged => this.refresh(cx),
            AppEvent::JobChanged => cx.notify(),
            _ => {}
        })];
        if let Some(st) = state(cx) {
            subs.push(cx.subscribe(&st, |this, _, e: &SynapseEvent, cx| {
                if matches!(e, SynapseEvent::SortingLoaded(_) | SynapseEvent::Curated(_)) {
                    this.refresh(cx);
                }
            }));
        }
        let mut this = Self {
            store,
            strip,
            described: None,
            sorter: 0,
            inputs: Vec::new(),
            inputs_for: String::new(),
            start,
            duration,
            channel: None,
            cards: Vec::new(),
            clusters: None,
            computing: false,
            shown: None,
            _describe: None,
            _work: None,
            _subs: subs,
        };
        this.describe(cx);
        this.refresh(cx);
        this
    }

    /// Asks the installed program what it can run (in the background).
    fn describe(&mut self, cx: &mut Context<Self>) {
        self.described = None;
        let work = cx.background_spawn(async { sort::describe() });
        self._describe = Some(cx.spawn(async move |this, cx| {
            let d = work.await;
            let _ = this.update(cx, |w, cx| {
                w.described = Some(d);
                w.inputs_for.clear();
                cx.notify();
            });
        }));
    }

    fn sorting(&self, cx: &App) -> Option<Arc<SortingData>> {
        state(cx)?.read(cx).active_sorting(cx)
    }

    /// Inputs for the chosen sorter's parameters (rebuilt when it changes).
    fn sync_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(Ok(d)) = &self.described else { return };
        let Some(s) = d.sorters.get(self.sorter) else { return };
        if self.inputs_for == s.name {
            return;
        }
        self.inputs_for = s.name.clone();
        self.inputs = s
            .params
            .iter()
            .map(|p| {
                let hint = if p.choices.is_empty() { p.default.clone() } else { format!("{} ({})", p.default, p.choices.join(" / ")) };
                (p.clone(), cx.new(|cx| InputState::new(window, cx).placeholder(if hint.is_empty() { "unset".to_string() } else { hint })))
            })
            .collect();
    }

    /// The channel to show, the cards and clusters when the sorting changed.
    fn refresh(&mut self, cx: &mut Context<Self>) {
        let sorting = self.sorting(cx);
        let ptr = sorting.as_ref().map(Arc::as_ptr);
        if ptr != self.shown {
            self.shown = ptr;
            if let Some(s) = &sorting {
                if !self.channel.is_some_and(|c| s.alive().any(|u| s.units[u].info.primary_channel == c)) {
                    self.channel = s.alive().max_by_key(|&u| s.units[u].info.num_spikes).map(|u| s.units[u].info.primary_channel);
                }
            }
            self.compute(cx);
        }
        cx.notify();
    }

    fn pick_channel(&mut self, channel: usize, cx: &mut Context<Self>) {
        self.channel = Some(channel);
        self.compute(cx);
    }

    /// The units of the channel, largest first.
    fn channel_units(sorting: &SortingData, channel: usize) -> Vec<usize> {
        let mut units: Vec<usize> = sorting.alive().filter(|&u| sorting.units[u].info.primary_channel == channel).collect();
        units.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
        units
    }

    /// Cards and clusters of the channel's units, on a background thread.
    fn compute(&mut self, cx: &mut Context<Self>) {
        let (Some(sorting), Some(channel)) = (self.sorting(cx), self.channel) else {
            self.cards.clear();
            self.clusters = None;
            return;
        };
        let s = self.store.read(cx);
        let Some((signal, _, _)) = view_signal(s) else { return };
        // A subset view sorted on its own output: the sorting's channel i is the view's i-th
        let map = s.active_view().and_then(|v| v.recording.channels.clone()).filter(|c| sorting.channels == Some(c.len()));
        let to_view = move |c: usize| map.as_ref().map_or(Some(c), |m| m.get(c).copied());
        let Some(view_channel) = to_view(channel) else { return };
        let mut units = Self::channel_units(&sorting, channel);
        units.truncate(MAX_CARDS);
        let palette = s.palette();
        let lead_channels: Vec<usize> = units.first().map(|&u| sorting.units[u].extent.channels.iter().filter_map(|&c| to_view(c)).collect()).unwrap_or_else(|| vec![view_channel]);
        let strip_units = units.clone();
        self.strip.update(cx, |st, cx| st.set_focus(strip_units, lead_channels, cx));
        self.computing = true;
        let work = cx.background_spawn(async move {
            let snippets = Snippets::read(&sorting, signal.as_ref(), &units, view_channel, SPIKES_PER_UNIT);
            let cards = units
                .iter()
                .filter_map(|&u| {
                    let rows: Vec<&Vec<f32>> = snippets.rows.iter().filter(|(x, _, _)| *x == Some(u as u32)).map(|(_, _, r)| r).collect();
                    let isi = isi_histogram(&sorting.unit_times(u), sorting.sample_rate, ISI_MAX_MS, ISI_BIN_MS);
                    let frame = plots::unit_card(&rows, &isi, (REFRACTORY_MS / ISI_BIN_MS) as f32, palette.unit(u), &palette, CARD_PX.0, CARD_PX.1, 1.5);
                    plots::to_image(frame).map(|i| (u, i))
                })
                .collect::<Vec<_>>();
            let points = snippets.pca2();
            let labels: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
            let clusters = plots::to_image(plots::scatter(&points, &labels, &palette, CLUSTER_PX.0, CLUSTER_PX.1, 1.5));
            (cards, clusters)
        });
        self._work = Some(cx.spawn(async move |this, cx| {
            let (cards, clusters) = work.await;
            let _ = this.update(cx, |w, cx| {
                w.cards = cards;
                w.clusters = clusters;
                w.computing = false;
                cx.notify();
            });
        }));
    }

    /// `Some(seconds)` from an input; empty: `None`.
    fn seconds(input: &Entity<InputState>, cx: &App) -> Result<Option<f64>, String> {
        let text = input.read(cx).value().trim().to_string();
        if text.is_empty() {
            return Ok(None);
        }
        text.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0).map(Some).ok_or_else(|| format!("{text:?} is not a number of seconds"))
    }

    /// The visible window becomes the part to sort.
    fn use_visible_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (start, win) = {
            let tl = &self.store.read(cx).timeline;
            let origin = view_signal(self.store.read(cx)).map_or(0.0, |v| v.1.start_time_sec);
            ((tl.window_start_sec - origin).max(0.0), tl.visible_window_sec)
        };
        self.start.update(cx, |i, cx| i.set_value(format!("{start:.3}"), window, cx));
        self.duration.update(cx, |i, cx| i.set_value(format!("{win:.3}"), window, cx));
    }

    /// Runs the chosen sorter on the active view over the part asked for; the result is linked
    /// to the view and added to its runs.
    fn run(&mut self, cx: &mut Context<Self>) {
        let status = |w: &Self, text: String, cx: &mut Context<Self>| w.store.update(cx, |s, cx| s.set_status(text, cx));
        let Some(Ok(d)) = &self.described else { return };
        let Some(sorter) = d.sorters.get(self.sorter).map(|s| s.name.clone()) else { return };
        let Some(view) = self.store.read(cx).active_view().cloned() else { return };
        let (start, duration) = match (Self::seconds(&self.start, cx), Self::seconds(&self.duration, cx)) {
            (Ok(s), Ok(d)) => (s.filter(|s| *s > 0.0), d),
            (Err(e), _) | (_, Err(e)) => return status(self, e, cx),
        };
        let params: Vec<(String, String)> = self
            .inputs
            .iter()
            .filter_map(|(p, input)| {
                let v = input.read(cx).value().trim().to_string();
                (!v.is_empty() && v != p.default).then(|| (p.flag.clone(), v))
            })
            .collect();
        let processed = view.ops.iter().any(|o| o.enabled);
        let (recording, source) = if processed {
            let written = processed_path(&view.recording.path, &view.id);
            if !written.exists() {
                return status(self, format!("Write the processed recording of {} first (Pipeline or Explore → Processing → Write): sorters read files", view.name), cx);
            }
            (written, None)
        } else {
            (view.recording.path.clone(), view.recording.source.clone())
        };
        let probe = state(cx).map_or(0, |s| s.read(cx).probe);
        let output = sort::run_path(&recording, &sorter, start, duration);
        let req = SortRequest { sorter: sorter.clone(), recording, source, probe: PROBES[probe].0.map(str::to_string), output, params, start_sec: start, duration_sec: duration };
        let (id, part) = (view.id.clone(), match (start, duration) {
            (None, None) => "the whole recording".to_string(),
            (s, d) => format!("{} s{}", s.unwrap_or(0.0), d.map(|d| format!(" + {d} s")).unwrap_or_else(|| " to the end".into())),
        });
        self.store.update(cx, |s, cx| {
            s.start_job(
                format!("{sorter} on {} ({part})", view.name),
                move |send| sort::sort(&req, move |m| send(m)),
                move |s, path, cx| {
                    if let Some(t) = start {
                        if let Err(e) = write_start_sec(&path, t) {
                            s.set_status(format!("{}: {e}", path.display()), cx);
                        }
                    }
                    let Some(v) = s.doc.view(&id).cloned() else { return };
                    let data = plugin_data(&v, Some(&path), Some(&path));
                    let _ = s.execute(SessionCommand::SetPluginData { view: id.clone(), plugin: ID.into(), data }, cx);
                },
                cx,
            );
        });
    }

    /// Shows run `path` (links it to the view; Explore, Curation and this follow).
    fn show_run(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let Some(view) = self.store.read(cx).active_view().cloned() else { return };
        let data = plugin_data(&view, Some(&path), None);
        let _ = self.store.update(cx, |s, cx| s.execute(SessionCommand::SetPluginData { view: view.id.clone(), plugin: ID.into(), data }, cx));
    }

    // ------------------------------------------------------------------------
    // Columns
    // ------------------------------------------------------------------------

    fn control_grid(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        self.sync_inputs(window, cx);
        let t = cx.theme();
        let (muted, danger) = (t.muted_foreground, t.danger);
        let running = self.store.read(cx).job.as_ref().map(|j| (j.label.clone(), j.fraction, j.line.clone()));
        let me = cx.entity().downgrade();
        let sorter_section = match &self.described {
            None => Section::new("Sorter").child(div().text_xs().text_color(muted).child("Asking dspk-cli-sorter what it runs…")),
            Some(Err(e)) => Section::new("Sorter").child(div().text_xs().text_color(danger).child(e.clone())).child({
                let me = me.clone();
                Button::new("sorters-retry").outline().small().label("Look again").on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.describe(cx));
                })
            }),
            Some(Ok(d)) => {
                let names: Vec<SharedString> = d.sorters.iter().map(|s| SharedString::from(s.name.clone())).collect();
                let pick = {
                    let me = me.clone();
                    MenuSelect::new("sorting-sorter", names, Some(self.sorter), move |i, _, cx| {
                        let _ = me.update(cx, |w, cx| {
                            w.sorter = i;
                            cx.notify();
                        });
                    })
                    .full_width()
                };
                let about = d.sorters.get(self.sorter).map(|s| s.about.clone()).unwrap_or_default();
                Section::new("Sorter")
                    .child(pick)
                    .child(div().text_xs().text_color(muted).child(about))
                    .child(div().text_xs().text_color(muted).truncate().child(format!("dspk-cli-sorter {} · {}", d.version, d.program.display())))
            }
        };
        let probe = state(cx).map_or(0, |s| s.read(cx).probe);
        let probe_pick = MenuSelect::new("sorting-probe", PROBES.iter().map(|(_, n)| SharedString::from(*n)).collect(), Some(probe), move |i, _, cx| {
            if let Some(st) = state(cx) {
                st.update(cx, |s, cx| {
                    s.probe = i;
                    s.choices_changed(cx);
                });
            }
        });
        let part = Section::new("Part of the recording")
            .child(row("Start (s)", div().w(px(110.)).child(Input::new(&self.start).small()), cx))
            .child(row("Duration (s)", div().w(px(110.)).child(Input::new(&self.duration).small()), cx))
            .child({
                let me = me.clone();
                Button::new("sorting-visible-window").ghost().xsmall().label("Use the visible window").on_click(move |_, window, cx| {
                    let _ = me.update(cx, |w, cx| w.use_visible_window(window, cx));
                })
            })
            .child(div().text_xs().text_color(muted).child("Sort a minute first to try settings; an empty duration sorts to the end."))
            .child(row("Probe", probe_pick, cx));
        let params = Section::new(format!("Parameters ({})", self.inputs.len())).children(self.inputs.iter().map(|(p, input)| {
            v_flex()
                .gap_0p5()
                .child(row(p.name.replace('_', " "), div().w(px(120.)).child(Input::new(input).small()), cx))
                .when(!p.help.is_empty(), |d| d.child(div().text_xs().text_color(muted).line_clamp(2).child(p.help.clone())))
        }));
        let run = match running {
            Some((label, fraction, line)) => v_flex()
                .gap_1()
                .child(div().text_xs().child(format!("{label}{}", fraction.map(|f| format!(" · {:.0} %", f * 100.0)).unwrap_or_default())))
                .child(div().text_xs().text_color(muted).truncate().child(line))
                .child({
                    let store = self.store.clone();
                    Button::new("sorting-cancel").outline().small().label("Cancel").on_click(move |_, _, cx| store.update(cx, |s, cx| s.cancel_job(cx)))
                })
                .into_any_element(),
            None => {
                let me = me.clone();
                let ready = matches!(self.described, Some(Ok(_))) && self.store.read(cx).active_view().is_some();
                Button::new("sorting-run").primary().icon(Icon::new(IconName::Play)).label("Run").disabled(!ready).on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.run(cx));
                }).into_any_element()
            }
        };
        v_flex()
            .id("sorting-controls")
            .w(px(LEFT_W))
            .flex_none()
            .h_full()
            .overflow_y_scroll()
            .p_3()
            .gap_4()
            .child(sorter_section)
            .child(part)
            .child(run)
            .child(params)
            .into_any_element()
    }

    fn runs(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, border, accent) = (t.muted_foreground, t.border, t.accent);
        let view = self.store.read(cx).active_view().cloned();
        let runs = view.as_ref().map(runs_of).unwrap_or_default();
        let linked = view.as_ref().and_then(sorting_of);
        let me = cx.entity().downgrade();
        let chips: Vec<_> = runs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let on = linked.as_ref() == Some(p);
                let name = file_label(p).trim_end_matches(".sorting.zarr").to_string();
                let (me, p) = (me.clone(), p.clone());
                h_flex()
                    .id(("sorting-run", i))
                    .h(px(26.))
                    .px_2()
                    .gap_1()
                    .flex_none()
                    .rounded_md()
                    .border_1()
                    .border_color(border)
                    .cursor_pointer()
                    .when(on, |d| d.bg(accent))
                    .child(div().text_xs().child(name))
                    .on_click(move |_, _, cx| {
                        let _ = me.update(cx, |w, cx| w.show_run(p.clone(), cx));
                    })
            })
            .collect();
        let store = self.store.clone();
        h_flex()
            .id("sorting-runs")
            .h(px(40.))
            .flex_none()
            .px_3()
            .gap_2()
            .overflow_x_scroll()
            .border_b_1()
            .border_color(border)
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).flex_none().child("RUNS"))
            .when(chips.is_empty(), |d| d.child(div().text_xs().text_color(muted).child("None yet: run a sorter on the left (each run is kept here).")))
            .children(chips)
            .child(div().flex_1())
            .child(
                Button::new("sorting-to-curation")
                    .ghost()
                    .xsmall()
                    .icon(Icon::new(gpui_kit::assets::IconName::Microscope))
                    .label("Curate")
                    .tooltip("Judge and edit the shown run's units")
                    .disabled(linked.is_none())
                    .on_click(move |_, _, cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Curation, cx))),
            )
    }

    fn units(&self, sorting: &SortingData, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, border, bg) = (t.muted_foreground, t.border, t.background);
        let palette = self.store.read(cx).palette();
        let channel = self.channel;
        let n_units = channel.map_or(0, |c| Self::channel_units(sorting, c).len());
        let cards = self.cards.iter().map(|(u, image)| {
            let info = &sorting.units[*u].info;
            v_flex()
                .w(px(CARD_W))
                .flex_none()
                .rounded_md()
                .border_1()
                .border_color(border)
                .bg(bg)
                .overflow_hidden()
                .child(
                    h_flex()
                        .h(px(24.))
                        .px_2()
                        .gap_1p5()
                        .child(div().size(px(8.)).rounded_full().bg(color(palette.unit(*u))))
                        .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).child(format!("Unit {}", info.unit_id)))
                        .child(div().flex_1())
                        .child(div().text_xs().text_color(muted).child(info.quality_label.clone())),
                )
                .child(img(image.clone()).w(px(CARD_W)).h(px(CARD_IMG_H)))
                .child(h_flex().h(px(22.)).px_2().text_xs().text_color(muted).child(format!("spikes {} · {:.2} Hz · SNR {:.1}", info.num_spikes, info.firing_rate_hz, info.snr)))
        });
        let title = match channel {
            Some(c) => format!("Channel {c} · {n_units} unit{}{}", if n_units == 1 { "" } else { "s" }, if n_units > MAX_CARDS { format!(" (largest {MAX_CARDS} shown)") } else { String::new() }),
            None => "Pick a channel on the right".to_string(),
        };
        v_flex()
            .id("sorting-units")
            .flex_1()
            .min_w_0()
            .h_full()
            .overflow_y_scroll()
            .p_3()
            .gap_3()
            .child(h_flex().gap_2().child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title)).when(self.computing, |d| d.child(div().text_xs().text_color(muted).child("Reading spikes…"))))
            .child(h_flex().flex_wrap().gap_2().children(cards))
            .child(
                v_flex()
                    .gap_1()
                    .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child("CLUSTERS · PC1 × PC2 of the channel's spikes (unsorted grey)"))
                    .child(match &self.clusters {
                        Some(i) => img(i.clone()).w(px(480.)).h(px(360.)).rounded_md().into_any_element(),
                        None => div().h(px(120.)).text_xs().text_color(muted).child("—").into_any_element(),
                    }),
            )
    }

    fn channels(&self, sorting: &SortingData, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, border, accent) = (t.muted_foreground, t.border, t.accent);
        let palette = self.store.read(cx).palette();
        // Channel → its units (largest first) and spikes
        let mut by: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for u in sorting.alive() {
            by.entry(sorting.units[u].info.primary_channel).or_default().push(u);
        }
        let me = cx.entity().downgrade();
        let rows = by.into_iter().map(|(c, mut units)| {
            units.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
            let spikes: usize = units.iter().map(|&u| sorting.units[u].info.num_spikes).sum();
            let on = self.channel == Some(c);
            let me = me.clone();
            h_flex()
                .id(("sorting-channel", c))
                .h(px(24.))
                .px_2()
                .gap_2()
                .cursor_pointer()
                .rounded_sm()
                .when(on, |d| d.bg(accent))
                .child(div().w(px(40.)).text_xs().font_weight(FontWeight::SEMIBOLD).child(format!("{c}")))
                .child(h_flex().w(px(80.)).gap_0p5().children(units.iter().take(6).map(|&u| div().size(px(8.)).rounded_full().bg(color(palette.unit(u))))))
                .child(div().w(px(24.)).text_xs().text_color(muted).child(units.len().to_string()))
                .child(div().flex_1().text_xs().text_color(muted).text_right().child(spikes.to_string()))
                .on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.pick_channel(c, cx));
                })
        });
        v_flex()
            .w(px(RIGHT_W))
            .flex_none()
            .h_full()
            .border_l_1()
            .border_color(border)
            .child(
                h_flex()
                    .h(px(28.))
                    .flex_none()
                    .px_2()
                    .gap_2()
                    .border_b_1()
                    .border_color(border)
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(muted)
                    .child(div().w(px(40.)).child("Ch"))
                    .child(div().w(px(80.)).child("Units"))
                    .child(div().w(px(24.)).child("#"))
                    .child(div().flex_1().text_right().child("Spikes")),
            )
            .child(v_flex().id("sorting-channel-list").flex_1().min_h_0().overflow_y_scroll().p_1().children(rows))
    }
}

impl Render for SortingWorkspace {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bg = cx.theme().background;
        let controls = self.control_grid(window, cx);
        let runs = self.runs(cx);
        let body: AnyElement = match self.sorting(cx) {
            Some(sorting) => h_flex().flex_1().min_h_0().child(self.units(&sorting, cx)).child(self.channels(&sorting, cx)).into_any_element(),
            None => div()
                .flex_1()
                .child(EmptyState::new(
                    gpui_kit::assets::IconName::SlidersHorizontal,
                    "No sorting shown yet",
                    "Choose a sorter and a part of the recording on the left, then Run. Each run shows here, channel by channel.",
                ))
                .into_any_element(),
        };
        v_flex()
            .id("sorting-workspace")
            .size_full()
            .bg(bg)
            .child(h_flex().flex_1().min_h_0().child(controls).child(v_flex().flex_1().min_w_0().h_full().border_l_1().border_color(cx.theme().border).child(runs).child(body)))
            .child(self.strip.clone())
    }
}
