//! The Curation workspace (phy's template GUI): Clusters over Similar on the left, Waveforms and
//! Correlograms in the centre, in a dock like Explore's (drag tabs to rearrange). Selected clusters
//! keep one colour in every view (first blue, second red, …).

use std::rc::Rc;

use gpui_kit::base::dock::{DockArea, DockLayout, DockPlacement, Panel as BasePanel, PanelEvent};
use gpui_kit::component::dock::{panel_handle, DockSkin, Panel, PanelControl};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, point, px, size, uniform_list, App, AppContext as _, Bounds, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, Hsla, InteractiveElement as _,
    IntoElement, ParentElement as _, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::engine::curation::{ClusterGroup, ClusterSortColumn};
use crate::store::Store;
use crate::viewmodels::curation::{CurationEvent, CurationVm, REFRACTORY_MS};
use crate::views::plot::{cluster_hsla, paint_polyline};
use crate::widgets::EmptyState;

fn group_color(g: ClusterGroup, cx: &App) -> Hsla {
    let t = cx.theme();
    match g {
        ClusterGroup::Good => t.success,
        ClusterGroup::Mua => t.warning,
        ClusterGroup::Noise => t.muted_foreground,
        ClusterGroup::Unsorted => t.foreground,
    }
}

macro_rules! curation_panel {
    ($ty:ty, $name:literal, $title:literal, $zoom:expr) => {
        impl EventEmitter<PanelEvent> for $ty {}

        impl Focusable for $ty {
            fn focus_handle(&self, _: &App) -> FocusHandle {
                self.focus.clone()
            }
        }

        impl BasePanel for $ty {
            fn panel_name(&self) -> &'static str {
                $name
            }

            fn closable(&self, _: &App) -> bool {
                false
            }

            fn zoomable(&self, _: &App) -> bool {
                $zoom
            }
        }

        impl Panel for $ty {
            fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                $title
            }

            fn zoom_control(&self, _: &App) -> Option<PanelControl> {
                $zoom.then_some(PanelControl::Toolbar)
            }

            fn inner_padding(&self, _: &App) -> bool {
                false
            }
        }
    };
}

// ----------------------------------------------------------------------------
// Clusters
// ----------------------------------------------------------------------------

const COLUMNS: [(ClusterSortColumn, &str, f32); 9] = [
    (ClusterSortColumn::Id, "id", 44.),
    (ClusterSortColumn::Channel, "ch", 40.),
    (ClusterSortColumn::Depth, "depth", 52.),
    (ClusterSortColumn::Spikes, "spikes", 60.),
    (ClusterSortColumn::FiringRate, "fr Hz", 52.),
    (ClusterSortColumn::Amplitude, "amp", 52.),
    (ClusterSortColumn::ContamPct, "contam %", 64.),
    (ClusterSortColumn::KsLabel, "KS label", 64.),
    (ClusterSortColumn::Group, "group", 64.),
];

pub struct ClustersPanel {
    vm: Entity<CurationVm>,
    focus: FocusHandle,
    filter: Entity<InputState>,
    _subs: Vec<Subscription>,
}

curation_panel!(ClustersPanel, "dsp-app.clusters", "Clusters", false);

impl ClustersPanel {
    pub fn new(vm: Entity<CurationVm>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let filter = cx.new(|cx| InputState::new(window, cx).placeholder("Filter, e.g. group == 'good' && fr > 1"));
        let subs = vec![
            cx.observe(&vm, |_, _, cx| cx.notify()),
            cx.subscribe(&filter, |this, state, e: &InputEvent, cx| {
                if matches!(e, InputEvent::Change) {
                    let text = state.read(cx).value().to_string();
                    this.vm.update(cx, |vm, cx| vm.set_filter(text, cx));
                }
            }),
        ];
        Self { vm, focus: cx.focus_handle(), filter, _subs: subs }
    }
}

impl Render for ClustersPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        let rows = Rc::new(vm.rows());
        let total = vm.data.as_ref().map_or(0, |d| d.clusters.len());
        let (sort, descending) = vm.sort;
        let selected = vm.selected.clone();
        let t = cx.theme();
        let (muted, border, hover, sel_bg) = (t.muted_foreground, t.border, t.list_hover, t.list_active);
        let header = h_flex().h(px(26.)).px_2().border_b_1().border_color(border).children(COLUMNS.map(|(col, label, w)| {
            let vm = self.vm.clone();
            let arrow = if col == sort { if descending { " ↓" } else { " ↑" } } else { "" };
            div()
                .id(SharedString::from(format!("sort-{label}")))
                .w(px(w))
                .flex_none()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(if col == sort { t.foreground } else { muted })
                .cursor_pointer()
                .child(format!("{label}{arrow}"))
                .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.sort_by(col, cx)))
        }));
        let vm_handle = self.vm.clone();
        let group_colors: Vec<Hsla> = [ClusterGroup::Good, ClusterGroup::Mua, ClusterGroup::Noise, ClusterGroup::Unsorted].map(|g| group_color(g, cx)).to_vec();
        let n_rows = rows.len();
        let list = uniform_list("cluster-rows", n_rows, move |range, _, _| {
            range
                .map(|i| {
                    let m = &rows[i];
                    let id = m.id;
                    let rank = selected.iter().position(|&c| c == id);
                    let vm = vm_handle.clone();
                    let gc = group_colors[m.group as usize];
                    let cells = [
                        id.to_string(),
                        m.ch.to_string(),
                        format!("{:.0}", m.depth),
                        m.n_spikes.to_string(),
                        format!("{:.2}", m.fr),
                        format!("{:.1}", m.amp),
                        format!("{:.1}", m.contam_pct),
                        m.ks_label.clone(),
                        m.group.as_str().to_string(),
                    ];
                    h_flex()
                        .id(("cluster", id as usize))
                        .test_support()
                        .h(px(24.))
                        .px_2()
                        .text_xs()
                        .cursor_pointer()
                        .when_some(rank, |d, _| d.bg(sel_bg))
                        .when(rank.is_none(), |d| d.hover(|s| s.bg(hover)))
                        .on_click(move |e, _, cx| {
                            let add = e.modifiers().control || e.modifiers().shift || e.modifiers().platform;
                            vm.update(cx, |vm, cx| vm.select(id, add, cx));
                        })
                        .children(cells.into_iter().zip(COLUMNS).enumerate().map(move |(k, (text, (_, _, w)))| {
                            let cell = div().w(px(w)).flex_none().truncate().child(text);
                            match k {
                                // The id carries the selection colour, the group its own
                                0 => match rank {
                                    Some(r) => cell.font_weight(FontWeight::BOLD).text_color(cluster_hsla(r)),
                                    None => cell,
                                },
                                8 => cell.text_color(gc),
                                _ => cell,
                            }
                        }))
                })
                .collect()
        })
        .flex_1()
        .min_h_0();
        v_flex()
            .id("clusters-panel")
            .test_support()
            .size_full()
            .child(h_flex().gap_2().p_2().child(div().flex_1().child(Input::new(&self.filter).small())).child(div().text_xs().text_color(muted).child(format!("{n_rows} of {total}"))))
            .child(header)
            .child(list)
    }
}

// ----------------------------------------------------------------------------
// Similar
// ----------------------------------------------------------------------------

pub struct SimilarPanel {
    vm: Entity<CurationVm>,
    focus: FocusHandle,
    _sub: Subscription,
}

curation_panel!(SimilarPanel, "dsp-app.similar", "Similar", false);

impl SimilarPanel {
    pub fn new(vm: Entity<CurationVm>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self { vm, focus: cx.focus_handle(), _sub: sub }
    }
}

impl Render for SimilarPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        let similar = Rc::new(vm.similar());
        let best = vm.selected.first().copied();
        let selected = vm.selected.clone();
        let clusters = vm.data.as_ref().map(|d| d.clusters.clone()).unwrap_or_default();
        let t = cx.theme();
        let (muted, border, hover, sel_bg) = (t.muted_foreground, t.border, t.list_hover, t.list_active);
        let vm_handle = self.vm.clone();
        let n = similar.len();
        let list = uniform_list("similar-rows", n, move |range, _, _| {
            range
                .map(|i| {
                    let (cid, score) = similar[i];
                    let rank = selected.iter().position(|&c| c == cid);
                    let m = clusters.get(&cid);
                    let vm = vm_handle.clone();
                    h_flex()
                        .id(("similar", cid as usize))
                        .test_support()
                        .h(px(24.))
                        .px_2()
                        .gap_3()
                        .text_xs()
                        .cursor_pointer()
                        .when_some(rank, |d, _| d.bg(sel_bg))
                        .when(rank.is_none(), |d| d.hover(|s| s.bg(hover)))
                        .on_click(move |e, _, cx| {
                            let add = e.modifiers().control || e.modifiers().shift || e.modifiers().platform;
                            vm.update(cx, |vm, cx| vm.select_similar(cid, add, cx));
                        })
                        .child(div().w(px(44.)).when_some(rank, |d, r| d.font_weight(FontWeight::BOLD).text_color(cluster_hsla(r))).child(cid.to_string()))
                        .child(div().w(px(52.)).child(format!("{score:.3}")))
                        .child(div().w(px(40.)).text_color(muted).child(m.map_or(String::new(), |m| format!("ch {}", m.ch))))
                        .child(div().w(px(70.)).text_color(muted).child(m.map_or(String::new(), |m| format!("{} spikes", m.n_spikes))))
                        .child(div().text_color(muted).child(m.map_or(String::new(), |m| m.group.as_str().to_string())))
                })
                .collect()
        })
        .flex_1()
        .min_h_0();
        v_flex()
            .id("similar-panel")
            .test_support()
            .size_full()
            .child(
                h_flex()
                    .h(px(26.))
                    .px_2()
                    .border_b_1()
                    .border_color(border)
                    .text_xs()
                    .text_color(muted)
                    .child(best.map_or("Select a cluster".to_string(), |b| format!("Most similar to {b} (template similarity)"))),
            )
            .child(list)
    }
}

// ----------------------------------------------------------------------------
// Waveforms
// ----------------------------------------------------------------------------

/// Spikes drawn per cluster (of those read), so the mean stays readable.
const DRAWN_SPIKES: usize = 40;
const LABELS: f32 = 76.;

pub struct WaveformsPanel {
    vm: Entity<CurationVm>,
    focus: FocusHandle,
    _sub: Subscription,
}

curation_panel!(WaveformsPanel, "dsp-app.waveforms", "Waveforms", true);

impl WaveformsPanel {
    pub fn new(vm: Entity<CurationVm>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self { vm, focus: cx.focus_handle(), _sub: sub }
    }
}

impl Render for WaveformsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        let waves = vm.waveforms.clone();
        let has_recording = vm.recording(cx).is_some();
        let muted = cx.theme().muted_foreground;
        let Some(first) = waves.first() else {
            return div().size_full().flex().items_center().justify_center().text_sm().text_color(muted).child("Select a cluster").into_any_element();
        };
        // Channels top to bottom by depth (deepest at the bottom, as on the probe)
        let mut order: Vec<usize> = (0..first.channels.len()).collect();
        order.sort_by(|&a, &b| first.positions[b][1].total_cmp(&first.positions[a][1]));
        let labels: Vec<String> = order.iter().map(|&r| format!("ch {} · {:.0} µm", first.channels[r], first.positions[r][1])).collect();
        let lanes = order.len();
        let note = (!has_recording).then_some("No recording attached: templates only. Open the recording in Explore to see spikes.");
        let ink = cx.theme().foreground;
        let plot = canvas(
            |_, _, _| {},
            move |bounds, _, window, _| {
                let ns = waves.first().map_or(0, |w| w.num_samples).max(2);
                // One vertical scale for every lane: the largest mean (or template) swing
                let peak = waves
                    .iter()
                    .flat_map(|w| if w.mean.is_empty() { w.template.iter() } else { w.mean.iter() })
                    .fold(1e-6f32, |m, v| m.max(v.abs()));
                let x0 = bounds.origin.x + px(LABELS);
                let width = (bounds.size.width - px(LABELS + 8.)).max(px(10.));
                let lane_h = bounds.size.height / lanes.max(1) as f32;
                let trace = |values: &[f32], row: usize, lane: usize| -> Vec<gpui_kit::Point<Pixels>> {
                    let center = bounds.origin.y + lane_h * (lane as f32 + 0.5);
                    (0..ns)
                        .map(|s| {
                            let v = values.get(row * ns + s).copied().unwrap_or(0.0);
                            point(x0 + width * (s as f32 / (ns - 1) as f32), center - lane_h * 0.48 * (v / peak))
                        })
                        .collect()
                };
                for (k, w) in waves.iter().enumerate() {
                    let c = cluster_hsla(k);
                    for (lane, &row) in order.iter().enumerate() {
                        for spike in w.sampled.iter().take(DRAWN_SPIKES) {
                            paint_polyline(window, &trace(spike, row, lane), 1.0, c.opacity(0.12));
                        }
                        if !w.mean.is_empty() {
                            paint_polyline(window, &trace(&w.mean, row, lane), 2.0, c);
                        }
                        if !w.template.is_empty() {
                            paint_polyline(window, &trace(&w.template, row, lane), 1.0, if w.mean.is_empty() { c } else { ink.opacity(0.5) });
                        }
                    }
                }
            },
        )
        .absolute()
        .size_full();
        div()
            .id("waveforms-panel")
            .test_support()
            .size_full()
            .relative()
            .child(plot)
            .child(v_flex().absolute().left(px(6.)).top_0().bottom_0().w(px(LABELS - 8.)).children(labels.into_iter().map(|l| div().flex_1().flex().items_center().text_xs().text_color(muted).child(l))))
            .when_some(note, |d, n| d.child(div().absolute().right(px(8.)).top(px(6.)).text_xs().text_color(muted).child(n)))
            .into_any_element()
    }
}

// ----------------------------------------------------------------------------
// Correlograms
// ----------------------------------------------------------------------------

/// Clusters shown in the correlogram grid.
const CCG_CLUSTERS: usize = 4;

pub struct CorrelogramsPanel {
    vm: Entity<CurationVm>,
    focus: FocusHandle,
    _sub: Subscription,
}

curation_panel!(CorrelogramsPanel, "dsp-app.correlograms", "Correlograms", true);

impl CorrelogramsPanel {
    pub fn new(vm: Entity<CurationVm>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self { vm, focus: cx.focus_handle(), _sub: sub }
    }
}

impl Render for CorrelogramsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        let muted = cx.theme().muted_foreground;
        let Some(m) = vm.correlograms.clone() else {
            return div().size_full().flex().items_center().justify_center().text_sm().text_color(muted).child("Select a cluster").into_any_element();
        };
        let n = m.clusters.len().min(CCG_CLUSTERS);
        let (border, neutral) = (cx.theme().border, cx.theme().muted_foreground);
        let window_ms = m.window_ms;
        let cell = move |i: usize, j: usize| {
            let ccg = m.cells[i][j].clone();
            let color = if i == j { cluster_hsla(i) } else { neutral };
            let title = if i == j { format!("{}", m.clusters[i]) } else { format!("{} × {}", m.clusters[i], m.clusters[j]) };
            div()
                .flex_1()
                .h_full()
                .relative()
                .border_1()
                .border_color(border)
                .child(
                    canvas(
                        |_, _, _| {},
                        move |bounds, _, window, _| {
                            let max = ccg.counts.iter().copied().max().unwrap_or(0).max(1) as f32;
                            let nb = ccg.counts.len().max(1);
                            let w = bounds.size.width / nb as f32;
                            let h = bounds.size.height - px(14.);
                            for (b, &count) in ccg.counts.iter().enumerate() {
                                let bh = h * (count as f32 / max);
                                let origin = point(bounds.origin.x + w * b as f32, bounds.origin.y + bounds.size.height - bh);
                                window.paint_quad(gpui_kit::fill(Bounds { origin, size: size(w.max(px(1.)), bh) }, color.opacity(0.85)));
                            }
                            // Refractory period, both sides of zero
                            for lag in [-REFRACTORY_MS, REFRACTORY_MS] {
                                let x = bounds.origin.x + bounds.size.width * ((lag + window_ms / 2.0) / window_ms);
                                window.paint_quad(gpui_kit::fill(Bounds { origin: point(x, bounds.origin.y), size: size(px(1.), bounds.size.height) }, neutral.opacity(0.5)));
                            }
                        },
                    )
                    .absolute()
                    .size_full(),
                )
                .child(div().absolute().left(px(4.)).top(px(1.)).text_xs().text_color(color).child(title))
        };
        v_flex()
            .id("correlograms-panel")
            .test_support()
            .size_full()
            .p_1()
            .gap_1()
            .children((0..n).map(|i| h_flex().flex_1().gap_1().children((0..n).map(|j| cell(i, j)))))
            .child(h_flex().justify_between().text_xs().text_color(muted).child(format!("−{:.0} ms", window_ms / 2.0)).child("0").child(format!("+{:.0} ms", window_ms / 2.0)))
            .into_any_element()
    }
}

// ----------------------------------------------------------------------------
// Workspace
// ----------------------------------------------------------------------------

pub struct CurationView {
    pub vm: Entity<CurationVm>,
    pub dock: Entity<DockArea>,
    store: Entity<Store>,
    _sub: Subscription,
}

impl CurationView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let vm = cx.new(|cx| CurationVm::new(store.clone(), cx));
        let mut skin = None;
        let dock = cx.new(|cx| {
            let s = DockSkin::new(cx);
            skin = Some(s.clone());
            DockArea::new("curation", Some(1), window, cx).with_renderer(s)
        });
        if let Some(skin) = skin {
            skin.set_toggle_button_visible(false, cx);
        }
        let clusters = cx.new(|cx| ClustersPanel::new(vm.clone(), window, cx));
        let similar = cx.new(|cx| SimilarPanel::new(vm.clone(), cx));
        let waveforms = cx.new(|cx| WaveformsPanel::new(vm.clone(), cx));
        let correlograms = cx.new(|cx| CorrelogramsPanel::new(vm.clone(), cx));
        dock.update(cx, |d, cx| {
            d.set_dock(
                DockPlacement::Left,
                DockLayout::v_split().child(DockLayout::tabs().panel_view(panel_handle(clusters), cx), Some(px(420.))).child(DockLayout::tabs().panel_view(panel_handle(similar), cx), None),
                window,
                cx,
            );
            d.set_dock_size(DockPlacement::Left, px(640.), window, cx);
            d.set_center(
                DockLayout::h_split().child(DockLayout::tabs().panel_view(panel_handle(waveforms), cx), None).child(DockLayout::tabs().panel_view(panel_handle(correlograms), cx), Some(px(420.))),
                window,
                cx,
            );
        });
        let sub = cx.subscribe(&vm, |_, _, e: &CurationEvent, cx| {
            if matches!(e, CurationEvent::Opened) {
                cx.notify();
            }
        });
        Self { vm, dock, store, _sub: sub }
    }

    pub fn prompt_open(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(gpui_kit::PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some("Open sorting folder".into()) });
        let vm = self.vm.clone();
        cx.spawn(async move |_, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                vm.update(cx, |vm, cx| vm.open(p, cx));
            }
        })
        .detach();
    }
}

impl Render for CurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        if vm.data.is_some() {
            return self.dock.clone().into_any_element();
        }
        let muted = cx.theme().muted_foreground;
        let recent = self.store.read(cx).session.recent_sortings.clone();
        let text = match (&vm.opening, &vm.error) {
            (Some(name), _) => format!("Opening {name}…"),
            (None, Some(e)) => format!("Could not open it: {e}"),
            _ => "A phy / Kilosort output folder (spike_times.npy, spike_clusters.npy, templates.npy…), a .sorting.zarr or NWB units. Its recording opens in Explore when params.py names one.".into(),
        };
        let mut screen = EmptyState::new(gpui_kit::assets::IconName::Microscope, "Open a sorting", text).child(
            gpui_kit::component::button::Button::new("open-sorting")
                .icon(IconName::FolderOpen)
                .label("Open sorting folder…")
                .on_click(cx.listener(|this, _, _, cx| this.prompt_open(cx))),
        );
        if !recent.is_empty() {
            screen = screen.child(v_flex().mt_4().w(px(420.)).gap_1().child(div().text_xs().text_color(muted).child("Recent")).children(recent.into_iter().take(6).enumerate().map(|(i, p)| {
                let label = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                let dir = p.parent().map(|d| d.display().to_string()).unwrap_or_default();
                let vm = self.vm.clone();
                gpui_kit::component::button::Button::new(("recent-sorting", i))
                    .w_full()
                    .child(h_flex().w_full().gap_2().child(div().text_sm().child(label)).child(div().flex_1().text_xs().text_color(muted).truncate().child(dir)))
                    .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.open(p.clone(), cx)))
            })));
        }
        div().id("curation-start").test_support().size_full().child(screen).into_any_element()
    }
}
