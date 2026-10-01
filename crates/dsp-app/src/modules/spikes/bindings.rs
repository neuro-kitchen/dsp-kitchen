//! Spike module ⇄ Slint: pushes `SpikeModule` state and the shared clusters into
//! `SpikeState`, wires `SpikeLogic`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use slint::{Color, ComponentHandle, Image, Model, ModelRc, VecModel};

use crate::app::controller::{divider_row, from_side, rgb, sync_rows, Controller};
use crate::shared::dock::{SeatBox, ViewId};
use crate::ui;

use super::module::PlotContext;
use super::plots::{self, selection_color, FeatureAxes};
use super::view::SpikeViewKind;

/// Slint-side models of the Spike module.
#[derive(Default)]
pub struct SpikeUi {
    pub images: RefCell<HashMap<ViewId, Image>>,
    pub seats: Rc<VecModel<ui::SpikeSeat>>,
    pub dividers: Rc<VecModel<ui::DividerData>>,
    pub clusters: Rc<VecModel<ui::ClusterRow>>,
}

fn to_ui_kind(kind: SpikeViewKind) -> ui::SpikeViewKind {
    match kind {
        SpikeViewKind::Waveforms => ui::SpikeViewKind::Waveforms,
        SpikeViewKind::Features => ui::SpikeViewKind::Features,
        SpikeViewKind::Correlograms => ui::SpikeViewKind::Correlograms,
        SpikeViewKind::Isi => ui::SpikeViewKind::Isi,
        SpikeViewKind::Amplitudes => ui::SpikeViewKind::Amplitudes,
    }
}

fn labels(v: Vec<(String, f32)>) -> ModelRc<ui::AxisLabel> {
    ModelRc::new(VecModel::from(v.into_iter().map(|(label, frac)| ui::AxisLabel { label: label.into(), frac }).collect::<Vec<_>>()))
}

impl Controller {
    pub(crate) fn install_spike_models(&self, ui: &ui::AppWindow) {
        let state = ui.global::<ui::SpikeState>();
        state.set_seats(ModelRc::from(self.spike_ui.seats.clone()));
        state.set_dividers(ModelRc::from(self.spike_ui.dividers.clone()));
        state.set_cluster_rows(ModelRc::from(self.spike_ui.clusters.clone()));
    }

    /// Shared inputs of the spike plots.
    pub(crate) fn with_plot_context<R>(&self, f: impl FnOnce(&PlotContext) -> R) -> R {
        let app = self.app.borrow();
        let window = self.time.borrow().window();
        f(&PlotContext { sorting: app.sorting.as_ref(), selected: &app.selected_clusters, window })
    }

    /// Pushes everything of the Spike module that can change after an intent.
    pub(crate) fn sync_spikes(&self) {
        self.sync_spike_seats();
        self.sync_spike_focused();
        self.sync_clusters();
        self.sync_spike_panels();
    }

    fn spike_seat_row(&self, seat: &SeatBox) -> ui::SpikeSeat {
        let spikes = self.spikes.borrow();
        let tabs: Vec<ui::TabData> = seat
            .views
            .iter()
            .filter_map(|&id| spikes.ws.view(id))
            .map(|v| ui::TabData { view_id: v.id as i32, title: v.title.clone().into() })
            .collect();
        let mut row = ui::SpikeSeat {
            x: seat.x,
            y: seat.y,
            width: seat.width,
            height: seat.height,
            tabs: ModelRc::new(VecModel::from(tabs)),
            active: seat.active as i32,
            view_id: -1,
            empty_message: "Empty workspace — add a view from View ▸ Add Spike View".into(),
            ..Default::default()
        };
        let Some(v) = seat.active_view().and_then(|id| spikes.ws.view(id)) else { return row };
        row.view_id = v.id as i32;
        row.kind = to_ui_kind(v.kind);
        row.focused = spikes.ws.focused == Some(v.id);
        row.image = self.spike_ui.images.borrow().get(&v.id).cloned().unwrap_or_default();
        let app = self.app.borrow();
        row.empty_message = if app.sorting.is_none() {
            "Waiting for spike sorting… (Spikes ▸ Run Sorting, Ctrl+R)".into()
        } else if app.selected_clusters.is_empty() {
            "Select a cluster in the Clusters panel".into()
        } else {
            "".into()
        };
        drop(app);
        if let Some(req) = self.with_plot_context(|ctx| spikes.plot_request(v.id, ctx)) {
            let axes = plots::axes(&req);
            row.y_labels = labels(axes.lanes);
            row.x_labels = labels(axes.ticks);
            row.note = axes.note.into();
        }
        row
    }

    pub(crate) fn sync_spike_seats(&self) {
        let seats = self.spikes.borrow().ws.layout.seats.clone();
        let rows: Vec<ui::SpikeSeat> = seats.iter().map(|s| self.spike_seat_row(s)).collect();
        if rows.len() == self.spike_ui.seats.row_count() {
            for (i, row) in rows.into_iter().enumerate() {
                self.spike_ui.seats.set_row_data(i, row);
            }
        } else {
            self.spike_ui.seats.set_vec(rows);
        }
        let dividers = self.spikes.borrow().ws.layout.dividers.iter().map(divider_row).collect();
        sync_rows(&self.spike_ui.dividers, dividers);
    }

    fn sync_spike_focused(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let spikes = self.spikes.borrow();
        let focused = match spikes.ws.focused_view() {
            Some(v) => ui::SpikeFocused {
                valid: true,
                view_id: v.id as i32,
                title: v.title.clone().into(),
                kind: to_ui_kind(v.kind),
                kind_index: SpikeViewKind::ALL.iter().position(|&k| k == v.kind).unwrap_or(0) as i32,
                feature_axes: FeatureAxes::ALL.iter().position(|&a| a == v.feature_axes).unwrap_or(0) as i32,
            },
            None => ui::SpikeFocused::default(),
        };
        let state = ui.global::<ui::SpikeState>();
        state.set_focused(focused);
        state.set_sort_k(spikes.sort_params.num_clusters as i32);
    }

    pub(crate) fn sync_clusters(&self) {
        let rows: Vec<ui::ClusterRow> = self
            .app
            .borrow()
            .cluster_rows()
            .into_iter()
            .map(|r| ui::ClusterRow {
                id: r.id as i32,
                color: r.selection_index.map_or(Color::from_argb_u8(0, 0, 0, 0), |i| rgb(selection_color(i))),
                selected: r.selection_index.is_some(),
                channel: r.peak_channel as i32,
                spikes: r.spikes as i32,
                amplitude: format!("{:.0}", r.amplitude).into(),
                rate: format!("{:.1}", r.firing_rate_hz).into(),
                snr: format!("{:.1}", r.snr).into(),
                isi: format!("{:.1}", r.isi_violation_pct).into(),
                isi_bad: r.isi_violation_pct > 1.0,
            })
            .collect();
        if rows.len() == self.spike_ui.clusters.row_count() {
            sync_rows(&self.spike_ui.clusters, rows);
        } else {
            self.spike_ui.clusters.set_vec(rows);
        }
    }

    fn sync_spike_panels(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let spikes = self.spikes.borrow();
        let state = ui.global::<ui::SpikeState>();
        state.set_show_clusters(spikes.panels.clusters);
        state.set_show_properties(spikes.panels.properties);
    }

    /// A spike view's frame arrived from the worker.
    pub(crate) fn on_spike_frame(&self, view: ViewId, image: Image) {
        self.spike_ui.images.borrow_mut().insert(view, image.clone());
        for i in 0..self.spike_ui.seats.row_count() {
            if let Some(mut row) = self.spike_ui.seats.row_data(i) {
                if row.view_id == view as i32 {
                    row.image = image.clone();
                    self.spike_ui.seats.set_row_data(i, row);
                }
            }
        }
    }

    /// The cluster selection (shared data) changed: spike plots and the traces overlay are stale.
    pub(crate) fn selection_changed(&self) {
        self.spikes.borrow_mut().mark_all_dirty();
        self.time.borrow_mut().mark_traces_dirty();
        self.sync_spikes();
    }

    pub(crate) fn wire_spikes(self: &Rc<Self>, ui: &ui::AppWindow) {
        let logic = ui.global::<ui::SpikeLogic>();

        // Runs an intent on the Spike module, then refreshes the Spike UI
        macro_rules! intent {
            ($setter:ident, |$m:ident $(, $arg:ident)*| $body:expr) => {{
                let weak = Rc::downgrade(self);
                logic.$setter(move |$($arg),*| {
                    let Some(c) = weak.upgrade() else { return };
                    {
                        #[allow(unused_mut)]
                        let mut $m = c.spikes.borrow_mut();
                        $body;
                    }
                    c.sync_spikes();
                });
            }};
        }

        intent!(on_set_num_clusters, |m, k| m.set_num_clusters(k.max(1) as usize));
        intent!(on_activate_view, |m, id| m.ws.activate(id as ViewId));
        intent!(on_close_view, |m, id| m.ws.close(id as ViewId));
        intent!(on_set_view_kind, |m, id, index| {
            if let Some(&kind) = SpikeViewKind::ALL.get(index.max(0) as usize) {
                m.set_kind(id as ViewId, kind)
            }
        });
        intent!(on_set_feature_axes, |m, id, index| {
            if let Some(&axes) = FeatureAxes::ALL.get(index.max(0) as usize) {
                m.set_feature_axes(id as ViewId, axes)
            }
        });
        intent!(on_add_view, |m, index| {
            if let Some(&kind) = SpikeViewKind::ALL.get(index.max(0) as usize) {
                m.add_view(kind);
            }
        });
        intent!(on_dock_drop, |m, view, target, side| m.ws.drop_view(view as ViewId, target as ViewId, from_side(side)));
        intent!(on_divider_pressed, |m, i| m.ws.divider_pressed(i as usize));
        intent!(on_divider_dragged, |m, i, delta| m.ws.divider_dragged(i as usize, delta));
        intent!(on_toggle_panel, |m, which| {
            if which == 0 {
                m.panels.clusters = !m.panels.clusters;
            } else {
                m.panels.properties = !m.panels.properties;
            }
        });

        {
            let weak = Rc::downgrade(self);
            logic.on_focus_view(move |id| {
                let Some(c) = weak.upgrade() else { return };
                c.spikes.borrow_mut().ws.focus(id as ViewId);
                c.sync_spikes();
                c.request_keyboard_focus();
            });
        }

        // Cluster selection is shared data (the Time module can overlay it)
        {
            let weak = Rc::downgrade(self);
            logic.on_select_cluster(move |id, additive| {
                let Some(c) = weak.upgrade() else { return };
                c.app.borrow_mut().select_cluster(id.max(0) as u32, additive);
                c.selection_changed();
                c.request_keyboard_focus();
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_step_cluster(move |delta| {
                let Some(c) = weak.upgrade() else { return };
                c.app.borrow_mut().step_cluster(delta);
                c.selection_changed();
            });
        }

        {
            let weak = Rc::downgrade(self);
            logic.on_workspace_resized(move |w, h| {
                let Some(c) = weak.upgrade() else { return };
                let scale = c.scale();
                c.spikes.borrow_mut().ws.resized(w, h, scale);
                c.sync_spikes();
            });
        }
    }
}
