//! Spike module view model: the sorting widgets' dock, their settings, the module's panels
//! and the sorting parameters. The sorting result and cluster selection are shared data
//! owned by the parent app.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::shared::dock::{Dock, DropSide, ViewId};
use crate::shared::render_worker::RenderJob;
use crate::shared::workspace::DockWorkspace;

use super::plots::{self, FeatureAxes, SpikePlotRequest};
use super::sorting::{SortParams, Sorting};
use super::view::{SpikeView, SpikeViewKind};

/// Render key namespace of this module.
pub const MODULE_ID: u8 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikePanels {
    pub clusters: bool,
    pub properties: bool,
}

impl Default for SpikePanels {
    fn default() -> Self {
        Self { clusters: true, properties: true }
    }
}

/// Inputs every spike plot needs from the shared data.
pub struct PlotContext<'a> {
    pub sorting: Option<&'a Arc<Sorting>>,
    pub selected: &'a [u32],
    /// Time module's window (seconds), shaded in the amplitude view.
    pub window: (f64, f64),
}

pub struct SpikeModule {
    pub ws: DockWorkspace<SpikeView>,
    pub panels: SpikePanels,
    pub sort_params: SortParams,
}

impl SpikeModule {
    pub fn new(channels: usize) -> Self {
        let mut m = Self {
            ws: DockWorkspace::new(Vec::new(), Dock::empty(), None),
            panels: SpikePanels::default(),
            sort_params: SortParams::for_channels(channels),
        };
        m.reset_layout();
        m
    }

    /// Default arrangement:
    /// | waveforms    | features                 |
    /// | correlograms | ISI | amplitudes         |
    pub fn reset_layout(&mut self) {
        use SpikeViewKind::*;
        let views: Vec<SpikeView> =
            [Waveforms, Features, Correlograms, Isi, Amplitudes].iter().enumerate().map(|(i, &k)| SpikeView::new(i as ViewId + 1, k)).collect();
        let split = |columns: bool, ratio: f32, first: Dock, second: Dock| Dock::Split {
            columns,
            ratio,
            first: Box::new(first),
            second: Box::new(second),
        };
        let top = split(true, 0.45, Dock::single(1), Dock::single(2));
        let bottom = split(true, 0.36, Dock::single(3), split(true, 0.4, Dock::single(4), Dock::single(5)));
        self.ws.reset(views, split(false, 0.55, top, bottom), Some(1));
    }

    pub fn add_view(&mut self, kind: SpikeViewKind) -> ViewId {
        let v = SpikeView::new(self.ws.next_id(), kind);
        self.ws.add(v, DropSide::Right)
    }

    pub fn set_kind(&mut self, id: ViewId, kind: SpikeViewKind) {
        if let Some(v) = self.ws.view_mut(id) {
            if v.kind != kind {
                v.kind = kind;
                v.title = format!("{} {}", kind.name(), v.id);
                v.needs_render = true;
            }
        }
    }

    pub fn set_feature_axes(&mut self, id: ViewId, axes: FeatureAxes) {
        if let Some(v) = self.ws.view_mut(id) {
            v.feature_axes = axes;
            v.needs_render = true;
        }
    }

    pub fn set_num_clusters(&mut self, k: usize) {
        self.sort_params.num_clusters = k.clamp(1, 64);
    }

    /// Everything depends on the sorting and the selection.
    pub fn mark_all_dirty(&mut self) {
        self.ws.mark_all_dirty();
    }

    /// Only the amplitude view shows the time window.
    pub fn window_changed(&mut self) {
        for v in self.ws.views.iter_mut().filter(|v| v.kind == SpikeViewKind::Amplitudes) {
            v.needs_render = true;
        }
    }

    /// Plot request for a view (the same parameters its frame is rendered with).
    pub fn plot_request(&self, id: ViewId, ctx: &PlotContext) -> Option<SpikePlotRequest> {
        let v = self.ws.view(id)?;
        Some(SpikePlotRequest {
            plot: v.plot(),
            width: v.canvas_width.max(1),
            height: v.canvas_height.max(1),
            scale: v.scale_factor,
            sorting: ctx.sorting?.clone(),
            selected: ctx.selected.to_vec(),
            window: ctx.window,
        })
    }

    /// Render jobs for visible, stale views; views wait (stay stale) until a sorting exists.
    pub fn take_jobs(&mut self, ctx: &PlotContext) -> Vec<RenderJob> {
        if ctx.sorting.is_none() {
            return Vec::new();
        }
        let visible = self.ws.visible();
        let stale: Vec<ViewId> = self.ws.views.iter().filter(|v| v.needs_render && visible.contains(&v.id)).map(|v| v.id).collect();
        let mut out = Vec::new();
        for id in stale {
            let Some(req) = self.plot_request(id, ctx) else { continue };
            self.ws.view_mut(id).expect("listed").needs_render = false;
            out.push(RenderJob { key: (MODULE_ID, id), samples_per_px: None, render: Box::new(move || plots::render(&req)) });
        }
        out
    }

    pub fn session(&self) -> SpikeSession {
        SpikeSession { workspace: self.ws.clone(), panels: self.panels.clone(), sort_params: self.sort_params.clone() }
    }

    pub fn restore(&mut self, session: SpikeSession) {
        if session.workspace.is_consistent() {
            self.ws.adopt(session.workspace);
        }
        self.panels = session.panels;
        self.sort_params = session.sort_params;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeSession {
    pub workspace: DockWorkspace<SpikeView>,
    pub panels: SpikePanels,
    pub sort_params: SortParams,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::modules::spikes::sorting::tests::synthetic_params;

    #[test]
    fn test_layout_and_jobs_wait_for_sorting() {
        let mut m = SpikeModule::new(8);
        m.ws.resized(1100.0, 700.0, 1.0);
        assert_eq!(m.ws.layout.seats.len(), 5);
        let none = PlotContext { sorting: None, selected: &[], window: (0.0, 0.1) };
        assert!(m.take_jobs(&none).is_empty());

        let ds = Dataset::generate_synthetic(8, 30_000.0, 2.0);
        let sorting = Arc::new(Sorting::run(&ds, &synthetic_params(8)));
        let selected = [sorting.clusters[0].id];
        let ctx = PlotContext { sorting: Some(&sorting), selected: &selected, window: (0.0, 0.1) };
        let jobs = m.take_jobs(&ctx);
        assert_eq!(jobs.len(), 5);
        assert!(jobs.iter().all(|j| j.key.0 == MODULE_ID));
        assert!(m.take_jobs(&ctx).is_empty());

        // Window change only affects the amplitude view
        m.window_changed();
        assert_eq!(m.take_jobs(&ctx).len(), 1);
    }

    #[test]
    fn test_add_kind_and_session() {
        let mut m = SpikeModule::new(8);
        m.ws.resized(1100.0, 700.0, 1.0);
        let id = m.add_view(SpikeViewKind::Features);
        m.set_feature_axes(id, FeatureAxes::DepthPc1);
        m.set_kind(1, SpikeViewKind::Isi);
        m.panels.clusters = false;
        m.set_num_clusters(7);
        let json = serde_json::to_string(&m.session()).unwrap();

        let mut b = SpikeModule::new(8);
        b.restore(serde_json::from_str(&json).unwrap());
        assert_eq!(b.ws.dock, m.ws.dock);
        assert_eq!(b.ws.view(id).unwrap().feature_axes, FeatureAxes::DepthPc1);
        assert_eq!(b.ws.view(1).unwrap().kind, SpikeViewKind::Isi);
        assert!(!b.panels.clusters);
        assert_eq!(b.sort_params.num_clusters, 7);
    }
}
