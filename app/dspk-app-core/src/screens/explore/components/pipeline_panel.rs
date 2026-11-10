//! The Pipeline docked in Explore: the same graph editor as the Pipeline workspace, as a panel of
//! Explore's dock (under the focused view by default, dragged anywhere like a view), so the chain
//! and the views showing its preview sit side by side. Closing its tab hides it again.

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::dock::Panel;
use gpui_kit::{div, App, Context, Entity, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement, ParentElement as _, Render, Styled as _, WeakEntity, Window};

use crate::screens::explore::explore_screen::ExploreView;
use crate::screens::pipeline::PipelineView;

pub struct PipelinePanel {
    pipeline: Entity<PipelineView>,
    explore: WeakEntity<ExploreView>,
    focus: FocusHandle,
}

impl PipelinePanel {
    pub fn new(pipeline: Entity<PipelineView>, explore: WeakEntity<ExploreView>, cx: &mut Context<Self>) -> Self {
        Self { pipeline, explore, focus: cx.focus_handle() }
    }
}

impl Render for PipelinePanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().track_focus(&self.focus).child(self.pipeline.clone())
    }
}

impl EventEmitter<PanelEvent> for PipelinePanel {}

impl Focusable for PipelinePanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for PipelinePanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.pipeline"
    }

    /// Its tab was closed: Explore's Pipeline button follows.
    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let ex = self.explore.clone();
        cx.defer(move |cx| {
            if let Some(ex) = ex.upgrade() {
                ex.update(cx, |e, cx| e.pipeline_closed(cx));
            }
        });
    }
}

impl Panel for PipelinePanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Pipeline"
    }
}
