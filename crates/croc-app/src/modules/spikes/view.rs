//! State of one spike view (waveforms, features, correlograms, ISI, amplitudes).

use serde::{Deserialize, Serialize};

use crate::shared::dock::ViewId;
use crate::shared::workspace::DockView;

use super::plots::{FeatureAxes, SpikePlot};

/// Spike-module view kinds (the cluster list is a fixed panel, not a view).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpikeViewKind {
    #[default]
    Waveforms,
    Features,
    Correlograms,
    Isi,
    Amplitudes,
}

impl SpikeViewKind {
    pub const ALL: [SpikeViewKind; 5] = [Self::Waveforms, Self::Features, Self::Correlograms, Self::Isi, Self::Amplitudes];

    pub fn name(self) -> &'static str {
        match self {
            Self::Waveforms => "Waveforms",
            Self::Features => "Features",
            Self::Correlograms => "Correlograms",
            Self::Isi => "ISI",
            Self::Amplitudes => "Amplitudes",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeView {
    pub id: ViewId,
    pub kind: SpikeViewKind,
    pub title: String,
    /// Feature view: which axes to plot.
    #[serde(default)]
    pub feature_axes: FeatureAxes,
    /// Plot area in physical pixels, and physical px per logical px.
    #[serde(skip)]
    pub canvas_width: u32,
    #[serde(skip)]
    pub canvas_height: u32,
    #[serde(skip, default = "one")]
    pub scale_factor: f32,
    #[serde(skip)]
    pub needs_render: bool,
}

fn one() -> f32 {
    1.0
}

impl SpikeView {
    pub fn new(id: ViewId, kind: SpikeViewKind) -> Self {
        Self {
            id,
            kind,
            title: format!("{} {id}", kind.name()),
            feature_axes: FeatureAxes::default(),
            canvas_width: 0,
            canvas_height: 0,
            scale_factor: 1.0,
            needs_render: true,
        }
    }

    pub fn plot(&self) -> SpikePlot {
        match self.kind {
            SpikeViewKind::Waveforms => SpikePlot::Waveforms,
            SpikeViewKind::Features => SpikePlot::Features(self.feature_axes),
            SpikeViewKind::Correlograms => SpikePlot::Correlograms,
            SpikeViewKind::Isi => SpikePlot::Isi,
            SpikeViewKind::Amplitudes => SpikePlot::Amplitudes,
        }
    }
}

impl DockView for SpikeView {
    fn id(&self) -> ViewId {
        self.id
    }
    fn set_canvas(&mut self, width: u32, height: u32, scale: f32) {
        if (width, height, scale) != (self.canvas_width, self.canvas_height, self.scale_factor) {
            self.canvas_width = width;
            self.canvas_height = height;
            self.scale_factor = scale;
            self.needs_render = true;
        }
    }
    fn mark_dirty(&mut self) {
        self.needs_render = true;
    }
}
