//! The workspaces of the title bar (Blender style): each is a stage of the work on the open
//! recording, with its own arrangement of views. Explore is built; the others say what they will
//! hold.

use gpui_kit::assets::IconName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Workspace {
    /// Traces, heatmaps, channels and the timeline: the entry point.
    #[default]
    Explore,
    Sorting,
    Pipeline,
    Curation,
}

impl Workspace {
    pub const ALL: [Workspace; 4] = [Workspace::Explore, Workspace::Sorting, Workspace::Pipeline, Workspace::Curation];

    pub fn title(self) -> &'static str {
        match self {
            Workspace::Explore => "Explore",
            Workspace::Sorting => "Sorting",
            Workspace::Pipeline => "Pipeline",
            Workspace::Curation => "Curation",
        }
    }

    /// One line for the tab's tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Workspace::Explore => "Look at the recording: traces, heatmaps, channels, timeline",
            Workspace::Sorting => "Define a spike-sorting run",
            Workspace::Pipeline => "Snap processing steps together in a node graph",
            Workspace::Curation => "Inspect and judge sorting results",
        }
    }

    /// What a workspace that is not built yet will hold.
    pub fn planned(self) -> Option<&'static str> {
        match self {
            Workspace::Explore => None,
            Workspace::Sorting => Some(
                "Choose the channels and the part of the recording to sort, the sorter and its parameters, \
                 then run it and follow its progress here.",
            ),
            Workspace::Pipeline => Some(
                "Build a processing pipeline as a graph: filters, re-referencing, detection and sorting \
                 steps snap together, and each step's output can be shown in a view.",
            ),
            Workspace::Curation => Some(
                "A template GUI in honour of phy: units with their waveforms, features, correlograms and \
                 amplitudes, to inspect, merge, split and label sorting results.",
            ),
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Workspace::Explore => IconName::AudioWaveform,
            Workspace::Sorting => IconName::SlidersHorizontal,
            Workspace::Pipeline => IconName::Workflow,
            Workspace::Curation => IconName::Microscope,
        }
    }
}
