//! The workspaces of the title bar (Blender style): each is a stage of the work on the open
//! recording, with its own arrangement of views. Explore is built; the others say what they will
//! hold (Curation is parked in `.refactoring/dspk-app/` while the app is rewired).

use gpui_kit::assets::IconName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Workspace {
    /// Traces, heatmaps, channels and the timeline: the entry point.
    #[default]
    Explore,
    Sorting,
    Pipeline,
    Curation,
    /// Charts across recordings, sortings and runs (what it holds: to be decided).
    Analytics,
}

impl Workspace {
    pub const ALL: [Workspace; 5] = [Workspace::Explore, Workspace::Sorting, Workspace::Pipeline, Workspace::Curation, Workspace::Analytics];

    pub fn title(self) -> &'static str {
        match self {
            Workspace::Explore => "Explore",
            Workspace::Sorting => "Sorting",
            Workspace::Pipeline => "Pipeline",
            Workspace::Curation => "Curation",
            Workspace::Analytics => "Analytics",
        }
    }

    /// One line for the tab's tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Workspace::Explore => "Look at the recording: traces, heatmaps, channels, timeline",
            Workspace::Sorting => "Run the installed sorters on the recording or a part of it, compare their units",
            Workspace::Pipeline => "The processing graph: recordings, operations, views and writes on a canvas",
            Workspace::Curation => "Judge a sorting: units, waveforms, features, correlograms; label and merge",
            Workspace::Analytics => "Signals and units aligned to events: averages, rasters, PETHs",
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
            Workspace::Pipeline => Some("Open a recording: its view's processing shows here as a chain of operators."),
            Workspace::Curation => Some(
                "Inspect and judge sorting results: clusters, waveforms, correlograms and amplitudes, \
                 with merges and splits recorded for the sorting files.",
            ),
            Workspace::Analytics => None,
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Workspace::Explore => IconName::AudioWaveform,
            Workspace::Sorting => IconName::SlidersHorizontal,
            Workspace::Pipeline => IconName::Workflow,
            Workspace::Curation => IconName::Microscope,
            Workspace::Analytics => IconName::ChartPie,
        }
    }
}
