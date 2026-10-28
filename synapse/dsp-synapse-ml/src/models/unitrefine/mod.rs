//! SpikeInterface UnitRefine & Bombcell automated unit curation family.

pub mod classifier;

pub use classifier::{
    UNITREFINE_BOMBCELL_MODEL_ID, UNITREFINE_CURATION_MODEL_ID, UnitCurationResult,
    UnitRefineClassifier,
};
