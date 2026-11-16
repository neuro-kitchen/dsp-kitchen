//! The unit table is `dspk_synapse::curation::table::UnitTable` (a Polars data frame). This file
//! keeps how the Curation view shows it.

pub use dspk_synapse::curation::table::UnitTable;

/// Columns, in display order: (key in the frame, header, width in logical px, decimals).
pub const COLUMNS: [(&str, &str, f32, usize); 9] = [
    ("unit", "Unit", 64., 0),
    ("label", "Label", 72., 0),
    ("spikes", "Spikes", 64., 0),
    ("rate", "Rate", 56., 2),
    ("snr", "SNR", 48., 1),
    ("isi", "ISI viol.", 64., 3),
    ("presence", "Presence", 64., 2),
    ("amp_cut", "Amp. cut", 64., 3),
    ("score", "Score", 48., 2),
];
