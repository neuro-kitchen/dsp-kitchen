//! TSQ event-type and data-format codes (TDT `TTank` constants).

use crate::model::SampleType;

pub const EVTYPE_UNKNOWN: u32 = 0x0000_0000;
pub const EVTYPE_STRON: u32 = 0x0000_0101;
pub const EVTYPE_STROFF: u32 = 0x0000_0102;
pub const EVTYPE_SCALAR: u32 = 0x0000_0201;
pub const EVTYPE_STREAM: u32 = 0x0000_8101;
pub const EVTYPE_SNIP: u32 = 0x0000_8201;
pub const EVTYPE_MARK: u32 = 0x0000_8801;
/// Set on streams whose values were stored unscaled ("use channel factors").
pub const EVTYPE_UCF: u32 = 0x0000_0010;
pub const EVTYPE_MASK: u32 = 0x0000_FF0F;
pub const EVTYPE_INVALID_MASK: u32 = 0xFFFF_0000;

/// Codes stored in the name field of the block start / stop markers.
pub const EVMARK_STARTBLOCK: u32 = 0x0001;
pub const EVMARK_STOPBLOCK: u32 = 0x0002;

/// What a store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    /// Continuous multi-channel signal in fixed-size packets.
    Stream,
    /// Triggered waveform snippets.
    Snip,
    /// Strobe onsets (epocs) and marks.
    EpocOnset,
    /// Strobe offsets, stored as a separate store whose name ends in `\`.
    EpocOffset,
    /// Timestamped single or multi-channel values.
    Scalar,
}

impl StoreKind {
    pub fn from_evtype(evtype: u32) -> Option<Self> {
        match evtype {
            EVTYPE_STRON | EVTYPE_MARK => Some(StoreKind::EpocOnset),
            EVTYPE_STROFF => Some(StoreKind::EpocOffset),
            EVTYPE_SCALAR => Some(StoreKind::Scalar),
            EVTYPE_SNIP => Some(StoreKind::Snip),
            t if t & EVTYPE_MASK == EVTYPE_STREAM => Some(StoreKind::Stream),
            _ => None,
        }
    }
}

/// TDT data-format code → sample type. Code 8 ("rawpacked", RS4 single-unit + LFP packing) is
/// not supported yet.
pub fn sample_type(code: u32) -> Option<SampleType> {
    match code {
        0 => Some(SampleType::F32),
        1 => Some(SampleType::I32),
        2 => Some(SampleType::I16),
        3 => Some(SampleType::I8),
        4 => Some(SampleType::F64),
        5 => Some(SampleType::I64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kinds_and_formats() {
        assert_eq!(StoreKind::from_evtype(0x8101), Some(StoreKind::Stream));
        // Unscaled streams keep the stream bits under the mask
        assert_eq!(StoreKind::from_evtype(0x8101 | EVTYPE_UCF), Some(StoreKind::Stream));
        assert_eq!(StoreKind::from_evtype(0x102), Some(StoreKind::EpocOffset));
        assert_eq!(StoreKind::from_evtype(0x8801), Some(StoreKind::EpocOnset));
        assert_eq!(StoreKind::from_evtype(0), None);
        assert_eq!(sample_type(2), Some(SampleType::I16));
        assert_eq!(sample_type(8), None);
    }
}
