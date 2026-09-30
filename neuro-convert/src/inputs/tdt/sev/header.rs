//! The 40-byte SEV header and its versions.
//!
//! ```text
//! u64 size_bytes · [u8;3] "SEV" · u8 version · [u8;4] event name · u16 channel ·
//! u16 total channels · u16 sample width (bytes) · u16 reserved · u8 data format (low 3 bits) ·
//! u8 decimate · u16 rate · padding to 40 bytes
//! ```
//! - **v0**: empty header (OpenEx < 2.18): assume float32 at 24414.0625 Hz; name and channel come
//!   from the file name.
//! - **v1, v2**: the event name in the header is unreliable (OpenEx and RS4 disagreed on byte
//!   order), so the name comes from the file name.
//! - **v3**: header name trusted.
//! - Newer versions are rejected, as in TDT's reader (`file_version < 4`).
//!
//! Sample rate = `2^(rate − 12) × 25 MHz / decimate`.

use crate::error::{Error, Result};

pub const HEADER_BYTES: usize = 40;
/// Assumed rate of headerless (v0) files, as in TDT's reader.
pub const V0_SAMPLE_RATE: f64 = 24_414.0625;

/// Data format code 8: RS4 single-unit (high 16 bits) + LFP (low 16 bits) in one int32 word.
/// The header's format field is masked to 3 bits (as TDT's reader does), so this code never
/// comes from a SEV header; the block's `.Tbk` (`DataFormat=8`) declares it.

#[derive(Debug, Clone, PartialEq)]
pub struct SevHeader {
    pub version: u8,
    /// From the header (v3+) — otherwise take it from the file name.
    pub event_name: Option<String>,
    /// From the header (v1+).
    pub channel: Option<u16>,
    pub total_channels: Option<u16>,
    /// TDT data format code (0 f32, 1 i32, 2 i16, 3 i8, 4 f64, 5 i64).
    pub format: u8,
    pub sample_rate: f64,
}

impl SevHeader {
    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < HEADER_BYTES {
            return Err(Error::format("tdt-sev", "file shorter than the 40-byte header"));
        }
        let version = b[11];
        if &b[8..11] != b"SEV" || version == 0 {
            // Headerless (v0): nothing in the header can be trusted
            return Ok(Self { version: 0, event_name: None, channel: None, total_channels: None, format: 0, sample_rate: V0_SAMPLE_RATE });
        }
        if version > 3 {
            return Err(Error::Unsupported(format!("SEV header version {version}")));
        }
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        let name: String = b[12..16].iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
        // Format code: low 3 bits, exactly like TDT's reader
        let format = b[24] & 0b111;
        let decimate = b[25].max(1);
        let rate = u16_at(26);
        Ok(Self {
            version,
            event_name: (version >= 3 && !name.is_empty()).then_some(name),
            channel: Some(u16_at(16)),
            total_channels: Some(u16_at(18)),
            format,
            sample_rate: 2f64.powf(rate as f64 - 12.0) * 25_000_000.0 / decimate as f64,
        })
    }

    /// Bytes per stored item.
    pub fn item_bytes(&self) -> Option<usize> {
        crate::inputs::tdt::codes::sample_type(self.format as u32).map(|t| t.bytes())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Encodes a header like Synapse / RS4 write it.
    pub fn header(version: u8, name: &[u8; 4], chan: u16, total: u16, width: u16, format: u8, decimate: u8, rate: u16) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_BYTES];
        b[8..11].copy_from_slice(b"SEV");
        b[11] = version;
        b[12..16].copy_from_slice(name);
        b[16..18].copy_from_slice(&chan.to_le_bytes());
        b[18..20].copy_from_slice(&total.to_le_bytes());
        b[20..22].copy_from_slice(&width.to_le_bytes());
        b[24] = format;
        b[25] = decimate;
        b[26..28].copy_from_slice(&rate.to_le_bytes());
        b
    }

    #[test]
    fn test_versions_and_rate() {
        // rate code 2, decimate 1: 2^-10 * 25e6 = 24414.0625 Hz
        let h = SevHeader::parse(&header(3, b"Wav1", 3, 16, 4, 0, 1, 2)).unwrap();
        assert_eq!((h.event_name.as_deref(), h.channel, h.total_channels), (Some("Wav1"), Some(3), Some(16)));
        assert!((h.sample_rate - 24_414.0625).abs() < 1e-9);
        assert_eq!(h.item_bytes(), Some(4));

        // v2: header name ignored
        assert_eq!(SevHeader::parse(&header(2, b"1vaW", 1, 1, 2, 2, 1, 12)).unwrap().event_name, None);
        // v0: empty header
        let h0 = SevHeader::parse(&[0u8; 40]).unwrap();
        assert_eq!((h0.version, h0.sample_rate, h0.format), (0, V0_SAMPLE_RATE, 0));
        // Format byte masked to 3 bits (8 → 0, float32), as TDT's reader does
        assert_eq!(SevHeader::parse(&header(3, b"RSn1", 1, 1, 4, 8, 1, 2)).unwrap().format, 0);
        assert!(SevHeader::parse(&header(4, b"Wav1", 1, 1, 4, 0, 1, 2)).is_err());
        assert!(SevHeader::parse(&header(9, b"Wav1", 1, 1, 4, 0, 1, 12)).is_err());
    }
}
