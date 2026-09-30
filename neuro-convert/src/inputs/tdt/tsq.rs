//! The TSQ event index: 40-byte records, grouped into per-store indexes.
//!
//! Record layout (little-endian): `size: i32` (32-bit words incl. this header), `evtype: i32`,
//! `name: [u8; 4]`, `chan: u16`, `sortcode: u16`, `timestamp: f64` (Unix seconds),
//! `offset_or_value: u64` (TEV byte offset, or the value itself for epocs / scalars),
//! `format: i32`, `frequency: f32`.
//!
//! Record 0 is a file header; the block start marker follows (code 1 in the name field) and
//! the stop marker (code 2) is the last record — missing when a block did not end cleanly.

use std::collections::BTreeMap;

use super::codes::{self, StoreKind};

pub const RECORD_BYTES: usize = 40;
/// Header words at the start of every data packet.
const HEADER_WORDS: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Record {
    pub size_words: u32,
    pub evtype: u32,
    pub name: [u8; 4],
    pub chan: u16,
    pub sortcode: u16,
    pub timestamp: f64,
    pub offset: u64,
    pub format: u32,
    pub frequency: f32,
    /// Position among records with a non-empty name field (TDT drops the others as bad
    /// headers); `.SortResult` files are indexed by it.
    pub seq: u64,
}

impl Record {
    pub fn parse(b: &[u8]) -> Self {
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        Self {
            size_words: u32_at(0),
            evtype: u32_at(4),
            name: b[8..12].try_into().unwrap(),
            chan: u16::from_le_bytes([b[12], b[13]]),
            sortcode: u16::from_le_bytes([b[14], b[15]]),
            timestamp: f64::from_le_bytes(b[16..24].try_into().unwrap()),
            offset: u64::from_le_bytes(b[24..32].try_into().unwrap()),
            format: u32_at(32),
            frequency: f32::from_le_bytes(b[36..40].try_into().unwrap()),
            seq: 0,
        }
    }

    /// Payload bytes following the 40-byte packet header in the TEV.
    pub fn data_bytes(&self) -> u64 {
        self.size_words.saturating_sub(HEADER_WORDS) as u64 * 4
    }

    /// Epoc / scalar value stored in the offset field.
    pub fn value(&self) -> f64 {
        f64::from_bits(self.offset)
    }
}

/// TDT hardware timestamps tick at this rate (RZ/RX base clock).
pub const DEVICE_CLOCK_HZ: f64 = 195_312.5;

/// Seconds from the block start, snapped to the device clock like TDT's own reader.
pub fn session_time(timestamp: f64, block_start: f64) -> f64 {
    let ticks = ((timestamp - block_start) * DEVICE_CLOCK_HZ * 1e9).round() / 1e9;
    ticks.round() / DEVICE_CLOCK_HZ
}

/// Store names are 4 bytes; unused bytes are NUL.
pub fn store_name(name: &[u8; 4]) -> String {
    name.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect()
}

/// Everything the TSQ says about one store.
#[derive(Debug, Clone)]
pub struct StoreIndex {
    pub name: String,
    pub kind: StoreKind,
    pub evtype: u32,
    pub format: u32,
    pub frequency: f64,
    /// Payload bytes per packet (streams, snips).
    pub packet_bytes: u64,
    /// Streams: TEV offsets of each channel's packets, keyed by 1-based channel.
    pub packets: BTreeMap<u16, Vec<u64>>,
    /// Streams: timestamp of each channel's first packet.
    pub first_timestamp: BTreeMap<u16, f64>,
    /// Snips, epocs, scalars: every record in file order.
    pub records: Vec<Record>,
}

/// The parsed index of a block.
#[derive(Debug, Clone, Default)]
pub struct TsqIndex {
    /// Unix time of the block start marker.
    pub start: f64,
    pub stop: Option<f64>,
    pub stores: BTreeMap<String, StoreIndex>,
    pub record_count: usize,
    pub warnings: Vec<String>,
}

impl TsqIndex {
    pub fn parse(bytes: &[u8]) -> Self {
        let mut idx = TsqIndex::default();
        let whole = bytes.len() / RECORD_BYTES;
        if bytes.len() % RECORD_BYTES != 0 {
            idx.warnings.push(format!("TSQ ends with a partial record ({} stray bytes ignored)", bytes.len() % RECORD_BYTES));
        }
        idx.record_count = whole;
        let mut invalid = 0usize;
        let mut start_seen = false;
        let mut seq = 0u64;

        for i in 0..whole {
            let mut r = Record::parse(&bytes[i * RECORD_BYTES..(i + 1) * RECORD_BYTES]);
            let code = u32::from_le_bytes(r.name);
            if code != 0 {
                r.seq = seq;
                seq += 1;
            }
            if r.evtype == codes::EVTYPE_MARK && code == codes::EVMARK_STARTBLOCK {
                idx.start = r.timestamp;
                start_seen = true;
                continue;
            }
            if r.evtype == codes::EVTYPE_MARK && code == codes::EVMARK_STOPBLOCK {
                idx.stop = Some(r.timestamp);
                continue;
            }
            if r.evtype == codes::EVTYPE_UNKNOWN {
                // File header (record 0) or padding
                continue;
            }
            let Some(kind) = StoreKind::from_evtype(r.evtype).filter(|_| r.evtype & codes::EVTYPE_INVALID_MASK == 0) else {
                invalid += 1;
                continue;
            };

            let name = store_name(&r.name);
            let store = idx.stores.entry(name.clone()).or_insert_with(|| StoreIndex {
                name,
                kind,
                evtype: r.evtype,
                format: r.format,
                frequency: r.frequency as f64,
                packet_bytes: r.data_bytes(),
                packets: BTreeMap::new(),
                first_timestamp: BTreeMap::new(),
                records: Vec::new(),
            });
            match kind {
                StoreKind::Stream => {
                    store.first_timestamp.entry(r.chan).or_insert(r.timestamp);
                    store.packets.entry(r.chan).or_default().push(r.offset);
                }
                _ => store.records.push(r),
            }
        }

        if !start_seen {
            idx.warnings.push("no block start marker; times are relative to the first event".into());
            idx.start = idx.stores.values().flat_map(|s| s.first_timestamp.values().copied().chain(s.records.first().map(|r| r.timestamp))).fold(f64::INFINITY, f64::min);
            if !idx.start.is_finite() {
                idx.start = 0.0;
            }
        }
        if idx.stop.is_none() {
            idx.warnings.push("block did not end cleanly (no stop marker); the last packets may be incomplete".into());
        }
        if invalid > 0 {
            idx.warnings.push(format!("{invalid} TSQ records had invalid event types and were skipped"));
        }
        idx
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Encodes one TSQ record.
    pub fn record(size_words: u32, evtype: u32, name: &[u8; 4], chan: u16, sort: u16, ts: f64, offset: u64, format: u32, fs: f32) -> Vec<u8> {
        let mut b = Vec::with_capacity(RECORD_BYTES);
        b.extend(size_words.to_le_bytes());
        b.extend(evtype.to_le_bytes());
        b.extend(name);
        b.extend(chan.to_le_bytes());
        b.extend(sort.to_le_bytes());
        b.extend(ts.to_le_bytes());
        b.extend(offset.to_le_bytes());
        b.extend(format.to_le_bytes());
        b.extend(fs.to_le_bytes());
        b
    }

    #[test]
    fn test_groups_stores_and_markers() {
        let mut b = record(0, 0, &[0; 4], 0, 0, 0.0, 0, 0, 0.0);
        b.extend(record(10, codes::EVTYPE_MARK, &1u32.to_le_bytes(), 0, 0, 1000.0, 0, 0, 0.0));
        b.extend(record(12, codes::EVTYPE_STREAM, b"Wav1", 1, 0, 1000.0, 40, 0, 100.0));
        b.extend(record(12, codes::EVTYPE_STREAM, b"Wav1", 2, 0, 1000.0, 88, 0, 100.0));
        b.extend(record(10, codes::EVTYPE_STRON, b"Tick", 0, 0, 1001.0, 3f64.to_bits(), 4, 0.0));
        let idx = TsqIndex::parse(&b);
        assert_eq!(idx.start, 1000.0);
        assert!(idx.stop.is_none());
        assert_eq!(idx.warnings.len(), 1, "{:?}", idx.warnings);
        let w = &idx.stores["Wav1"];
        assert_eq!(w.kind, StoreKind::Stream);
        assert_eq!(w.packet_bytes, 8);
        assert_eq!(w.packets[&2], vec![88]);
        assert_eq!(idx.stores["Tick"].records[0].value(), 3.0);
    }
}
