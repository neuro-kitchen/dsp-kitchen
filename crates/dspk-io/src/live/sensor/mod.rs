//! The minimal sensor format: how a small device (an Arduino, a microcontroller board) sends samples
//! to a dspk device. Small enough to write in ~20 lines of C; no protobuf, no TLS (so it is accepted
//! only on a Unix socket, a serial port or loopback, unless the device's configuration says
//! otherwise).
//!
//! A packet, little-endian:
//!
//! | Offset | Field | Type |
//! |---|---|---|
//! | 0 | magic `DSKF` | 4 bytes |
//! | 4 | version (1) | u8 |
//! | 5 | sample type: 1 int16, 2 int32, 3 float16, 4 float32 | u8 |
//! | 6 | channels | u16 |
//! | 8 | sequence (packets, wraps) | u32 |
//! | 12 | samples per channel in this packet | u16 |
//! | 14 | flags (bit 0: source time present) | u16 |
//! | 16 | source time, µs (0 when absent) | u64 |
//! | 24 | samples, **time-major** (as an ADC produces them) | channels × samples × size |
//! | … | CRC-32 (IEEE) of everything before | u32 |
//!
//! [`PacketReader`] finds packets in a byte stream (resynchronising on the magic after garbage),
//! checks them, and turns them into channel-major [`LiveBlock`]s. Names, units and gains come from
//! the device's configuration, never from packets.

mod stream;

use dspk_core::{DspError, DspResult, SampleFormat};

use super::{unix_now_ns, LiveBlock, LiveLayout, SourceCounters};

pub use stream::{SensorListener, SensorSource, SensorTransport};

pub const MAGIC: [u8; 4] = *b"DSKF";
pub const VERSION: u8 = 1;
/// Header bytes before the samples.
pub const HEADER_BYTES: usize = 24;
/// Bytes of the checksum after the samples.
pub const CRC_BYTES: usize = 4;
/// Flag: the packet carries the sensor's clock.
pub const FLAG_SOURCE_TIME: u16 = 1;
/// Largest packet accepted (bytes): a bound before allocating for a corrupt header.
pub const MAX_PACKET_BYTES: usize = 1 << 20;

/// The sample type codes of the format.
pub fn sample_type_code(format: SampleFormat) -> Option<u8> {
    match format {
        SampleFormat::I16 => Some(1),
        SampleFormat::I32 => Some(2),
        SampleFormat::F32 => Some(4),
        _ => None,
    }
}

fn format_of_code(code: u8) -> Option<SampleFormat> {
    match code {
        1 => Some(SampleFormat::I16),
        2 => Some(SampleFormat::I32),
        // 3 (float16) is part of the format; dspk stores it once float16 lands in dspk-core
        4 => Some(SampleFormat::F32),
        _ => None,
    }
}

/// A packet's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    pub sample_type: u8,
    pub channels: u16,
    pub sequence: u32,
    pub samples: u16,
    pub flags: u16,
    pub source_time_us: u64,
}

impl PacketHeader {
    fn parse(b: &[u8]) -> Self {
        Self {
            sample_type: b[5],
            channels: u16::from_le_bytes([b[6], b[7]]),
            sequence: u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            samples: u16::from_le_bytes([b[12], b[13]]),
            flags: u16::from_le_bytes([b[14], b[15]]),
            source_time_us: u64::from_le_bytes(b[16..24].try_into().unwrap_or([0; 8])),
        }
    }

    fn sample_bytes(&self) -> Option<usize> {
        let size = match self.sample_type {
            1 | 3 => 2,
            2 | 4 => 4,
            _ => return None,
        };
        Some(self.channels as usize * self.samples as usize * size)
    }
}

/// Writes one packet (what a sensor does; tests and the `dspk device` tools use it too).
/// `time_major` holds `channels × samples` values of `format` in time-major order.
pub fn encode_packet(format: SampleFormat, channels: u16, sequence: u32, samples: u16, source_time_us: Option<u64>, time_major: &[u8]) -> DspResult<Vec<u8>> {
    let code = sample_type_code(format).ok_or_else(|| DspError::UnsupportedFormat(format!("sensor packets carry int16, int32 or float32, not {}", format.name())))?;
    let expected = channels as usize * samples as usize * format.bytes();
    if time_major.len() != expected {
        return Err(DspError::ShapeMismatch { expected: vec![expected], actual: vec![time_major.len()] });
    }
    let mut p = Vec::with_capacity(HEADER_BYTES + expected + CRC_BYTES);
    p.extend_from_slice(&MAGIC);
    p.push(VERSION);
    p.push(code);
    p.extend_from_slice(&channels.to_le_bytes());
    p.extend_from_slice(&sequence.to_le_bytes());
    p.extend_from_slice(&samples.to_le_bytes());
    p.extend_from_slice(&(if source_time_us.is_some() { FLAG_SOURCE_TIME } else { 0 }).to_le_bytes());
    p.extend_from_slice(&source_time_us.unwrap_or(0).to_le_bytes());
    p.extend_from_slice(time_major);
    let crc = crc32(&p);
    p.extend_from_slice(&crc.to_le_bytes());
    Ok(p)
}

/// CRC-32 (IEEE 802.3, as zlib and Arduino's CRC32 libraries compute it).
pub fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc = TABLE[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

/// Finds and checks packets in a byte stream and turns them into blocks of `layout`.
pub struct PacketReader {
    layout: LiveLayout,
    buffer: Vec<u8>,
    counters: SourceCounters,
    next_sample: u64,
    last_sequence: Option<u32>,
    /// The sensor's clock: µs of the first packet, and the 32-bit wraps seen (an Arduino's
    /// `micros()` wraps every ~71 min).
    last_time_us: Option<u64>,
    time_wraps: u64,
    first_time_us: Option<u64>,
}

impl PacketReader {
    pub fn new(layout: LiveLayout) -> Self {
        Self { layout, buffer: Vec::new(), counters: SourceCounters::default(), next_sample: 0, last_sequence: None, last_time_us: None, time_wraps: 0, first_time_us: None }
    }

    pub fn layout(&self) -> &LiveLayout {
        &self.layout
    }

    pub fn counters(&self) -> SourceCounters {
        self.counters
    }

    /// Adds bytes as they arrived.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next whole, valid packet as a block; `None` until one is complete. Garbage before a
    /// magic, packets with a wrong checksum or a layout other than the configured one are counted
    /// and skipped.
    pub fn next_block(&mut self) -> Option<LiveBlock> {
        loop {
            // Resynchronise on the magic
            let start = self.buffer.windows(4).position(|w| w == MAGIC);
            match start {
                None => {
                    // Keep the last 3 bytes: a magic may be arriving in pieces
                    let keep = self.buffer.len().min(3);
                    self.buffer.drain(..self.buffer.len() - keep);
                    return None;
                }
                Some(0) => {}
                Some(n) => {
                    self.buffer.drain(..n);
                }
            }
            if self.buffer.len() < HEADER_BYTES {
                return None;
            }
            let header = PacketHeader::parse(&self.buffer[..HEADER_BYTES]);
            let Some(sample_bytes) = header.sample_bytes().filter(|&n| HEADER_BYTES + n + CRC_BYTES <= MAX_PACKET_BYTES) else {
                self.refuse_at_magic();
                continue;
            };
            if self.buffer[4] != VERSION {
                self.refuse_at_magic();
                continue;
            }
            let total = HEADER_BYTES + sample_bytes + CRC_BYTES;
            if self.buffer.len() < total {
                return None;
            }
            let body = &self.buffer[..total - CRC_BYTES];
            let crc = u32::from_le_bytes(self.buffer[total - CRC_BYTES..total].try_into().unwrap_or([0; 4]));
            if crc32(body) != crc {
                self.counters.checksum_errors += 1;
                // A false magic inside garbage, or a damaged packet: look for the next magic
                self.buffer.drain(..1);
                continue;
            }
            let fits = format_of_code(header.sample_type) == Some(self.layout.format) && header.channels as usize == self.layout.channel_count();
            if !fits {
                self.counters.refused_packets += 1;
                self.buffer.drain(..total);
                continue;
            }
            let block = self.to_block(&header, &self.buffer[HEADER_BYTES..total - CRC_BYTES].to_vec());
            self.buffer.drain(..total);
            return Some(block);
        }
    }

    fn refuse_at_magic(&mut self) {
        self.counters.refused_packets += 1;
        self.buffer.drain(..1);
    }

    fn to_block(&mut self, header: &PacketHeader, time_major: &[u8]) -> LiveBlock {
        // Packets lost: the sequence jumped
        let lost_packets = match self.last_sequence {
            Some(last) => header.sequence.wrapping_sub(last).saturating_sub(1) as u64,
            None => 0,
        };
        self.last_sequence = Some(header.sequence);
        let samples = header.samples as usize;
        let lost = lost_packets * samples as u64;
        self.counters.lost_samples += lost;
        self.next_sample += lost;

        // Channel-major
        let (channels, size) = (self.layout.channel_count(), self.layout.format.bytes());
        let mut data = vec![0u8; time_major.len()];
        for t in 0..samples {
            for c in 0..channels {
                let src = (t * channels + c) * size;
                let dst = (c * samples + t) * size;
                data[dst..dst + size].copy_from_slice(&time_major[src..src + size]);
            }
        }

        let first = self.next_sample;
        self.next_sample += samples as u64;
        let source_time_ns = if header.flags & FLAG_SOURCE_TIME != 0 {
            self.unwrap_time(header.source_time_us) * 1000
        } else {
            super::sample_time_ns(first, self.layout.sample_rate)
        };
        LiveBlock { first_sample: first, samples, source_time_ns, received_unix_ns: unix_now_ns(), lost_before: lost, data }
    }

    /// A sensor clock that wraps at 2³² µs (32-bit `micros()`), made continuous, from its first
    /// packet.
    fn unwrap_time(&mut self, us: u64) -> u64 {
        const WRAP: u64 = 1 << 32;
        if us < WRAP {
            if let Some(last) = self.last_time_us {
                // Gone backwards by more than half the range: it wrapped
                if us + WRAP / 2 < last {
                    self.time_wraps += 1;
                }
            }
            self.last_time_us = Some(us);
            let t = us + self.time_wraps * WRAP;
            t - *self.first_time_us.get_or_insert(t)
        } else {
            us - *self.first_time_us.get_or_insert(us)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::{ChannelInfo, SampleRate, SignalUnit};

    fn layout() -> LiveLayout {
        let channels = ["red", "ir"].iter().map(|n| ChannelInfo { name: n.to_string(), gain: 1.0, offset: 0.0, unit: SignalUnit::Dimensionless }).collect();
        LiveLayout { name: "ppg".into(), channels, sample_rate: SampleRate::new(400.0).unwrap(), format: SampleFormat::I32 }
    }

    /// Time-major int32 samples: channel c at t is `c * 1000 + first + t`.
    fn packet(seq: u32, first: i32, samples: u16, time: Option<u64>) -> Vec<u8> {
        let mut tm = Vec::new();
        for t in 0..samples as i32 {
            for c in 0..2 {
                tm.extend_from_slice(&(c * 1000 + first + t).to_le_bytes());
            }
        }
        encode_packet(SampleFormat::I32, 2, seq, samples, time, &tm).unwrap()
    }

    #[test]
    fn crc_matches_zlib() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn reads_packets_split_anywhere_and_through_garbage() {
        let mut stream = b"noise\x00DSK".to_vec();
        stream.extend(packet(0, 0, 10, None));
        stream.extend(b"DSKFxx garbage");
        stream.extend(packet(1, 10, 10, None));
        let mut r = PacketReader::new(layout());
        let mut blocks = Vec::new();
        for piece in stream.chunks(7) {
            r.feed(piece);
            while let Some(b) = r.next_block() {
                blocks.push(b);
            }
        }
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].first_sample, 10);
        // Channel-major now
        let v = blocks[0].to_f32(r.layout());
        assert_eq!(&v[..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&v[10..13], &[1000.0, 1001.0, 1002.0]);
        assert_eq!(r.counters().lost_samples, 0);
    }

    #[test]
    fn counts_lost_packets_and_bad_checksums() {
        let mut r = PacketReader::new(layout());
        r.feed(&packet(0, 0, 10, None));
        let mut bad = packet(1, 10, 10, None);
        let n = bad.len();
        bad[n - 10] ^= 0xff;
        r.feed(&bad);
        r.feed(&packet(3, 30, 10, None));
        let a = r.next_block().unwrap();
        let b = r.next_block().unwrap();
        assert!(r.next_block().is_none());
        assert_eq!(a.first_sample, 0);
        // Packets 1 (damaged) and 2 (never sent) are lost
        assert_eq!((b.first_sample, b.lost_before), (30, 20));
        assert_eq!(r.counters().checksum_errors, 1);
        assert_eq!(r.counters().lost_samples, 20);
    }

    #[test]
    fn refuses_another_layout() {
        let mut r = PacketReader::new(layout());
        let tm = vec![0u8; 3 * 4 * 2];
        r.feed(&encode_packet(SampleFormat::I32, 3, 0, 2, None, &tm).unwrap());
        assert!(r.next_block().is_none());
        assert_eq!(r.counters().refused_packets, 1);
    }

    #[test]
    fn a_32_bit_clock_wraps_into_a_continuous_one() {
        let mut r = PacketReader::new(layout());
        let wrap = 1u64 << 32;
        for (seq, us) in [(0u32, wrap - 1000), (1, wrap - 10), (2, 15)] {
            r.feed(&packet(seq, 0, 1, Some(us)));
        }
        let times: Vec<u64> = std::iter::from_fn(|| r.next_block()).map(|b| b.source_time_ns).collect();
        assert_eq!(times, vec![0, 990_000, 1_015_000]);
    }
}
