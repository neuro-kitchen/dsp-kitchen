//! Headerless (or fixed-header) binary recordings, memory-mapped.
//!
//! The layout comes from [`RawParams`], usually read from a JSON sidecar next to the data
//! file (`rec.bin` → `rec.meta`):
//! ```json
//! { "channels": 32, "sample_rate_hz": 30000.0, "format": "int16", "order": "time_major",
//!   "gain_uv": 0.195, "header_bytes": 0 }
//! ```
//! `format` defaults to `float32` and `order` to `channel_major` (the files written by
//! `dsp-cli generate` before `dsp-io` existed).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use dsp_core::recording::{check_read, check_read_stored};
use dsp_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate, SignalUnit};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::container::binary::codec::{decode_frames, decode_run, encode, scale_frames, select_stored};

mod format;
pub use format::Raw;

/// Layout of a raw binary file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawParams {
    pub channels: usize,
    pub sample_rate_hz: f64,
    #[serde(default = "default_format", with = "format_name")]
    pub format: SampleFormat,
    #[serde(default = "default_order", with = "order_name")]
    pub order: MemoryOrder,
    /// Stored value → µV.
    #[serde(default = "default_gain")]
    pub gain_uv: f32,
    #[serde(default)]
    pub offset_uv: f32,
    /// Bytes to skip before the first sample.
    #[serde(default)]
    pub header_bytes: u64,
    /// Samples per channel; inferred from the file size when absent.
    #[serde(default)]
    pub samples: Option<u64>,
}

fn default_format() -> SampleFormat {
    SampleFormat::F32
}
fn default_order() -> MemoryOrder {
    MemoryOrder::ChannelMajor
}
fn default_gain() -> f32 {
    1.0
}

impl RawParams {
    pub fn new(channels: usize, sample_rate_hz: f64, format: SampleFormat, order: MemoryOrder) -> Self {
        Self { channels, sample_rate_hz, format, order, gain_uv: 1.0, offset_uv: 0.0, header_bytes: 0, samples: None }
    }

    /// Sidecar path for a data file: same name with a `.meta` extension.
    pub fn sidecar_path(data_path: &Path) -> PathBuf {
        data_path.with_extension("meta")
    }

    /// Reads a JSON sidecar; `None` when it is missing or not JSON (e.g. a SpikeGLX `.meta`).
    pub fn from_sidecar(data_path: &Path) -> Option<Self> {
        let text = fs::read_to_string(Self::sidecar_path(data_path)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn write_sidecar(&self, data_path: &Path) -> DspResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        fs::write(Self::sidecar_path(data_path), json)?;
        Ok(())
    }
}

/// Memory-mapped raw binary recording.
pub struct RawRecording {
    info: RecordingInfo,
    map: Mmap,
    header: usize,
}

impl RawRecording {
    /// Opens `path` using its JSON sidecar.
    pub fn open(path: &Path) -> DspResult<Self> {
        let params = RawParams::from_sidecar(path).ok_or_else(|| {
            DspError::InvalidConfig(format!(
                "{} has no JSON sidecar ({}) describing channels, sample rate and format",
                path.display(),
                RawParams::sidecar_path(path).display()
            ))
        })?;
        Self::open_with(path, &params)
    }

    pub fn open_with(path: &Path, params: &RawParams) -> DspResult<Self> {
        if params.channels == 0 {
            return Err(DspError::InvalidConfig("channel count must be at least 1".into()));
        }
        let file = File::open(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        // SAFETY: the map is read-only; like every mmap reader we assume no other process
        // truncates the recording while it is open.
        let map = unsafe { Mmap::map(&file)? };

        let frame = (params.channels * params.format.bytes()) as u64;
        let available = (map.len() as u64).saturating_sub(params.header_bytes) / frame;
        let samples = params.samples.unwrap_or(available);
        if samples > available {
            return Err(DspError::InvalidConfig(format!(
                "{} holds {available} samples per channel but {samples} were declared",
                path.display()
            )));
        }

        let name = path.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
        let info = RecordingInfo::new(name, params.channels, samples, SampleRate::new(params.sample_rate_hz)?, params.format, params.order);
        // The sidecar's gain and offset are µV by its schema
        let mut info = info.with_gain(params.gain_uv, SignalUnit::Microvolt);
        for c in &mut info.channels {
            c.offset = params.offset_uv;
        }
        Ok(Self { info, map, header: params.header_bytes as usize })
    }

    /// Replaces the descriptor (format readers set names, per-channel gains, probe layout).
    pub fn with_info(mut self, info: RecordingInfo) -> Self {
        self.info = info;
        self
    }

    fn data(&self) -> &[u8] {
        &self.map[self.header..]
    }

    /// The stored samples exactly as mapped (after the header): `info().format` values in
    /// `info().order`, before gain / offset. Lives as long as the recording.
    pub fn stored_bytes(&self) -> &[u8] {
        let info = &self.info;
        let len = info.channels.len() * info.samples as usize * info.format.bytes();
        &self.data()[..len]
    }
}

impl RecordingSource for RawRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let (bps, nch, start) = (self.info.format.bytes(), self.info.channels.len(), samples.start as usize);
        let data = self.data();
        match self.info.order {
            MemoryOrder::ChannelMajor => {
                let total = self.info.samples as usize;
                for (dst, &ch) in out.chunks_exact_mut(n * bps).zip(channels) {
                    let base = (ch * total + start) * bps;
                    dst.copy_from_slice(&data[base..base + n * bps]);
                }
            }
            MemoryOrder::TimeMajor => {
                let frame = nch * bps;
                select_stored(&data[start * frame..(start + n) * frame], true, nch, n, channels, bps, out);
            }
        }
        Ok(())
    }

    fn read_native(&self, samples: Range<u64>, out: &mut [f32]) -> DspResult<MemoryOrder> {
        let nch = self.info.channels.len();
        if self.info.order == MemoryOrder::ChannelMajor {
            let all: Vec<usize> = (0..nch).collect();
            self.read(&all, samples, out)?;
            return Ok(MemoryOrder::ChannelMajor);
        }
        let all: Vec<usize> = (0..nch).collect();
        let n = check_read(&self.info, &all, &samples, out.len())?;
        let (fmt, frame) = (self.info.format, nch * self.info.format.bytes());
        let start = samples.start as usize;
        decode_run(fmt, &self.data()[start * frame..(start + n) * frame], out, 1.0, 0.0);
        scale_frames(&self.info, out);
        Ok(MemoryOrder::TimeMajor)
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let (fmt, bps) = (self.info.format, self.info.format.bytes());
        let nch = self.info.channels.len();
        let start = samples.start as usize;
        let data = self.data();

        match self.info.order {
            MemoryOrder::ChannelMajor => {
                let total = self.info.samples as usize;
                for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
                    let c = &self.info.channels[ch];
                    let base = (ch * total + start) * bps;
                    decode_run(fmt, &data[base..base + n * bps], dst, c.gain, c.offset);
                }
            }
            MemoryOrder::TimeMajor => {
                let frame = nch * bps;
                let selected: Vec<(usize, f32, f32)> = channels
                    .iter()
                    .map(|&ch| (ch, self.info.channels[ch].gain, self.info.channels[ch].offset))
                    .collect();
                decode_frames(fmt, &data[start * frame..(start + n) * frame], frame, &selected, n, out);
            }
        }
        Ok(())
    }
}

/// Streams `source` into a raw binary file plus JSON sidecar, `chunk_samples` at a time.
/// `params.channels`, `sample_rate_hz` and `samples` are taken from the source.
pub fn write_raw(
    source: &dyn RecordingSource,
    path: &Path,
    format: SampleFormat,
    order: MemoryOrder,
    gain_uv: f32,
    chunk_samples: usize,
    mut progress: impl FnMut(u64, u64),
) -> DspResult<RawParams> {
    let info = source.info();
    let (nch, total) = (info.channels.len(), info.samples);
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
    let channels: Vec<usize> = (0..nch).collect();
    let chunk = chunk_samples.max(1) as u64;
    let mut buf = Vec::new();
    let mut bytes = Vec::new();

    match order {
        MemoryOrder::TimeMajor => {
            let mut s0 = 0;
            while s0 < total {
                let s1 = (s0 + chunk).min(total);
                let n = (s1 - s0) as usize;
                buf.resize(nch * n, 0.0);
                source.read(&channels, s0..s1, &mut buf)?;
                bytes.clear();
                for s in 0..n {
                    for ch in 0..nch {
                        encode(format, buf[ch * n + s], gain_uv, 0.0, &mut bytes);
                    }
                }
                w.write_all(&bytes)?;
                progress(s1, total);
                s0 = s1;
            }
        }
        MemoryOrder::ChannelMajor => {
            // One channel at a time keeps memory bounded without seeking
            for ch in 0..nch {
                let mut s0 = 0;
                while s0 < total {
                    let s1 = (s0 + chunk).min(total);
                    buf.resize((s1 - s0) as usize, 0.0);
                    source.read(&[ch], s0..s1, &mut buf)?;
                    bytes.clear();
                    for &v in &buf {
                        encode(format, v, gain_uv, 0.0, &mut bytes);
                    }
                    w.write_all(&bytes)?;
                    s0 = s1;
                }
                progress((ch as u64 + 1) * total / nch as u64, total);
            }
        }
    }
    w.flush()?;

    let params = RawParams {
        samples: Some(total),
        gain_uv,
        ..RawParams::new(nch, info.sample_rate_hz(), format, order)
    };
    params.write_sidecar(path)?;
    Ok(params)
}

/// Serde helpers so sidecars use readable names (`"int16"`, `"time_major"`).
mod format_name {
    use dsp_core::SampleFormat;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(f: &SampleFormat, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(f.name())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SampleFormat, D::Error> {
        let s = String::deserialize(d)?;
        SampleFormat::parse(&s).ok_or_else(|| serde::de::Error::custom(format!("unknown sample format {s:?}")))
    }
}

mod order_name {
    use dsp_core::MemoryOrder;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(o: &MemoryOrder, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match o {
            MemoryOrder::ChannelMajor => "channel_major",
            MemoryOrder::TimeMajor => "time_major",
        })
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MemoryOrder, D::Error> {
        match String::deserialize(d)?.as_str() {
            "channel_major" | "channels_first" => Ok(MemoryOrder::ChannelMajor),
            "time_major" | "interleaved" => Ok(MemoryOrder::TimeMajor),
            other => Err(serde::de::Error::custom(format!("unknown memory order {other:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dsp_io_raw_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("rec.bin")
    }

    fn source() -> MemoryRecording {
        // 3 channels x 1000 samples, value = ch * 1000 + sample (exact in i16 with gain 1)
        let data = (0..3).flat_map(|c| (0..1000).map(move |s| (c * 1000 + s) as f32)).collect();
        MemoryRecording::new("src", data, 3, 30_000.0).unwrap()
    }

    #[test]
    fn test_roundtrip_all_orders_and_formats() {
        let src = source();
        for (i, (format, order)) in [
            (SampleFormat::F32, MemoryOrder::ChannelMajor),
            (SampleFormat::F32, MemoryOrder::TimeMajor),
            (SampleFormat::I16, MemoryOrder::ChannelMajor),
            (SampleFormat::I16, MemoryOrder::TimeMajor),
        ]
        .into_iter()
        .enumerate()
        {
            let path = temp(&format!("rt{i}"));
            write_raw(&src, &path, format, order, 1.0, 64, |_, _| {}).unwrap();
            let rec = RawRecording::open(&path).unwrap();
            assert_eq!(rec.info().samples, 1000);
            assert_eq!(rec.info().order, order);

            let mut out = vec![0.0; 2 * 10];
            rec.read(&[2, 0], 500..510, &mut out).unwrap();
            let expect: Vec<f32> = (500..510).map(|s| 2000.0 + s as f32).chain((500..510).map(|s| s as f32)).collect();
            assert_eq!(out, expect, "{format:?} {order:?}");
            fs::remove_dir_all(path.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn test_gain_header_and_size_checks() {
        let path = temp("hdr");
        // 4-byte header, then 2 channels interleaved i16: (1, -1), (2, -2)
        let mut bytes = vec![0xAA; 4];
        for v in [1i16, -1, 2, -2] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&path, &bytes).unwrap();
        let mut p = RawParams::new(2, 1000.0, SampleFormat::I16, MemoryOrder::TimeMajor);
        p.header_bytes = 4;
        p.gain_uv = 0.5;
        let rec = RawRecording::open_with(&path, &p).unwrap();
        let mut out = [0.0; 4];
        rec.read(&[0, 1], 0..2, &mut out).unwrap();
        assert_eq!(out, [0.5, 1.0, -0.5, -1.0]);

        p.samples = Some(3);
        assert!(RawRecording::open_with(&path, &p).is_err());
        // No sidecar → a clear error rather than guessed defaults
        assert!(RawRecording::open(&path).is_err());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn test_legacy_sidecar_defaults_to_f32_channel_major() {
        let path = temp("legacy");
        let values: Vec<u8> = (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        fs::write(&path, values).unwrap();
        fs::write(RawParams::sidecar_path(&path), r#"{"channels": 2, "samples": 3, "sample_rate_hz": 32000.0, "format": "float32-le"}"#).unwrap();
        let rec = RawRecording::open(&path).unwrap();
        let mut out = [0.0; 3];
        rec.read(&[1], 0..3, &mut out).unwrap();
        assert_eq!(out, [3.0, 4.0, 5.0]);
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
