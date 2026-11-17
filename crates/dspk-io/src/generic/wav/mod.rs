//! WAV (RIFF/WAVE) recordings: a header, then the samples interleaved (time-major), so a WAV
//! file is read as a raw recording laid out by its header ([`RawRecording`]): memory-mapped on
//! disk, through a [`ByteSource`] in a browser.
//!
//! Read: PCM 16- and 32-bit integers, IEEE float 32 and 64 bits, also in a
//! `WAVE_FORMAT_EXTENSIBLE` header; any channel count. Integers are scaled to ±1 full scale
//! (16-bit 32767 → ~1.0), floats are kept, the unit dimensionless. Not read: 8-bit (unsigned)
//! and 24-bit PCM, compressed WAV (µ-law, ADPCM…): the error says so. A `data` chunk declared
//! longer than the file (a recorder stopped early) is read up to the file's end.

mod format;

use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use dspk_core::{DspError, DspResult, MemoryOrder, SampleFormat, SignalUnit};

use crate::generic::raw::{ByteSource, RawParams, RawRecording};

pub use format::Wav;

/// The layout of a WAV file's samples, from its header.
#[derive(Debug, Clone, PartialEq)]
pub struct WavLayout {
    pub channels: usize,
    pub sample_rate_hz: f64,
    pub format: SampleFormat,
    /// Byte offset of the first sample (the start of the `data` chunk's payload).
    pub data_offset: u64,
    /// Samples per channel.
    pub samples: u64,
}

impl WavLayout {
    /// How [`RawRecording`] reads these samples: interleaved after the header, integers scaled
    /// to ±1 full scale.
    pub fn raw_params(&self) -> RawParams {
        let gain = match self.format {
            SampleFormat::I16 => 1.0 / 32768.0,
            SampleFormat::I32 => 1.0 / 2_147_483_648.0,
            _ => 1.0,
        };
        RawParams {
            channels: self.channels,
            sample_rate_hz: self.sample_rate_hz,
            format: self.format,
            order: MemoryOrder::TimeMajor,
            gain,
            offset: 0.0,
            unit: SignalUnit::Dimensionless,
            header_bytes: self.data_offset,
            samples: Some(self.samples),
        }
    }
}

/// Whether `path` is named like a WAV file.
pub fn is_wav_name(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".wav")
}

/// Reads a WAV header through `read_at(offset, buffer)` (fills the buffer or fails) from a file
/// of `len` bytes: walks the RIFF chunks to `fmt ` and `data`, skipping any others (`LIST`,
/// `fact`, `bext`…).
pub fn read_layout(len: u64, mut read_at: impl FnMut(u64, &mut [u8]) -> std::io::Result<()>) -> DspResult<WavLayout> {
    let bad = |m: &str| DspError::UnsupportedFormat(format!("not a WAV file we can read: {m}"));
    let io = |e: std::io::Error| DspError::Io(e.to_string());
    if len < 12 {
        return Err(bad("shorter than a RIFF header"));
    }
    let mut riff = [0u8; 12];
    read_at(0, &mut riff).map_err(io)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(bad("no RIFF/WAVE header"));
    }
    let mut fmt: Option<(u16, usize, f64, u16, Option<u16>)> = None;
    let mut at = 12u64;
    while at + 8 <= len {
        let mut head = [0u8; 8];
        read_at(at, &mut head).map_err(io)?;
        let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as u64;
        let body = at + 8;
        match &head[0..4] {
            b"fmt " => {
                if size < 16 {
                    return Err(bad("its fmt chunk is too short"));
                }
                let mut f = vec![0u8; size.min(40) as usize];
                read_at(body, &mut f).map_err(io)?;
                let u16_at = |i: usize| u16::from_le_bytes([f[i], f[i + 1]]);
                let tag = u16_at(0);
                let channels = u16_at(2) as usize;
                let rate = u32::from_le_bytes([f[4], f[5], f[6], f[7]]) as f64;
                let bits = u16_at(14);
                // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of its GUID
                let sub = (tag == 0xFFFE && f.len() >= 26).then(|| u16_at(24));
                fmt = Some((tag, channels, rate, bits, sub));
            }
            b"data" => {
                let (tag, channels, rate, bits, sub) = fmt.ok_or_else(|| bad("its data chunk comes before its fmt chunk"))?;
                let format = sample_format(sub.unwrap_or(tag), bits)?;
                if channels == 0 || rate <= 0.0 {
                    return Err(bad("no channels or no sample rate"));
                }
                let frame = (channels * format.bytes()) as u64;
                // A recorder stopped early leaves the declared size longer than the file
                let size = size.min(len - body);
                return Ok(WavLayout { channels, sample_rate_hz: rate, format, data_offset: body, samples: size / frame });
            }
            _ => {}
        }
        // Chunks are padded to an even size
        at = body + size + (size & 1);
    }
    Err(bad("no data chunk"))
}

/// The sample format of WAV format tag `tag` at `bits` per sample.
fn sample_format(tag: u16, bits: u16) -> DspResult<SampleFormat> {
    let unsupported = |what: String| DspError::UnsupportedFormat(format!("{what} WAV is not read yet: convert it to 16- or 32-bit PCM, or 32-bit float"));
    match (tag, bits) {
        (1, 16) => Ok(SampleFormat::I16),
        (1, 32) => Ok(SampleFormat::I32),
        (3, 32) => Ok(SampleFormat::F32),
        (3, 64) => Ok(SampleFormat::F64),
        (1, 8) => Err(unsupported("8-bit".into())),
        (1, 24) => Err(unsupported("24-bit".into())),
        (1 | 3, b) => Err(unsupported(format!("{b}-bit"))),
        (6, _) => Err(unsupported("A-law".into())),
        (7, _) => Err(unsupported("µ-law".into())),
        (t, _) => Err(unsupported(format!("Compressed (format tag {t:#06x})"))),
    }
}

/// The layout of the WAV file at `path`.
pub fn layout_of(path: &Path) -> DspResult<WavLayout> {
    let mut file = File::open(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
    let len = file.metadata().map_err(|e| DspError::Io(e.to_string()))?.len();
    read_layout(len, |at, buf| {
        file.seek(SeekFrom::Start(at))?;
        file.read_exact(buf)
    })
    .map_err(|e| with_name(e, path.display()))
}

/// Opens the WAV file at `path` (memory-mapped).
pub fn open_wav(path: &Path) -> DspResult<RawRecording> {
    RawRecording::open_with(path, &layout_of(path)?.raw_params())
}

/// Opens WAV file `name` read through `source` (a browser's picked file).
pub fn wav_from_source(name: impl Into<String>, source: Arc<dyn ByteSource>) -> DspResult<RawRecording> {
    let name = name.into();
    let layout = read_layout(source.len(), |at, buf| source.read_at(at, buf)).map_err(|e| with_name(e, &name))?;
    RawRecording::from_source(name, source, &layout.raw_params())
}

fn with_name(e: DspError, name: impl std::fmt::Display) -> DspError {
    match e {
        DspError::UnsupportedFormat(m) => DspError::UnsupportedFormat(format!("{name}: {m}")),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Format as _;
    use dspk_core::RecordingSource;

    /// A WAV file: `fmt` chunk of `tag` at `bits`, an optional `LIST` chunk before the data
    /// (odd-sized, so padded), then `data`.
    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8], list: bool) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        let block = channels * bits / 8;
        fmt.extend_from_slice(&(rate * block as u32).to_le_bytes());
        fmt.extend_from_slice(&block.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if tag == 0xFFFE {
            fmt.extend_from_slice(&22u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&0u32.to_le_bytes());
            // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: 3, then the GUID's rest
            fmt.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        }
        let mut body = b"WAVE".to_vec();
        let mut chunk = |id: &[u8], payload: &[u8]| {
            body.extend_from_slice(id);
            body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                body.push(0);
            }
        };
        chunk(b"fmt ", &fmt);
        if list {
            chunk(b"LIST", b"INFOx");
        }
        chunk(b"data", data);
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("dspk_io_wav_{}_{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// 16-bit PCM, two channels, a LIST chunk first: read channel-wise, scaled to ±1.
    #[test]
    fn a_16_bit_wav_reads_as_its_samples_scaled_to_full_scale() {
        let frames: [[i16; 2]; 4] = [[0, 100], [16384, -16384], [32767, -32768], [-1, 1]];
        let data: Vec<u8> = frames.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let path = write("pcm16.wav", &wav(1, 2, 44_100, 16, &data, true));
        let rec = open_wav(&path).unwrap();
        assert_eq!((rec.info().channel_count(), rec.info().samples), (2, 4));
        assert_eq!(rec.info().sample_rate_hz(), 44_100.0);
        let mut out = vec![0f32; 4];
        rec.read(&[1], 0..4, &mut out).unwrap();
        // `read` scales to the unit: ±1 full scale
        let want: Vec<f32> = frames.iter().map(|f| f[1] as f32 / 32768.0).collect();
        assert!(out.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6), "{out:?} vs {want:?}");
        assert!(Wav.detect(&path));
        let _ = std::fs::remove_file(&path);
    }

    /// 32-bit float in an extensible header, three channels; through a byte source as in a browser.
    #[test]
    fn a_float_wav_reads_through_a_byte_source_too() {
        let frames: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 0.1, -(i as f32), 0.5]).collect();
        let data: Vec<u8> = frames.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let bytes = wav(0xFFFE, 3, 30_000, 32, &data, false);
        struct Mem(Vec<u8>);
        impl ByteSource for Mem {
            fn len(&self) -> u64 {
                self.0.len() as u64
            }
            fn read_at(&self, at: u64, out: &mut [u8]) -> std::io::Result<()> {
                let at = at as usize;
                out.copy_from_slice(self.0.get(at..at + out.len()).ok_or_else(|| std::io::Error::from(std::io::ErrorKind::UnexpectedEof))?);
                Ok(())
            }
        }
        let rec = wav_from_source("tone.wav", Arc::new(Mem(bytes))).unwrap();
        assert_eq!((rec.info().channel_count(), rec.info().samples), (3, 10));
        let mut out = vec![0f32; 10];
        rec.read(&[1], 0..10, &mut out).unwrap();
        assert_eq!(out, frames.iter().map(|f| f[1]).collect::<Vec<_>>());
    }

    /// What is not read says so, and why.
    #[test]
    fn unsupported_wavs_say_what_to_convert_to() {
        let path = write("pcm24.wav", &wav(1, 1, 48_000, 24, &[0; 6], false));
        let Err(e) = open_wav(&path) else { panic!("a 24-bit WAV opened") };
        let e = e.to_string();
        assert!(e.contains("24-bit") && e.contains("convert"), "{e}");
        let _ = std::fs::remove_file(&path);
        let path = write("notwav.wav", b"RIFF\x04\x00\x00\x00AVI ");
        let Err(e) = open_wav(&path) else { panic!("an AVI opened as WAV") };
        assert!(e.to_string().contains("RIFF/WAVE"), "{e}");
        let _ = std::fs::remove_file(&path);
    }

    /// A data chunk declared longer than the file (a recorder stopped early) reads what is there.
    #[test]
    fn a_truncated_wav_reads_what_it_holds() {
        let data: Vec<u8> = [1i16, 2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut bytes = wav(1, 1, 8_000, 16, &data, false);
        let n = bytes.len();
        bytes[n - 8 - 4..n - 8].copy_from_slice(&1000u32.to_le_bytes());
        let layout = read_layout(bytes.len() as u64, |at, buf| {
            buf.copy_from_slice(&bytes[at as usize..at as usize + buf.len()]);
            Ok(())
        })
        .unwrap();
        assert_eq!(layout.samples, 4);
    }
}
