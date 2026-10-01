//! mtscomp-compressed recordings (`.cbin` + `.ch`), the format of IBL's public raw data.
//!
//! The `.ch` JSON lists chunk boundaries (in samples) and byte offsets. Each chunk is an int16
//! `[samples, channels]` block, differenced along time (and optionally channels), serialized in
//! `chunk_order` (`F` = channel-major within the chunk) and zlib-compressed. Chunks decode
//! independently: a read decodes the chunks it overlaps in parallel and keeps them for the next
//! read (sequential and overlapping reads share their boundary chunks).

use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dsp_core::recording::{check_read, check_read_stored};
use dsp_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate};
use flate2::read::ZlibDecoder;
use memmap2::Mmap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ChFile {
    chunk_bounds: Vec<u64>,
    chunk_offsets: Vec<u64>,
    n_channels: usize,
    sample_rate: f64,
    dtype: String,
    #[serde(default = "yes")]
    do_time_diff: bool,
    #[serde(default)]
    do_spatial_diff: bool,
    #[serde(default = "fortran")]
    chunk_order: String,
}

fn yes() -> bool {
    true
}
fn fortran() -> String {
    "F".into()
}

/// Channel-major int16 samples of one chunk: `[channel * n + sample]`.
type Chunk = Arc<Vec<i16>>;

pub struct MtscompRecording {
    info: RecordingInfo,
    map: Mmap,
    ch: ChFile,
    /// Decoded chunks of the most recent read, as `(chunk index, samples)`.
    cache: Mutex<Vec<(usize, Chunk)>>,
}

impl MtscompRecording {
    pub fn ch_path(cbin: &Path) -> PathBuf {
        cbin.with_extension("ch")
    }

    pub fn open(cbin: &Path) -> DspResult<Self> {
        let ch_path = Self::ch_path(cbin);
        let text = std::fs::read_to_string(&ch_path).map_err(|e| DspError::Io(format!("{}: {e}", ch_path.display())))?;
        let ch: ChFile = serde_json::from_str(&text).map_err(|e| DspError::InvalidConfig(format!("{}: {e}", ch_path.display())))?;
        if SampleFormat::parse(&ch.dtype) != Some(SampleFormat::I16) {
            return Err(DspError::UnsupportedFormat(format!("mtscomp dtype {} (only int16)", ch.dtype)));
        }
        if ch.chunk_bounds.len() < 2 || ch.chunk_bounds.len() != ch.chunk_offsets.len() || ch.n_channels == 0 {
            return Err(DspError::InvalidConfig(format!("{} has inconsistent chunk tables", ch_path.display())));
        }
        let file = File::open(cbin).map_err(|e| DspError::Io(format!("{}: {e}", cbin.display())))?;
        // SAFETY: read-only map of a file we assume is not truncated while open
        let map = unsafe { Mmap::map(&file)? };
        if *ch.chunk_offsets.last().unwrap() > map.len() as u64 {
            return Err(DspError::InvalidConfig(format!("{} is shorter than its chunk table (incomplete download?)", cbin.display())));
        }

        let name = cbin.file_name().map_or_else(|| "recording.cbin".into(), |n| n.to_string_lossy().into_owned());
        let samples = *ch.chunk_bounds.last().unwrap();
        let mut info = RecordingInfo::new(name, ch.n_channels, samples, SampleRate::new(ch.sample_rate)?, SampleFormat::I16, MemoryOrder::TimeMajor);
        info.metadata.insert("compression".into(), "mtscomp".into());
        Ok(Self { info, map, ch, cache: Mutex::new(Vec::new()) })
    }

    /// Replaces the descriptor (SpikeGLX sets names, gains and geometry); keeps `samples`.
    pub fn with_info(mut self, info: RecordingInfo) -> Self {
        let samples = self.info.samples;
        self.info = RecordingInfo { samples, ..info };
        self
    }

    pub fn chunk_count(&self) -> usize {
        self.ch.chunk_bounds.len() - 1
    }

    /// Decoded chunks `range`, reusing those of the previous read and decoding the rest in
    /// parallel (at most `available_parallelism()` threads). The cache then holds `range`.
    fn chunks(&self, range: Range<usize>) -> DspResult<Vec<Chunk>> {
        let mut out: Vec<Option<Chunk>> = {
            let cache = self.cache.lock().unwrap();
            range.clone().map(|i| cache.iter().find(|(c, _)| *c == i).map(|(_, v)| v.clone())).collect()
        };
        let missing: Vec<usize> = out.iter().enumerate().filter(|(_, c)| c.is_none()).map(|(k, _)| k).collect();
        if !missing.is_empty() {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(missing.len());
            let next = AtomicUsize::new(0);
            let decoded: Vec<Mutex<Option<DspResult<Chunk>>>> = missing.iter().map(|_| Mutex::new(None)).collect();
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| loop {
                        let j = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&k) = missing.get(j) else { break };
                        *decoded[j].lock().unwrap() = Some(self.decode(range.start + k).map(Arc::new));
                    });
                }
            });
            for (&k, d) in missing.iter().zip(decoded) {
                out[k] = Some(d.into_inner().unwrap().expect("every missing chunk decoded")?);
            }
        }
        let out: Vec<Chunk> = out.into_iter().map(|c| c.expect("filled")).collect();
        *self.cache.lock().unwrap() = range.zip(out.iter().cloned()).collect();
        Ok(out)
    }

    fn decode(&self, idx: usize) -> DspResult<Vec<i16>> {
        let nch = self.ch.n_channels;
        let ns = (self.ch.chunk_bounds[idx + 1] - self.ch.chunk_bounds[idx]) as usize;
        let (o0, o1) = (self.ch.chunk_offsets[idx] as usize, self.ch.chunk_offsets[idx + 1] as usize);

        let mut bytes = Vec::with_capacity(ns * nch * 2);
        ZlibDecoder::new(&self.map[o0..o1])
            .read_to_end(&mut bytes)
            .map_err(|e| DspError::Io(format!("mtscomp chunk {idx} is corrupted: {e}")))?;
        if bytes.len() != ns * nch * 2 {
            return Err(DspError::Io(format!("mtscomp chunk {idx}: {} bytes, expected {}", bytes.len(), ns * nch * 2)));
        }
        let stored = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]));

        // To channel-major
        let mut x: Vec<i16> = if self.ch.chunk_order == "F" {
            stored.collect()
        } else {
            let rows: Vec<i16> = stored.collect();
            let mut t = vec![0i16; rows.len()];
            for s in 0..ns {
                for c in 0..nch {
                    t[c * ns + s] = rows[s * nch + c];
                }
            }
            t
        };

        // Undo the differencing in reverse order (channels, then time); numpy wraps int16
        if self.ch.do_spatial_diff {
            for c in 1..nch {
                for s in 0..ns {
                    x[c * ns + s] = x[c * ns + s].wrapping_add(x[(c - 1) * ns + s]);
                }
            }
        }
        if self.ch.do_time_diff {
            for row in x.chunks_exact_mut(ns.max(1)) {
                for s in 1..ns {
                    row[s] = row[s].wrapping_add(row[s - 1]);
                }
            }
        }
        Ok(x)
    }
}

impl RecordingSource for MtscompRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    /// The first chunk's length (mtscomp chunks are equal except the last).
    fn chunk_samples(&self) -> Option<u64> {
        Some(self.ch.chunk_bounds[1] - self.ch.chunk_bounds[0])
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let bounds = &self.ch.chunk_bounds;
        let first = bounds.partition_point(|&b| b <= samples.start) - 1;
        let last = bounds.partition_point(|&b| b < samples.end) - 1;
        let chunks = self.chunks(first..last + 1)?;
        for (idx, chunk) in (first..=last).zip(&chunks) {
            let (b0, b1) = (bounds[idx], bounds[idx + 1]);
            let ns = (b1 - b0) as usize;
            let (s0, s1) = (samples.start.max(b0), samples.end.min(b1));
            let (src0, len, dst0) = ((s0 - b0) as usize, (s1 - s0) as usize, (s0 - samples.start) as usize);
            for (dst, &ch) in out.chunks_exact_mut(n * 2).zip(channels) {
                let src = &chunk[ch * ns + src0..ch * ns + src0 + len];
                for (o, &v) in dst[dst0 * 2..(dst0 + len) * 2].chunks_exact_mut(2).zip(src) {
                    o.copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let bounds = &self.ch.chunk_bounds;
        let first = bounds.partition_point(|&b| b <= samples.start) - 1;
        let last = bounds.partition_point(|&b| b < samples.end) - 1;
        let chunks = self.chunks(first..last + 1)?;
        for (idx, chunk) in (first..=last).zip(&chunks) {
            let (b0, b1) = (bounds[idx], bounds[idx + 1]);
            let ns = (b1 - b0) as usize;
            let (s0, s1) = (samples.start.max(b0), samples.end.min(b1));
            let (src0, len, dst0) = ((s0 - b0) as usize, (s1 - s0) as usize, (s0 - samples.start) as usize);
            for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
                let c = &self.info.channels[ch];
                let src = &chunk[ch * ns + src0..ch * ns + src0 + len];
                for (o, &v) in dst[dst0..dst0 + len].iter_mut().zip(src) {
                    *o = v as f32 * c.gain_uv + c.offset_uv;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    /// Writes a `.cbin`/`.ch` pair the way mtscomp does (time diff, `F` order, zlib).
    fn write_cbin(path: &Path, data: &[Vec<i16>], chunk: usize, spatial: bool) {
        let (nch, ns) = (data.len(), data[0].len());
        let mut bounds = vec![0u64];
        let mut offsets = vec![0u64];
        let mut out = Vec::new();
        let mut b0 = 0;
        while b0 < ns {
            let b1 = (b0 + chunk).min(ns);
            let mut x: Vec<Vec<i16>> = data.iter().map(|c| c[b0..b1].to_vec()).collect();
            for c in &mut x {
                for s in (1..c.len()).rev() {
                    c[s] = c[s].wrapping_sub(c[s - 1]);
                }
            }
            if spatial {
                for c in (1..nch).rev() {
                    for s in 0..x[c].len() {
                        x[c][s] = x[c][s].wrapping_sub(x[c - 1][s]);
                    }
                }
            }
            let raw: Vec<u8> = x.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&raw).unwrap();
            out.extend(enc.finish().unwrap());
            bounds.push(b1 as u64);
            offsets.push(out.len() as u64);
            b0 = b1;
        }
        std::fs::write(path, out).unwrap();
        let ch = serde_json::json!({
            "version": "1.0", "algorithm": "zlib", "cast": "", "comp_level": -1,
            "chunk_bounds": bounds, "chunk_offsets": offsets, "chunk_order": "F",
            "do_time_diff": true, "do_spatial_diff": spatial, "dtype": "int16",
            "n_channels": nch, "sample_rate": 30000.0, "shape": [ns, nch],
        });
        std::fs::write(MtscompRecording::ch_path(path), ch.to_string()).unwrap();
    }

    #[test]
    fn test_decodes_across_chunks_with_wrapping() {
        let dir = std::env::temp_dir().join(format!("dsp_io_mts_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Large steps force int16 wrap-around in the differences
        let data: Vec<Vec<i16>> = (0..3)
            .map(|c| (0..1000).map(|s| if (s / 7 + c) % 2 == 0 { 32000 } else { -32000 } as i16).collect())
            .collect();
        for spatial in [false, true] {
            let path = dir.join(format!("rec_{spatial}.cbin"));
            write_cbin(&path, &data, 300, spatial);
            let rec = MtscompRecording::open(&path).unwrap();
            assert_eq!(rec.chunk_count(), 4);
            assert_eq!(rec.info().samples, 1000);
            let mut out = vec![0.0; 2 * 400];
            rec.read(&[2, 0], 250..650, &mut out).unwrap();
            let expect: Vec<f32> = [2usize, 0].iter().flat_map(|&c| data[c][250..650].iter().map(|&v| v as f32)).collect();
            assert_eq!(out, expect, "spatial diff {spatial}");

            // Every chunk at once (parallel decode), then windows that reuse the previous read's
            // chunks, and a read ending exactly on a chunk boundary
            for (range, chans) in [(0u64..1000u64, vec![0usize, 1, 2]), (590..910, vec![1]), (600..900, vec![2, 1]), (0..300, vec![0])] {
                let n = (range.end - range.start) as usize;
                let mut out = vec![0.0; chans.len() * n];
                rec.read(&chans, range.clone(), &mut out).unwrap();
                let expect: Vec<f32> =
                    chans.iter().flat_map(|&c| data[c][range.start as usize..range.end as usize].iter().map(|&v| v as f32)).collect();
                assert_eq!(out, expect, "{range:?} spatial diff {spatial}");
                crate::tests::assert_stored_matches(&rec, &chans, range.clone());
            }
            assert_eq!(rec.cache.lock().unwrap().iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0], "cache holds the last window");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
