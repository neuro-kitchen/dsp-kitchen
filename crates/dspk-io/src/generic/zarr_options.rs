//! How a Zarr recording is laid out and compressed: plain values, there in every build (pipeline
//! specs name them), also where Zarr stores themselves are not built (feature `zarr` off, e.g. in
//! a browser).

/// Default chunk length: ~0.17 s at 30 kHz.
pub const DEFAULT_CHUNK_SAMPLES: usize = 5000;

/// How Zarr chunks are compressed (a bytes-to-bytes codec of the array).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrCompression {
    /// Uncompressed.
    None,
    /// The Zarr v3 `zstd` codec at `level` (zstd's levels; higher is smaller and slower).
    Zstd { level: i32 },
    /// The Zarr v3 `blosc` codec: LZ4 at `level` (0–9) after a byte shuffle of the float32
    /// samples (faster than zstd, usually a little larger).
    BloscLz4 { level: u8 },
}

/// The compression of [`ZarrOptions::default`]: zstd level 3, the codec Zarr's own tools default
/// to, fast enough to keep up with the device when chunks are compressed in parallel.
pub const DEFAULT_ZARR_COMPRESSION: ZarrCompression = ZarrCompression::Zstd { level: 3 };

/// Layout of a written Zarr recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZarrOptions {
    /// Samples per chunk along time (chunks span every channel). Several chunks per block of
    /// `ZarrWriter::push` are compressed and stored in parallel.
    pub chunk_samples: usize,
    pub compression: ZarrCompression,
    /// Compress and store the completed chunks of each block on several threads (default
    /// [`DEFAULT_PARALLEL_WRITES`]); `false`: one after the other on the calling thread.
    pub parallel: bool,
}

/// Default of [`ZarrOptions::parallel`]: chunks are compressed and stored in parallel.
pub const DEFAULT_PARALLEL_WRITES: bool = true;

impl Default for ZarrOptions {
    fn default() -> Self {
        Self { chunk_samples: DEFAULT_CHUNK_SAMPLES, compression: DEFAULT_ZARR_COMPRESSION, parallel: DEFAULT_PARALLEL_WRITES }
    }
}

/// Where Zarr stores are not built: a writer that cannot be made, so code that writes one
/// compiles and reports it instead.
#[cfg(not(feature = "zarr"))]
pub struct ZarrWriter(());

#[cfg(not(feature = "zarr"))]
impl ZarrWriter {
    pub fn create(path: &std::path::Path, _channels: usize, _samples: u64, _sample_rate_hz: f64, _unit: &dspk_core::SignalUnit, _options: &ZarrOptions) -> dspk_core::DspResult<Self> {
        Err(dspk_core::DspError::UnsupportedFormat(format!("{}: Zarr stores cannot be written in this build", path.display())))
    }

    pub fn set_channel_names(&mut self, _names: &[String]) -> dspk_core::DspResult<()> {
        Ok(())
    }

    pub fn chunk_samples(&self) -> u64 {
        DEFAULT_CHUNK_SAMPLES as u64
    }

    pub fn push(&mut self, _rows: &[f32], _row_len: usize, _cols: std::ops::Range<usize>) -> dspk_core::DspResult<()> {
        Ok(())
    }

    pub fn finish(self) -> dspk_core::DspResult<()> {
        Ok(())
    }
}
