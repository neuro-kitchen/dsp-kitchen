//! Recording file formats for `dsp-kitchen`.
//!
//! Every format implements [`dsp_core::RecordingSource`], so callers read bounded chunks of any
//! recording without knowing how it is stored. [`open`] picks the reader from the path.

mod codec;
pub mod mtscomp;
#[cfg(feature = "zarr")]
pub mod nwb;
pub mod raw;
pub mod spikeglx;
pub mod synthetic;
#[cfg(feature = "zarr")]
pub mod zarr;

use std::path::Path;

use dsp_core::{DspError, DspResult, RecordingSource};

pub use mtscomp::MtscompRecording;
pub use raw::{write_raw, RawParams, RawRecording};
pub use spikeglx::SpikeGlxMeta;
pub use synthetic::{SyntheticParams, SyntheticRecording};
#[cfg(feature = "zarr")]
pub use nwb::NwbZarrRecording;
#[cfg(feature = "zarr")]
pub use zarr::{write_zarr, ZarrRecording};

/// Opens a recording, detecting its format from the path:
/// - an NWB file stored as Zarr (`.nwb.zarr`; its largest electrical series),
/// - a Zarr v3 store with a `/traces` array (directory with `zarr.json`, or a `.zarr` path),
/// - SpikeGLX (`.bin` + `key=value` `.meta`), including IBL mtscomp `.cbin` + `.ch`,
/// - a raw binary file with a JSON sidecar (`rec.bin` + `rec.meta`).
pub fn open(path: &Path) -> DspResult<Box<dyn RecordingSource>> {
    if !path.exists() {
        return Err(DspError::Io(format!("{} does not exist", path.display())));
    }
    if path.is_dir() || path.extension().is_some_and(|e| e == "zarr") {
        #[cfg(feature = "zarr")]
        if path.join("zarr.json").exists() {
            if nwb::is_nwb_zarr(path) {
                return Ok(Box::new(NwbZarrRecording::open(path)?));
            }
            if !path.join("traces").join("zarr.json").exists() {
                return Err(DspError::UnsupportedFormat(format!(
                    "{} is a Zarr store but neither an NWB file nor a /traces recording",
                    path.display()
                )));
            }
            return Ok(Box::new(ZarrRecording::open(path)?));
        }
        return Err(DspError::UnsupportedFormat(format!("{} is a folder but not a Zarr v3 store", path.display())));
    }
    if SpikeGlxMeta::is_spikeglx(path) {
        return spikeglx::open(path);
    }
    if path.extension().is_some_and(|e| e == "cbin") {
        return Ok(Box::new(MtscompRecording::open(path)?));
    }
    if RawParams::from_sidecar(path).is_some() {
        return Ok(Box::new(RawRecording::open(path)?));
    }
    Err(DspError::UnsupportedFormat(format!(
        "cannot tell the format of {} (a raw binary file needs a JSON sidecar {})",
        path.display(),
        RawParams::sidecar_path(path).display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::{MemoryOrder, SampleFormat};

    #[test]
    fn test_open_detects_formats() {
        let dir = std::env::temp_dir().join(format!("dsp_io_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = SyntheticRecording::new(SyntheticParams { channels: 4, duration_sec: 0.1, ..Default::default() }).unwrap();

        let bin = dir.join("rec.bin");
        write_raw(&src, &bin, SampleFormat::I16, MemoryOrder::TimeMajor, 0.25, 1000, |_, _| {}).unwrap();
        let rec = open(&bin).unwrap();
        assert_eq!(rec.info().format, SampleFormat::I16);
        assert_eq!(rec.info().samples, src.info().samples);

        #[cfg(feature = "zarr")]
        {
            let z = dir.join("rec.zarr");
            write_zarr(&src, &z, 1000, |_, _| {}).unwrap();
            assert_eq!(open(&z).unwrap().info().channel_count(), 4);
        }

        std::fs::write(dir.join("other.dat"), [0u8; 8]).unwrap();
        assert!(matches!(open(&dir.join("other.dat")), Err(DspError::UnsupportedFormat(_))));
        assert!(open(&dir.join("missing.bin")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
