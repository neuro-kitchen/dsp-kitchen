//! NumPy `.npz` reader: a zip archive of `.npy` arrays (`numpy.savez` stores them, `savez_compressed`
//! deflates them). Entries are found through the zip central directory; stored (method 0) and
//! deflate (method 8) entries are read. Zip64 archives are not supported.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use dsp_core::{DspError, DspResult};

use super::{read_npy_bytes, NpyArray, NpyElement};

/// Signatures of the zip records used here.
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
/// Fixed sizes of the end-of-central-directory, central and local records.
const EOCD_LEN: usize = 22;
const CENTRAL_LEN: usize = 46;
const LOCAL_LEN: usize = 30;
/// Longest zip comment (the EOCD record is searched within it).
const MAX_COMMENT_LEN: usize = u16::MAX as usize;
/// Compression methods.
const METHOD_STORED: u16 = 0;
const METHOD_DEFLATE: u16 = 8;

fn malformed(path: &Path, what: impl std::fmt::Display) -> DspError {
    DspError::UnsupportedFormat(format!("{}: {what}", path.display()))
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The raw `.npy` bytes of every entry of the archive at `path`, by entry name without `.npy`.
pub fn read_npz_entries(path: &Path) -> DspResult<BTreeMap<String, Vec<u8>>> {
    let bytes = std::fs::read(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
    if bytes.len() < EOCD_LEN {
        return Err(malformed(path, "too short for a zip archive"));
    }
    // End of central directory: the last EOCD signature within the comment range
    let search_from = bytes.len().saturating_sub(EOCD_LEN + MAX_COMMENT_LEN);
    let eocd = (search_from..=bytes.len() - EOCD_LEN)
        .rev()
        .find(|&i| u32_at(&bytes, i) == EOCD_SIGNATURE)
        .ok_or_else(|| malformed(path, "no zip end-of-central-directory record"))?;
    let entries = u16_at(&bytes, eocd + 10) as usize;
    let cd_offset = u32_at(&bytes, eocd + 16) as usize;
    if cd_offset == u32::MAX as usize {
        return Err(malformed(path, "zip64 archives are not supported"));
    }

    let mut out = BTreeMap::new();
    let mut at = cd_offset;
    for _ in 0..entries {
        if at + CENTRAL_LEN > bytes.len() || u32_at(&bytes, at) != CENTRAL_SIGNATURE {
            return Err(malformed(path, "corrupt zip central directory"));
        }
        let method = u16_at(&bytes, at + 10);
        let compressed = u32_at(&bytes, at + 20) as usize;
        let uncompressed = u32_at(&bytes, at + 24) as usize;
        let (name_len, extra_len, comment_len) =
            (u16_at(&bytes, at + 28) as usize, u16_at(&bytes, at + 30) as usize, u16_at(&bytes, at + 32) as usize);
        let local = u32_at(&bytes, at + 42) as usize;
        let name_end = at + CENTRAL_LEN + name_len;
        if name_end > bytes.len() {
            return Err(malformed(path, "corrupt zip central directory"));
        }
        let name = String::from_utf8_lossy(&bytes[at + CENTRAL_LEN..name_end]).into_owned();
        at = name_end + extra_len + comment_len;

        if local + LOCAL_LEN > bytes.len() || u32_at(&bytes, local) != LOCAL_SIGNATURE {
            return Err(malformed(path, format!("corrupt local header of '{name}'")));
        }
        let data_start = local + LOCAL_LEN + u16_at(&bytes, local + 26) as usize + u16_at(&bytes, local + 28) as usize;
        let data = bytes
            .get(data_start..data_start + compressed)
            .ok_or_else(|| malformed(path, format!("entry '{name}' runs past the end of the archive")))?;
        let raw = match method {
            METHOD_STORED => data.to_vec(),
            METHOD_DEFLATE => {
                let mut v = Vec::with_capacity(uncompressed);
                flate2::read::DeflateDecoder::new(data)
                    .read_to_end(&mut v)
                    .map_err(|e| malformed(path, format!("entry '{name}': {e}")))?;
                v
            }
            other => return Err(malformed(path, format!("entry '{name}': unsupported zip method {other}"))),
        };
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key, raw);
    }
    Ok(out)
}

/// Every array of the `.npz` archive at `path`, as `T`, by name (`numpy.savez` keyword).
pub fn read_npz<T: NpyElement>(path: &Path) -> DspResult<BTreeMap<String, NpyArray<T>>> {
    read_npz_entries(path)?
        .into_iter()
        .map(|(name, raw)| {
            let label = path.join(&name);
            Ok((name, read_npy_bytes::<T>(&raw, &label)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored (uncompressed) zip with one `.npy` entry, built by hand.
    fn stored_zip(name: &str, payload: &[u8]) -> Vec<u8> {
        let mut z = Vec::new();
        let local_at = 0u32;
        z.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version, flags, method 0, time, date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc (not checked)
        z.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        z.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(payload);
        let cd_at = z.len() as u32;
        z.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // made by, needed, flags, method, time, date
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        z.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // extra, comment, disk, int attr, ext attr
        z.extend_from_slice(&local_at.to_le_bytes());
        z.extend_from_slice(name.as_bytes());
        let cd_len = z.len() as u32 - cd_at;
        z.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend_from_slice(&cd_len.to_le_bytes());
        z.extend_from_slice(&cd_at.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z
    }

    #[test]
    fn reads_a_stored_npz() {
        let dir = std::env::temp_dir().join(format!("dsp_npz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let npy = dir.join("a.npy");
        super::super::write_npy(&npy, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let zip = dir.join("arrays.npz");
        std::fs::write(&zip, stored_zip("wTEMP.npy", &std::fs::read(&npy).unwrap())).unwrap();
        let arrays = read_npz::<f32>(&zip).unwrap();
        let a = &arrays["wTEMP"];
        assert_eq!((a.shape.clone(), a.data.clone()), (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
