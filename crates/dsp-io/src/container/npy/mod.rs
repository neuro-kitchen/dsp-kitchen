//! NumPy `.npy` reader and writer (no dependencies).
//!
//! Reads what Kilosort 1–4, Phy, SpikeInterface and this crate write: format versions 1–3, any
//! numeric dtype (`b1`, `u1`/`i1`, `u2`/`i2`, `u4`/`i4`, `u8`/`i8`, `f4`/`f8`) in either byte
//! order, C or Fortran order (returned in C order), any number of dimensions. [`read_npy`]
//! converts to the element type asked for; integers convert exactly, never through `f64`.
//! Writes version 1.0, little-endian, C order.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use dsp_core::{DspError, DspResult};

mod npz;
pub use npz::{read_npz, read_npz_entries};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

fn io(e: std::io::Error) -> DspError {
    DspError::Io(e.to_string())
}

fn malformed(path: &Path, what: impl std::fmt::Display) -> DspError {
    DspError::UnsupportedFormat(format!("{}: {what}", path.display()))
}

/// The header of an `.npy` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    /// dtype string, e.g. `<f4`, `|u1`.
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl NpyHeader {
    /// Elements in the array (1 for a 0-d array).
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An array read from an `.npy` file, in C (row-major) order.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> NpyArray<T> {
    /// The shape as exactly `N` dimensions, else an error naming `path`.
    pub fn dims<const N: usize>(&self, path: &Path) -> DspResult<[usize; N]> {
        self.shape.as_slice().try_into().map_err(|_| malformed(path, format!("expected {N} dimensions, got shape {:?}", self.shape)))
    }
}

/// Element types an array can be read as, from any stored numeric dtype.
pub trait NpyElement: Copy + Default {
    fn from_u64(v: u64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_f64(v: f64) -> Self;
}

macro_rules! npy_element {
    ($($t:ty),*) => {$(
        impl NpyElement for $t {
            #[inline]
            fn from_u64(v: u64) -> Self { v as $t }
            #[inline]
            fn from_i64(v: i64) -> Self { v as $t }
            #[inline]
            fn from_f64(v: f64) -> Self { v as $t }
        }
    )*};
}
npy_element!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64);

/// Element types an array can be written as.
pub trait NpyWritable: Copy {
    const DESCR: &'static str;
    fn write_le(self, w: &mut impl Write) -> std::io::Result<()>;
}

macro_rules! npy_writable {
    ($($t:ty => $d:literal),*) => {$(
        impl NpyWritable for $t {
            const DESCR: &'static str = $d;
            #[inline]
            fn write_le(self, w: &mut impl Write) -> std::io::Result<()> { w.write_all(&self.to_le_bytes()) }
        }
    )*};
}
npy_writable!(u8 => "|u1", i32 => "<i4", u32 => "<u4", i64 => "<i8", u64 => "<u8", f32 => "<f4", f64 => "<f8");

/// Reads the header, leaving `r` at the first data byte.
pub fn read_header(r: &mut impl Read, path: &Path) -> DspResult<NpyHeader> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic).map_err(io)?;
    if &magic != MAGIC {
        return Err(malformed(path, "not a .npy file"));
    }
    let mut ver = [0u8; 2];
    r.read_exact(&mut ver).map_err(io)?;
    let h_len = match ver[0] {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).map_err(io)?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(io)?;
            u32::from_le_bytes(b) as usize
        }
        v => return Err(malformed(path, format!("unsupported .npy version {v}"))),
    };
    let mut buf = vec![0u8; h_len];
    r.read_exact(&mut buf).map_err(io)?;
    parse_header(&String::from_utf8_lossy(&buf)).ok_or_else(|| malformed(path, "malformed .npy header"))
}

/// The value of `key` in the header dictionary, as the text after its colon.
fn value_after<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let at = header.find(&format!("'{key}'")).or_else(|| header.find(&format!("\"{key}\"")))?;
    let rest = &header[at + key.len() + 2..];
    Some(rest[rest.find(':')? + 1..].trim_start())
}

fn parse_header(header: &str) -> Option<NpyHeader> {
    let d = value_after(header, "descr")?;
    let quote = d.chars().next().filter(|q| *q == '\'' || *q == '"')?;
    let descr = d[1..].split(quote).next()?.to_string();
    let fortran_order = value_after(header, "fortran_order")?.starts_with("True");
    let s = value_after(header, "shape")?;
    let inner = s.strip_prefix('(')?.split(')').next()?;
    let shape = inner.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse().ok()).collect::<Option<Vec<usize>>>()?;
    Some(NpyHeader { descr, fortran_order, shape })
}

/// Converts `bytes` of dtype `descr` into `T`s.
fn convert<T: NpyElement>(descr: &str, bytes: &[u8], path: &Path) -> DspResult<Vec<T>> {
    let big = descr.starts_with('>');
    let code = descr.trim_start_matches(['<', '>', '|', '=']);
    macro_rules! each {
        ($n:literal, $t:ty, $to:ident, $as:ty) => {
            bytes.chunks_exact($n).map(|c| {
                let a: [u8; $n] = c.try_into().unwrap();
                T::$to((if big { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }) as $as)
            }).collect()
        };
    }
    Ok(match code {
        "b1" | "u1" => bytes.iter().map(|&b| T::from_u64(b as u64)).collect(),
        "i1" => bytes.iter().map(|&b| T::from_i64(b as i8 as i64)).collect(),
        "u2" => each!(2, u16, from_u64, u64),
        "i2" => each!(2, i16, from_i64, i64),
        "u4" => each!(4, u32, from_u64, u64),
        "i4" => each!(4, i32, from_i64, i64),
        "u8" => each!(8, u64, from_u64, u64),
        "i8" => each!(8, i64, from_i64, i64),
        "f4" => each!(4, f32, from_f64, f64),
        "f8" => each!(8, f64, from_f64, f64),
        other => return Err(malformed(path, format!("unsupported dtype '{other}'"))),
    })
}

fn dtype_size(descr: &str) -> Option<usize> {
    descr.trim_start_matches(['<', '>', '|', '=']).get(1..)?.parse().ok()
}

/// Fortran (column-major) order to C (row-major) order, any number of dimensions.
fn to_c_order<T: Copy + Default>(data: Vec<T>, shape: &[usize]) -> Vec<T> {
    if shape.len() < 2 {
        return data;
    }
    let mut out = vec![T::default(); data.len()];
    let mut index = vec![0usize; shape.len()];
    for v in data {
        // `index` walks the Fortran order (first axis fastest); place it at its C offset
        let c = index.iter().zip(shape).fold(0usize, |acc, (&i, &n)| acc * n + i);
        out[c] = v;
        for (i, &n) in index.iter_mut().zip(shape) {
            *i += 1;
            if *i < n {
                break;
            }
            *i = 0;
        }
    }
    out
}

/// Reads any numeric `.npy` array as `T`, in C order.
pub fn read_npy<T: NpyElement>(path: &Path) -> DspResult<NpyArray<T>> {
    let file = File::open(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
    read_npy_from(&mut BufReader::new(file), path)
}

/// [`read_npy`] of an in-memory `.npy` (e.g. an `.npz` entry); `label` names it in errors.
pub fn read_npy_bytes<T: NpyElement>(bytes: &[u8], label: &Path) -> DspResult<NpyArray<T>> {
    read_npy_from(&mut std::io::Cursor::new(bytes), label)
}

fn read_npy_from<T: NpyElement>(r: &mut impl Read, path: &Path) -> DspResult<NpyArray<T>> {
    let h = read_header(r, path)?;
    let size = dtype_size(&h.descr).ok_or_else(|| malformed(path, format!("unsupported dtype '{}'", h.descr)))?;
    let mut bytes = vec![0u8; h.len() * size];
    r.read_exact(&mut bytes).map_err(|e| malformed(path, format!("truncated data: {e}")))?;
    let data = convert::<T>(&h.descr, &bytes, path)?;
    let data = if h.fortran_order { to_c_order(data, &h.shape) } else { data };
    Ok(NpyArray { data, shape: h.shape })
}

/// Writes `data` (C order) with `shape` as a version 1.0 `.npy` file.
pub fn write_npy<T: NpyWritable>(path: &Path, data: &[T], shape: &[usize]) -> DspResult<()> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(DspError::InvalidConfig(format!("{}: {} values do not fill shape {shape:?}", path.display(), data.len())));
    }
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {dims}, }}", T::DESCR);
    // Magic (6) + version (2) + length (2) + dictionary + newline: a multiple of 64 bytes
    let unpadded = 10 + dict.len() + 1;
    dict.push_str(&" ".repeat(unpadded.div_ceil(64) * 64 - unpadded));
    dict.push('\n');
    let mut w = BufWriter::new(File::create(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?);
    w.write_all(MAGIC).map_err(io)?;
    w.write_all(&[1, 0]).map_err(io)?;
    w.write_all(&(dict.len() as u16).to_le_bytes()).map_err(io)?;
    w.write_all(dict.as_bytes()).map_err(io)?;
    for &v in data {
        v.write_le(&mut w).map_err(io)?;
    }
    w.flush().map_err(io)
}

// ---------------------------------------------------------------------------
// Typed shorthands (any stored dtype converts)
// ---------------------------------------------------------------------------

pub fn read_npy_u64_1d(path: &Path) -> DspResult<Vec<u64>> {
    Ok(read_npy::<u64>(path)?.data)
}

pub fn read_npy_i32_1d(path: &Path) -> DspResult<Vec<i32>> {
    Ok(read_npy::<i32>(path)?.data)
}

pub fn read_npy_f32_1d(path: &Path) -> DspResult<Vec<f32>> {
    Ok(read_npy::<f32>(path)?.data)
}

pub fn read_npy_f32_2d(path: &Path) -> DspResult<(Vec<f32>, [usize; 2])> {
    let a = read_npy::<f32>(path)?;
    let dims = a.dims::<2>(path)?;
    Ok((a.data, dims))
}

pub fn read_npy_f32_3d(path: &Path) -> DspResult<(Vec<f32>, [usize; 3])> {
    let a = read_npy::<f32>(path)?;
    let dims = a.dims::<3>(path)?;
    Ok((a.data, dims))
}

pub fn write_npy_u64_1d(path: &Path, slice: &[u64]) -> DspResult<()> {
    write_npy(path, slice, &[slice.len()])
}

pub fn write_npy_i32_1d(path: &Path, slice: &[i32]) -> DspResult<()> {
    write_npy(path, slice, &[slice.len()])
}

pub fn write_npy_f32_1d(path: &Path, slice: &[f32]) -> DspResult<()> {
    write_npy(path, slice, &[slice.len()])
}

pub fn write_npy_f32_2d(path: &Path, slice: &[f32], shape: [usize; 2]) -> DspResult<()> {
    write_npy(path, slice, &shape)
}

pub fn write_npy_f32_3d(path: &Path, slice: &[f32], shape: [usize; 3]) -> DspResult<()> {
    write_npy(path, slice, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("dsp_npy_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An `.npy` file with a hand-written header (dtypes and orders this crate does not write).
    fn raw(path: &Path, descr: &str, fortran: bool, shape: &str, data: &[u8], version: u8) {
        let mut dict = format!("{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}", if fortran { "True" } else { "False" });
        let prefix = if version == 1 { 10 } else { 12 };
        let unpadded = prefix + dict.len() + 1;
        dict.push_str(&" ".repeat(unpadded.div_ceil(64) * 64 - unpadded));
        dict.push('\n');
        let mut f = File::create(path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&[version, 0]).unwrap();
        if version == 1 {
            f.write_all(&(dict.len() as u16).to_le_bytes()).unwrap();
        } else {
            f.write_all(&(dict.len() as u32).to_le_bytes()).unwrap();
        }
        f.write_all(dict.as_bytes()).unwrap();
        f.write_all(data).unwrap();
    }

    #[test]
    fn test_npy_roundtrip_1d_and_3d() {
        let dir = scratch("roundtrip");
        let u64_path = dir.join("times.npy");
        let data_u64 = vec![100u64, 250, 4000, 99999];
        write_npy_u64_1d(&u64_path, &data_u64).unwrap();
        assert_eq!(read_npy_u64_1d(&u64_path).unwrap(), data_u64);

        let f32_path = dir.join("templates.npy");
        let data_3d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        write_npy_f32_3d(&f32_path, &data_3d, [2, 2, 2]).unwrap();
        assert_eq!(read_npy_f32_3d(&f32_path).unwrap(), (data_3d, [2, 2, 2]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_reads_other_dtypes_exactly() {
        let dir = scratch("dtypes");
        // int64 spike times beyond f32 precision, read as u64 and as f64
        let p = dir.join("i8.npy");
        let v: Vec<i64> = vec![16_777_217, 3, 9_007_199_254_740_993];
        write_npy(&p, &v, &[3]).unwrap();
        assert_eq!(read_npy::<u64>(&p).unwrap().data, vec![16_777_217, 3, 9_007_199_254_740_993]);
        // float64 positions read as f32
        let p = dir.join("f8.npy");
        write_npy(&p, &[1.5f64, -2.25, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(read_npy_f32_2d(&p).unwrap(), (vec![1.5, -2.25, 3.0, 4.0], [2, 2]));
        // big-endian int16
        let p = dir.join("be.npy");
        raw(&p, ">i2", false, "(2,)", &[0xff, 0xfe, 0x00, 0x05], 1);
        assert_eq!(read_npy::<i32>(&p).unwrap().data, vec![-2, 5]);
        // uint8 / bool
        let p = dir.join("u1.npy");
        raw(&p, "|b1", false, "(3,)", &[1, 0, 1], 1);
        assert_eq!(read_npy::<u8>(&p).unwrap().data, vec![1, 0, 1]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_fortran_order_and_version_2() {
        let dir = scratch("fortran");
        // A 2×3 array [[0, 1, 2], [3, 4, 5]] stored column by column
        let p = dir.join("f.npy");
        let cols: Vec<u8> = [0i32, 3, 1, 4, 2, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        raw(&p, "<i4", true, "(2, 3)", &cols, 2);
        let a = read_npy::<i32>(&p).unwrap();
        assert_eq!((a.data, a.shape), (vec![0, 1, 2, 3, 4, 5], vec![2, 3]));
        // 3-d Fortran: element (i, j, k) of shape (2, 2, 2) has value 100i + 10j + k
        let mut f = Vec::new();
        for k in 0..2i32 {
            for j in 0..2i32 {
                for i in 0..2i32 {
                    f.extend((100 * i + 10 * j + k).to_le_bytes());
                }
            }
        }
        raw(&p, "<i4", true, "(2, 2, 2)", &f, 1);
        assert_eq!(read_npy::<i32>(&p).unwrap().data, vec![0, 1, 10, 11, 100, 101, 110, 111]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_errors_name_the_problem() {
        let dir = scratch("errors");
        let p = dir.join("bad.npy");
        std::fs::write(&p, b"not numpy").unwrap();
        assert!(read_npy::<f32>(&p).unwrap_err().to_string().contains("not a .npy"));
        write_npy(&p, &[1.0f32, 2.0], &[2]).unwrap();
        assert!(read_npy_f32_2d(&p).unwrap_err().to_string().contains("expected 2 dimensions"));
        assert!(write_npy(&p, &[1.0f32], &[2]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
