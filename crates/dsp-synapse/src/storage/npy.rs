//! Standalone NumPy `.npy` v1.0 File Reader & Writer (`npy.rs`).
//!
//! Provides zero-dependency binary parsing and serialization of C-contiguous
//! little-endian 1D, 2D, and 3D numeric arrays according to the NumPy `.npy` format
//! specification:
//! - Magic string: `\x93NUMPY`
//! - Version: `\x01\x00` (v1.0)
//! - Header length: 16-bit little-endian integer ($H$)
//! - Python dictionary string padded with spaces and ending with `\n` such that
//!   `6 + 2 + 2 + H` is a multiple of 64 bytes.
//! - Raw binary payload.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use dsp_core::{DspError, DspResult};

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

fn write_header(
    writer: &mut impl Write,
    descr: &str,
    shape: &[usize],
) -> DspResult<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let items: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({},)", items.join(", "))
    };

    let mut dict_str = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}}}",
        descr, shape_str
    );

    // Header prefix is 6 (magic) + 2 (version) + 2 (header_len) = 10 bytes.
    // Total header length (10 + dict_len) must be a multiple of 64.
    let min_len = 10 + dict_str.len() + 1; // +1 for '\n'
    let target_total = ((min_len + 63) / 64) * 64;
    let pad_len = target_total - (10 + dict_str.len() + 1);
    for _ in 0..pad_len {
        dict_str.push(' ');
    }
    dict_str.push('\n');

    let h_len = dict_str.len() as u16;
    writer.write_all(NPY_MAGIC).map_err(|e| DspError::Io(e.to_string()))?;
    writer.write_all(&[1, 0]).map_err(|e| DspError::Io(e.to_string()))?;
    writer.write_all(&h_len.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    writer.write_all(dict_str.as_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

fn parse_header(reader: &mut impl Read) -> DspResult<(String, Vec<usize>, usize)> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).map_err(|e| DspError::Io(e.to_string()))?;
    if &magic != NPY_MAGIC {
        return Err(DspError::UnsupportedFormat("Not a valid .npy file".into()));
    }
    let mut ver = [0u8; 2];
    reader.read_exact(&mut ver).map_err(|e| DspError::Io(e.to_string()))?;
    let mut h_len_bytes = [0u8; 2];
    reader.read_exact(&mut h_len_bytes).map_err(|e| DspError::Io(e.to_string()))?;
    let h_len = u16::from_le_bytes(h_len_bytes) as usize;

    let mut header_bytes = vec![0u8; h_len];
    reader.read_exact(&mut header_bytes).map_err(|e| DspError::Io(e.to_string()))?;
    let header_str = String::from_utf8_lossy(&header_bytes);

    // Extract 'descr': '<f4' or '<u8' or '<i4'
    let descr = header_str
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .or_else(|| header_str.split("\"descr\":").nth(1).and_then(|s| s.split('"').nth(1)))
        .ok_or_else(|| DspError::UnsupportedFormat("Malformed .npy descr".into()))?
        .to_string();

    // Extract 'shape': (100, 20)
    let shape_str = header_str
        .split("'shape':")
        .nth(1)
        .or_else(|| header_str.split("\"shape\":").nth(1))
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| DspError::UnsupportedFormat("Malformed .npy shape".into()))?;

    let mut shape = Vec::new();
    for token in shape_str.split(',') {
        let t = token.trim();
        if !t.is_empty() {
            if let Ok(val) = t.parse::<usize>() {
                shape.push(val);
            }
        }
    }

    let total_elements = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    Ok((descr, shape, total_elements))
}

// ---------------------------------------------------------------------------
// Typed Writers
// ---------------------------------------------------------------------------

pub fn write_npy_u64_1d(path: &Path, slice: &[u64]) -> DspResult<()> {
    let f = File::create(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut w = BufWriter::new(f);
    write_header(&mut w, "<u8", &[slice.len()])?;
    for &val in slice {
        w.write_all(&val.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    }
    w.flush().map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

pub fn write_npy_i32_1d(path: &Path, slice: &[i32]) -> DspResult<()> {
    let f = File::create(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut w = BufWriter::new(f);
    write_header(&mut w, "<i4", &[slice.len()])?;
    for &val in slice {
        w.write_all(&val.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    }
    w.flush().map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

pub fn write_npy_f32_1d(path: &Path, slice: &[f32]) -> DspResult<()> {
    let f = File::create(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut w = BufWriter::new(f);
    write_header(&mut w, "<f4", &[slice.len()])?;
    for &val in slice {
        w.write_all(&val.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    }
    w.flush().map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

pub fn write_npy_f32_2d(path: &Path, slice: &[f32], shape: [usize; 2]) -> DspResult<()> {
    assert_eq!(slice.len(), shape[0] * shape[1], "2D slice length must match shape");
    let f = File::create(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut w = BufWriter::new(f);
    write_header(&mut w, "<f4", &shape)?;
    for &val in slice {
        w.write_all(&val.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    }
    w.flush().map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

pub fn write_npy_f32_3d(path: &Path, slice: &[f32], shape: [usize; 3]) -> DspResult<()> {
    assert_eq!(slice.len(), shape[0] * shape[1] * shape[2], "3D slice length must match shape");
    let f = File::create(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut w = BufWriter::new(f);
    write_header(&mut w, "<f4", &shape)?;
    for &val in slice {
        w.write_all(&val.to_le_bytes()).map_err(|e| DspError::Io(e.to_string()))?;
    }
    w.flush().map_err(|e| DspError::Io(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Typed Readers
// ---------------------------------------------------------------------------

pub fn read_npy_u64_1d(path: &Path) -> DspResult<Vec<u64>> {
    let f = File::open(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut r = BufReader::new(f);
    let (_descr, _shape, n) = parse_header(&mut r)?;
    let mut out = vec![0u64; n];
    let mut b = [0u8; 8];
    for val in &mut out {
        r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
        *val = u64::from_le_bytes(b);
    }
    Ok(out)
}

pub fn read_npy_i32_1d(path: &Path) -> DspResult<Vec<i32>> {
    let f = File::open(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut r = BufReader::new(f);
    let (descr, _shape, n) = parse_header(&mut r)?;
    let mut out = vec![0i32; n];
    let is_i8 = descr.contains("i8") || descr.contains("u8");
    if is_i8 {
        let mut b = [0u8; 8];
        for val in &mut out {
            r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
            *val = i64::from_le_bytes(b) as i32;
        }
    } else {
        let mut b = [0u8; 4];
        for val in &mut out {
            r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
            *val = i32::from_le_bytes(b);
        }
    }
    Ok(out)
}

pub fn read_npy_f32_1d(path: &Path) -> DspResult<Vec<f32>> {
    let f = File::open(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut r = BufReader::new(f);
    let (_descr, _shape, n) = parse_header(&mut r)?;
    let mut out = vec![0.0f32; n];
    let mut b = [0u8; 4];
    for val in &mut out {
        r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
        *val = f32::from_le_bytes(b);
    }
    Ok(out)
}

pub fn read_npy_f32_2d(path: &Path) -> DspResult<(Vec<f32>, [usize; 2])> {
    let f = File::open(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut r = BufReader::new(f);
    let (_descr, shape, n) = parse_header(&mut r)?;
    if shape.len() != 2 {
        return Err(DspError::UnsupportedFormat(format!(
            "Expected 2D shape, got {:?}",
            shape
        )));
    }
    let mut out = vec![0.0f32; n];
    let mut b = [0u8; 4];
    for val in &mut out {
        r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
        *val = f32::from_le_bytes(b);
    }
    Ok((out, [shape[0], shape[1]]))
}

pub fn read_npy_f32_3d(path: &Path) -> DspResult<(Vec<f32>, [usize; 3])> {
    let f = File::open(path).map_err(|e| DspError::Io(e.to_string()))?;
    let mut r = BufReader::new(f);
    let (_descr, shape, n) = parse_header(&mut r)?;
    if shape.len() != 3 {
        return Err(DspError::UnsupportedFormat(format!(
            "Expected 3D shape, got {:?}",
            shape
        )));
    }
    let mut out = vec![0.0f32; n];
    let mut b = [0u8; 4];
    for val in &mut out {
        r.read_exact(&mut b).map_err(|e| DspError::Io(e.to_string()))?;
        *val = f32::from_le_bytes(b);
    }
    Ok((out, [shape[0], shape[1], shape[2]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_roundtrip_1d_and_3d() {
        let dir = std::env::temp_dir().join(format!("dsp_npy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let u64_path = dir.join("times.npy");
        let data_u64 = vec![100u64, 250, 4000, 99999];
        write_npy_u64_1d(&u64_path, &data_u64).unwrap();
        let loaded_u64 = read_npy_u64_1d(&u64_path).unwrap();
        assert_eq!(data_u64, loaded_u64);

        let f32_path = dir.join("templates.npy");
        let data_3d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [2, 2, 2];
        write_npy_f32_3d(&f32_path, &data_3d, shape).unwrap();
        let (loaded_3d, loaded_shape) = read_npy_f32_3d(&f32_path).unwrap();
        assert_eq!(data_3d, loaded_3d);
        assert_eq!(shape, loaded_shape);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
