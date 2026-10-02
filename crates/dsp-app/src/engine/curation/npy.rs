//! Flexible `.npy` v1.0 / v2.0 reader and writer supporting all numeric dtypes emitted by
//! Kilosort 1–4, Phy, and `dsp-synapse` (`<u8`, `<i8`, `<u4`, `<i4`, `<i2`, `<u2`, `<f8`, `<f4`).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

#[derive(Debug, Clone)]
pub struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl NpyHeader {
    pub fn len(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }
}

pub fn read_header(r: &mut impl Read) -> Result<NpyHeader> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a .npy file");
    }
    let mut ver = [0u8; 2];
    r.read_exact(&mut ver)?;
    let h_len = match ver[0] {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            u32::from_le_bytes(b) as usize
        }
        v => bail!("unsupported .npy version {v}"),
    };
    let mut buf = vec![0u8; h_len];
    r.read_exact(&mut buf)?;
    let header = String::from_utf8_lossy(&buf);

    let descr = extract_quoted(&header, "descr").context("missing 'descr' in .npy header")?;
    let fortran_order = header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True");
    let shape_inner = header
        .split("shape")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .context("missing 'shape' in .npy header")?;
    let mut shape = Vec::new();
    for tok in shape_inner.split(',') {
        let t = tok.trim();
        if !t.is_empty() {
            shape.push(t.parse::<usize>().with_context(|| format!("invalid dimension '{t}'"))?);
        }
    }
    if shape.is_empty() {
        shape.push(1);
    }
    Ok(NpyHeader { descr, fortran_order, shape })
}

fn extract_quoted(s: &str, key: &str) -> Option<String> {
    let after = s.split(key).nth(1)?;
    let colon = after.split(':').nth(1)?.trim();
    let quote = colon.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    Some(colon[1..].split(quote).next()?.to_string())
}

/// Reads any numeric `.npy` array as `f64` values plus its shape.
pub fn read_f64_nd(path: &Path) -> Result<(Vec<f64>, Vec<usize>)> {
    let f = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut r = BufReader::new(f);
    let h = read_header(&mut r).with_context(|| format!("in {}", path.display()))?;
    let n = h.len();
    let mut out = Vec::with_capacity(n);
    let dtype = h.descr.trim_start_matches(['<', '|', '=']);
    match dtype {
        "f8" => {
            let mut b = [0u8; 8];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(f64::from_le_bytes(b));
            }
        }
        "f4" => {
            let mut b = [0u8; 4];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(f32::from_le_bytes(b) as f64);
            }
        }
        "i8" => {
            let mut b = [0u8; 8];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(i64::from_le_bytes(b) as f64);
            }
        }
        "u8" => {
            let mut b = [0u8; 8];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(u64::from_le_bytes(b) as f64);
            }
        }
        "i4" => {
            let mut b = [0u8; 4];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(i32::from_le_bytes(b) as f64);
            }
        }
        "u4" => {
            let mut b = [0u8; 4];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(u32::from_le_bytes(b) as f64);
            }
        }
        "i2" => {
            let mut b = [0u8; 2];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(i16::from_le_bytes(b) as f64);
            }
        }
        "u2" => {
            let mut b = [0u8; 2];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(u16::from_le_bytes(b) as f64);
            }
        }
        "i1" => {
            let mut b = [0u8; 1];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push((b[0] as i8) as f64);
            }
        }
        "u1" | "b1" => {
            let mut b = [0u8; 1];
            for _ in 0..n {
                r.read_exact(&mut b)?;
                out.push(b[0] as f64);
            }
        }
        other => bail!("unsupported .npy dtype '{}' in {}", other, path.display()),
    }
    if h.fortran_order && h.shape.len() == 2 {
        let (rows, cols) = (h.shape[0], h.shape[1]);
        let mut c_order = vec![0.0; n];
        for row in 0..rows {
            for col in 0..cols {
                c_order[row * cols + col] = out[col * rows + row];
            }
        }
        out = c_order;
    }
    Ok((out, h.shape))
}

pub fn read_u64_vec(path: &Path) -> Result<Vec<u64>> {
    let (vals, _) = read_f64_nd(path)?;
    Ok(vals.into_iter().map(|v| v.round().max(0.0) as u64).collect())
}

pub fn read_i32_vec(path: &Path) -> Result<Vec<i32>> {
    let (vals, _) = read_f64_nd(path)?;
    Ok(vals.into_iter().map(|v| v.round() as i32).collect())
}

pub fn read_f32_vec(path: &Path) -> Result<Vec<f32>> {
    let (vals, _) = read_f64_nd(path)?;
    Ok(vals.into_iter().map(|v| v as f32).collect())
}

pub fn read_f32_nd(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let (vals, shape) = read_f64_nd(path)?;
    Ok((vals.into_iter().map(|v| v as f32).collect(), shape))
}

pub fn read_i32_nd(path: &Path) -> Result<(Vec<i32>, Vec<usize>)> {
    let (vals, shape) = read_f64_nd(path)?;
    Ok((vals.into_iter().map(|v| v.round() as i32).collect(), shape))
}

/// Writes a 1D `<i4` NumPy `.npy` file (`spike_clusters.npy`).
pub fn write_i32_1d(path: &Path, values: &[i32]) -> Result<()> {
    let f = File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    let mut dict = format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}", values.len());
    let min_len = 10 + dict.len() + 1;
    let total = min_len.div_ceil(64) * 64;
    let pad = total - (10 + dict.len() + 1);
    for _ in 0..pad {
        dict.push(' ');
    }
    dict.push('\n');
    let h_len = dict.len() as u16;
    w.write_all(MAGIC)?;
    w.write_all(&[1, 0])?;
    w.write_all(&h_len.to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    for &v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
pub fn write_u64_for_test(path: &Path, values: &[u64]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    let mut dict = format!("{{'descr': '<u8', 'fortran_order': False, 'shape': ({},), }}", values.len());
    let min_len = 10 + dict.len() + 1;
    let total = min_len.div_ceil(64) * 64;
    for _ in 0..(total - (10 + dict.len() + 1)) {
        dict.push(' ');
    }
    dict.push('\n');
    w.write_all(MAGIC).unwrap();
    w.write_all(&[1, 0]).unwrap();
    w.write_all(&(dict.len() as u16).to_le_bytes()).unwrap();
    w.write_all(dict.as_bytes()).unwrap();
    for &v in values {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}
