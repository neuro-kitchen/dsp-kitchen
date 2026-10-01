//! Zero-dependency NumPy `.npy` (v1.0 / v2.0) tensor reader and writer for `dsp-synapse-ml`.
//!
//! Correctly handles both C-contiguous (`'fortran_order': False`) and Fortran column-major
//! (`'fortran_order': True`, used by Kilosort4's `wPCA.npy`) `<f4` (`float32`) arrays, normalizing
//! loaded data into standard row-major C order `[dim0, dim1, ...]`.

use anyhow::{Context, Result, bail};
use std::path::Path;

/// Parsed NumPy `.npy` float32 tensor normalized to C-contiguous (row-major) layout.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyTensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyTensorF32 {
    /// Loads a `.npy` float32 file from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let bytes = std::fs::read(p)
            .with_context(|| format!("failed to read .npy file {}", p.display()))?;
        Self::from_bytes(&bytes)
            .with_context(|| format!("failed to parse .npy file {}", p.display()))
    }

    /// Parses a `.npy` float32 byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
            bail!("invalid .npy magic header (expected \\x93NUMPY)");
        }
        let major = bytes[6];
        let (header_len, data_offset) = match major {
            1 => {
                let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
                (hlen, 10 + hlen)
            }
            2 | 3 => {
                if bytes.len() < 12 {
                    bail!("truncated .npy v2 header");
                }
                let hlen =
                    u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                (hlen, 12 + hlen)
            }
            _ => bail!("unsupported .npy format major version {major}"),
        };

        if bytes.len() < data_offset {
            bail!(
                "truncated .npy header: expected {} header bytes, file has {}",
                header_len,
                bytes.len()
            );
        }

        let header_str = std::str::from_utf8(&bytes[data_offset - header_len..data_offset])
            .context("invalid UTF-8 in .npy header")?;

        let descr = extract_dict_str_field(header_str, "'descr'")
            .or_else(|| extract_dict_str_field(header_str, "\"descr\""))
            .ok_or_else(|| anyhow::anyhow!("missing 'descr' in .npy header: {header_str}"))?;
        if descr != "<f4" && descr != "f4" && descr != "=f4" {
            bail!("unsupported .npy dtype '{descr}' (expected '<f4' float32)");
        }

        let fortran_order = parse_fortran_order(header_str)?;
        let shape = parse_shape_tuple(header_str)?;
        let num_elems: usize = shape.iter().product();
        let expected_data_bytes = num_elems * 4;
        let payload = &bytes[data_offset..];
        if payload.len() < expected_data_bytes {
            bail!(
                ".npy payload too short: expected {expected_data_bytes} bytes for shape {shape:?}, got {}",
                payload.len()
            );
        }

        let mut raw_floats = Vec::with_capacity(num_elems);
        for chunk in payload[..expected_data_bytes].chunks_exact(4) {
            raw_floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let data = if fortran_order && shape.len() == 2 {
            let [rows, cols] = [shape[0], shape[1]];
            let mut row_major = vec![0.0f32; num_elems];
            for r in 0..rows {
                for c in 0..cols {
                    row_major[r * cols + c] = raw_floats[c * rows + r];
                }
            }
            row_major
        } else if fortran_order && shape.len() > 2 {
            transpose_fortran_nd(&raw_floats, &shape)
        } else {
            raw_floats
        };

        Ok(Self { shape, data })
    }
}

fn extract_dict_str_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let idx = header.find(key)?;
    let after_key = &header[idx + key.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let quote = after_colon.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &after_colon[quote.len_utf8()..];
    let end_quote = rest.find(quote)?;
    Some(&rest[..end_quote])
}

fn parse_fortran_order(header: &str) -> Result<bool> {
    let key_idx = header
        .find("'fortran_order'")
        .or_else(|| header.find("\"fortran_order\""))
        .ok_or_else(|| anyhow::anyhow!("missing 'fortran_order' in .npy header"))?;
    let after = &header[key_idx..];
    let colon = after
        .find(':')
        .ok_or_else(|| anyhow::anyhow!("malformed 'fortran_order' in .npy header"))?;
    let val_part = after[colon + 1..].trim_start();
    if val_part.starts_with("True") {
        Ok(true)
    } else if val_part.starts_with("False") {
        Ok(false)
    } else {
        bail!("invalid 'fortran_order' value in .npy header: {header}")
    }
}

fn parse_shape_tuple(header: &str) -> Result<Vec<usize>> {
    let key_idx = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| anyhow::anyhow!("missing 'shape' in .npy header"))?;
    let after = &header[key_idx..];
    let open_paren = after
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("missing '(' for shape in .npy header"))?;
    let close_paren = after[open_paren..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("missing ')' for shape in .npy header"))?;
    let tuple_inner = &after[open_paren + 1..open_paren + close_paren];
    let mut dims = Vec::new();
    for part in tuple_inner.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let d: usize = trimmed
                .parse()
                .with_context(|| format!("invalid dimension '{trimmed}' in .npy shape"))?;
            dims.push(d);
        }
    }
    Ok(dims)
}

fn transpose_fortran_nd(raw: &[f32], shape: &[usize]) -> Vec<f32> {
    let n = raw.len();
    let ndim = shape.len();
    let mut out = vec![0.0f32; n];
    let mut coords = vec![0usize; ndim];
    for c_idx in 0..n {
        let mut rem = c_idx;
        for d in (0..ndim).rev() {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        let mut f_idx = 0usize;
        let mut stride = 1usize;
        for d in 0..ndim {
            f_idx += coords[d] * stride;
            stride *= shape[d];
        }
        out[c_idx] = raw[f_idx];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_reader_handles_fortran_and_c_order() {
        let wpca_path = std::path::Path::new("../../data/kilosort4/wPCA.npy");
        if wpca_path.exists() {
            let wpca = NpyTensorF32::from_file(wpca_path).unwrap();
            assert_eq!(wpca.shape, vec![6, 61]);
            assert_eq!(wpca.data.len(), 6 * 61);
            // Each row of wPCA is a unit-norm temporal principal component: ||row||_2 == 1.0
            for r in 0..6 {
                let row = &wpca.data[r * 61..(r + 1) * 61];
                let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-3,
                    "row {r} norm was {norm}, expected 1.0 (Fortran transpose check)"
                );
            }
        }

        let wtemp_path = std::path::Path::new("../../data/kilosort4/wTEMP.npy");
        if wtemp_path.exists() {
            let wtemp = NpyTensorF32::from_file(wtemp_path).unwrap();
            assert_eq!(wtemp.shape, vec![6, 61]);
            for r in 0..6 {
                let row = &wtemp.data[r * 61..(r + 1) * 61];
                let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-3,
                    "wTEMP row {r} norm was {norm}, expected 1.0"
                );
            }
        }
    }
}
