//! Reshape, transpose and concatenation.

use anyhow::{Context, Result, bail};

use super::elementwise::compute_strides;
use super::super::tensor::DynTensor;
use crate::backend::SynapseMlDevice;

pub(crate) fn resolve_reshape_dims(input_shape: &[usize], target: &[i64]) -> Result<Vec<usize>> {
    let total: usize = input_shape.iter().product();
    let mut out = vec![0usize; target.len()];
    let mut infer_idx: Option<usize> = None;
    let mut known_prod = 1usize;

    for (i, &dim) in target.iter().enumerate() {
        if dim == -1 {
            if infer_idx.is_some() {
                bail!("Reshape target {:?} has multiple -1 dimensions", target);
            }
            infer_idx = Some(i);
        } else if dim == 0 {
            let copied = *input_shape
                .get(i)
                .with_context(|| format!("Reshape 0-dim at {} out of bounds for {:?}", i, input_shape))?;
            out[i] = copied;
            known_prod *= copied;
        } else if dim > 0 {
            out[i] = dim as usize;
            known_prod *= dim as usize;
        } else {
            bail!("Invalid Reshape dimension {}", dim);
        }
    }

    if let Some(idx) = infer_idx {
        if known_prod == 0 || total % known_prod != 0 {
            bail!(
                "Cannot infer -1 dimension for total {} with known product {}",
                total,
                known_prod
            );
        }
        out[idx] = total / known_prod;
    }

    Ok(out)
}

pub(crate) fn exec_transpose(x: &DynTensor, perm: &[usize]) -> Result<DynTensor> {
    let rank = x.shape.len();
    if perm.len() != rank {
        bail!(
            "Transpose perm {:?} does not match input rank {} ({:?})",
            perm,
            rank,
            x.shape
        );
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let in_strides = compute_strides(&x.shape);
    let out_strides = compute_strides(&out_shape);
    let total = x.numel();
    let mut out = vec![0.0f32; total];

    for flat_in in 0..total {
        let mut rem = flat_in;
        let mut flat_out = 0usize;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            // Find where dimension d lands in perm
            for (out_d, &p) in perm.iter().enumerate() {
                if p == d {
                    flat_out += coord * out_strides[out_d];
                    break;
                }
            }
        }
        out[flat_out] = x.data[flat_in];
    }

    DynTensor::new(out_shape, out, x.device)
}

pub(crate) fn exec_concat(
    tensors: &[DynTensor],
    axis: usize,
    device: SynapseMlDevice,
) -> Result<DynTensor> {
    let first = tensors.first().context("Concat requires at least 1 input")?;
    let rank = first.shape.len();
    if axis >= rank {
        bail!("Concat axis {} out of bounds for rank {}", axis, rank);
    }

    let mut out_shape = first.shape.clone();
    out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = first.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = first.shape[axis + 1..].iter().product::<usize>().max(1);
    let mut out_data = Vec::with_capacity(out_shape.iter().product());

    for o in 0..outer {
        for t in tensors {
            let dim_t = t.shape[axis];
            let chunk = dim_t * inner;
            let start = o * chunk;
            out_data.extend_from_slice(&t.data[start..start + chunk]);
        }
    }

    DynTensor::new(out_shape, out_data, device)
}
