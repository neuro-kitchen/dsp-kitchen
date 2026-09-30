//! NumPy-style broadcasting of binary operators.

use anyhow::{Result, bail};

use super::super::tensor::DynTensor;

pub(crate) fn broadcast_binary_op<F>(a: &DynTensor, b: &DynTensor, op: F) -> Result<DynTensor>
where
    F: Fn(f32, f32) -> f32,
{
    if a.shape == b.shape {
        let data = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return DynTensor::new(a.shape.clone(), data, a.device);
    }
    if b.numel() == 1 {
        let s = b.data[0];
        let data = a.data.iter().map(|&x| op(x, s)).collect();
        return DynTensor::new(a.shape.clone(), data, a.device);
    }
    if a.numel() == 1 {
        let s = a.data[0];
        let data = b.data.iter().map(|&y| op(s, y)).collect();
        return DynTensor::new(b.shape.clone(), data, b.device);
    }

    // General N-D NumPy/ONNX right-aligned broadcasting
    let rank = a.shape.len().max(b.shape.len());
    let mut out_shape = vec![1usize; rank];
    let mut a_padded = vec![1usize; rank];
    let mut b_padded = vec![1usize; rank];

    for i in 0..a.shape.len() {
        a_padded[rank - a.shape.len() + i] = a.shape[i];
    }
    for i in 0..b.shape.len() {
        b_padded[rank - b.shape.len() + i] = b.shape[i];
    }
    for i in 0..rank {
        if a_padded[i] == b_padded[i] {
            out_shape[i] = a_padded[i];
        } else if a_padded[i] == 1 {
            out_shape[i] = b_padded[i];
        } else if b_padded[i] == 1 {
            out_shape[i] = a_padded[i];
        } else {
            bail!(
                "Incompatible broadcast shapes {:?} and {:?}",
                a.shape,
                b.shape
            );
        }
    }

    let out_strides = compute_strides(&out_shape);
    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);
    let total: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; total];

    for flat in 0..total {
        let mut rem = flat;
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        for d in 0..rank {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let ac = if a_padded[d] == 1 { 0 } else { coord };
            let bc = if b_padded[d] == 1 { 0 } else { coord };
            a_idx += ac * a_strides[d];
            b_idx += bc * b_strides[d];
        }
        out_data[flat] = op(a.data[a_idx], b.data[b_idx]);
    }

    DynTensor::new(out_shape, out_data, a.device)
}

pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1].max(1);
    }
    strides
}
