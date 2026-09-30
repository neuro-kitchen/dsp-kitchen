//! Normalization and softmax.

use anyhow::{Result, bail};

use super::super::tensor::DynTensor;

pub(crate) fn exec_batch_norm(
    x: &DynTensor,
    scale: &DynTensor,
    bias: &DynTensor,
    mean: &DynTensor,
    var: &DynTensor,
    eps: f32,
) -> Result<DynTensor> {
    if x.shape.len() < 2 {
        bail!("BatchNormalization expects rank >= 2, got {:?}", x.shape);
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; x.data.len()];

    for ch in 0..c {
        let m = mean.data[ch];
        let inv_std = 1.0 / (var.data[ch] + eps).sqrt();
        let sc = scale.data[ch] * inv_std;
        let sh = bias.data[ch] - m * sc;
        for b in 0..n {
            let off = (b * c + ch) * spatial;
            for s in 0..spatial {
                out[off + s] = x.data[off + s] * sc + sh;
            }
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}

pub(crate) fn exec_layer_norm(
    x: &DynTensor,
    scale: &DynTensor,
    bias: Option<&DynTensor>,
    eps: f32,
) -> Result<DynTensor> {
    if x.shape.is_empty() {
        bail!("LayerNormalization requires rank >= 1");
    }
    let norm_dim = scale.numel();
    let outer = x.numel() / norm_dim.max(1);
    if outer * norm_dim != x.numel() {
        bail!(
            "LayerNormalization scale size {} does not divide input shape {:?}",
            norm_dim,
            x.shape
        );
    }
    let mut out = vec![0.0f32; x.numel()];
    let inv_d = 1.0 / (norm_dim as f32);

    for r in 0..outer {
        let row = &x.data[r * norm_dim..(r + 1) * norm_dim];
        let mean: f32 = row.iter().sum::<f32>() * inv_d;
        let var: f32 = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * inv_d;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..norm_dim {
            let b_val = bias.map(|b| b.data[c]).unwrap_or(0.0);
            out[r * norm_dim + c] = (row[c] - mean) * inv_std * scale.data[c] + b_val;
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}

pub(crate) fn exec_softmax(x: &DynTensor, axis: usize) -> Result<DynTensor> {
    if axis >= x.shape.len() {
        bail!("Softmax axis {} out of bounds for shape {:?}", axis, x.shape);
    }
    let outer: usize = x.shape[..axis].iter().product::<usize>().max(1);
    let dim = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; x.numel()];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                if x.data[idx] > max_v {
                    max_v = x.data[idx];
                }
            }
            let mut sum = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                let e = (x.data[idx] - max_v).exp();
                out[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                out[idx] *= inv;
            }
        }
    }
    DynTensor::new(x.shape.clone(), out, x.device)
}
