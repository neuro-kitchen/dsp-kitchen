//! Linear, MatMul and Gemm.

use anyhow::{Result, bail};

use super::super::tensor::DynTensor;
use crate::backend::{Tensor1D, Tensor2D};
use crate::hub::transpose_2d_slice;

pub(crate) fn exec_linear(
    x: &DynTensor,
    w: &DynTensor,
    bias: Option<&DynTensor>,
    transpose_weight: bool,
) -> Result<DynTensor> {
    if x.shape.is_empty() || w.shape.len() != 2 {
        bail!(
            "Linear expects input rank >= 1 and 2D weight, got input {:?} and weight {:?}",
            x.shape,
            w.shape
        );
    }
    let in_features = *x.shape.last().unwrap();
    let batch_rows: usize = x.shape[..x.shape.len() - 1].iter().product::<usize>().max(1);

    // If transpose_weight is true (PyTorch / ONNX Gemm transB=1 convention), W is [out_features, in_features].
    // Otherwise W is [in_features, out_features].
    let (w_in_out, out_features) = if transpose_weight {
        let [out_f, in_f] = [w.shape[0], w.shape[1]];
        if in_f != in_features {
            bail!(
                "Linear(transpose_weight=true) feature mismatch: input has {}, weight is [{}, {}]",
                in_features,
                out_f,
                in_f
            );
        }
        let transposed = transpose_2d_slice(&w.data, out_f, in_f);
        (
            Tensor2D::from_floats(transposed, [in_f, out_f], x.device),
            out_f,
        )
    } else {
        let [in_f, out_f] = [w.shape[0], w.shape[1]];
        if in_f != in_features {
            bail!(
                "Linear(transpose_weight=false) feature mismatch: input has {}, weight is [{}, {}]",
                in_features,
                in_f,
                out_f
            );
        }
        (
            Tensor2D::from_floats(w.data.to_vec(), [in_f, out_f], x.device),
            out_f,
        )
    };

    let x_2d = Tensor2D::from_floats(x.data.to_vec(), [batch_rows, in_features], x.device);
    let mut y_2d = x_2d.matmul(&w_in_out);

    if let Some(b) = bias {
        if b.numel() == out_features {
            let b_1d = Tensor1D::from_floats(b.data.to_vec(), [out_features], x.device);
            y_2d = y_2d.add_bias_1d(&b_1d);
        } else if b.numel() == 1 {
            let s = b.data[0];
            for v in &mut y_2d.data {
                *v += s;
            }
        } else {
            bail!(
                "Linear bias length {} does not match out_features {}",
                b.numel(),
                out_features
            );
        }
    }

    let mut out_shape = x.shape[..x.shape.len() - 1].to_vec();
    out_shape.push(out_features);
    DynTensor::new(out_shape, y_2d.into_vec(), x.device)
}

pub(crate) fn exec_gemm(
    a: &DynTensor,
    b: &DynTensor,
    c: Option<&DynTensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<DynTensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        bail!("Gemm requires rank-2 matrices, got {:?} and {:?}", a.shape, b.shape);
    }
    let inner_a = if trans_a { a.shape[0] } else { a.shape[1] };
    let inner_b = if trans_b { b.shape[1] } else { b.shape[0] };
    if inner_a != inner_b {
        bail!(
            "Gemm inner dimensions differ: A {:?}{} · B {:?}{}",
            a.shape,
            if trans_a { "ᵀ" } else { "" },
            b.shape,
            if trans_b { "ᵀ" } else { "" }
        );
    }
    let a_mat = if trans_a {
        let [r, col] = [a.shape[0], a.shape[1]];
        Tensor2D::from_floats(transpose_2d_slice(&a.data, r, col), [col, r], a.device)
    } else {
        Tensor2D::from_floats(a.data.to_vec(), [a.shape[0], a.shape[1]], a.device)
    };

    let b_mat = if trans_b {
        let [r, col] = [b.shape[0], b.shape[1]];
        Tensor2D::from_floats(transpose_2d_slice(&b.data, r, col), [col, r], b.device)
    } else {
        Tensor2D::from_floats(b.data.to_vec(), [b.shape[0], b.shape[1]], b.device)
    };

    let mut y = a_mat.matmul(&b_mat);
    if (alpha - 1.0).abs() > 1e-7 {
        y = y.mul_scalar(alpha);
    }

    if let Some(c_tensor) = c
        && beta.abs() > 0.0
    {
        let [m, n] = y.shape;
        if c_tensor.numel() == n {
            for r in 0..m {
                for col in 0..n {
                    y.data[r * n + col] += beta * c_tensor.data[col];
                }
            }
        } else if c_tensor.numel() == m * n {
            for i in 0..(m * n) {
                y.data[i] += beta * c_tensor.data[i];
            }
        } else if c_tensor.numel() == 1 {
            let s = beta * c_tensor.data[0];
            for v in &mut y.data {
                *v += s;
            }
        } else {
            bail!(
                "Unsupported Gemm C broadcast shape {:?} for output {:?}",
                c_tensor.shape,
                y.shape
            );
        }
    }

    Ok(DynTensor::from_tensor(&y))
}
