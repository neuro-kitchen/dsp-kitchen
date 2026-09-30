//! 1-D convolution and pooling.

use anyhow::{Result, bail};
use onnx_ir::node::padding::PaddingConfig1d;

use super::super::tensor::DynTensor;
use crate::backend::{Tensor1D, Tensor3D};

pub(crate) fn exec_conv1d(
    x: &DynTensor,
    w: &DynTensor,
    bias: Option<&DynTensor>,
    cfg: &onnx_ir::node::conv1d::Conv1dConfig,
) -> Result<DynTensor> {
    if x.shape.len() != 3 || w.shape.len() != 3 {
        bail!(
            "Conv1d expects 3D input [N, C_in, L] and 3D weight [C_out, C_in/groups, K], got {:?} and {:?}",
            x.shape,
            w.shape
        );
    }
    let [batch_size, c_in, l_in] = [x.shape[0], x.shape[1], x.shape[2]];
    let [c_out, c_in_per_group, k_size] = [w.shape[0], w.shape[1], w.shape[2]];
    let groups = cfg.groups.max(1);
    let stride = cfg.stride.max(1);
    let dilation = cfg.dilation.max(1);

    if c_in != c_in_per_group * groups {
        bail!(
            "Conv1d channel mismatch: c_in={} vs c_in_per_group={} * groups={}",
            c_in,
            c_in_per_group,
            groups
        );
    }

    let (pad_left, pad_right) = match cfg.padding {
        PaddingConfig1d::Valid => (0usize, 0usize),
        PaddingConfig1d::Explicit(l, r) => (l, r),
    };

    // Fast path for standard 1D convolution (groups == 1, dilation == 1, symmetric padding)
    if groups == 1 && dilation == 1 && pad_left == pad_right {
        let bias_1d = match bias {
            Some(b) => Tensor1D::from_floats(b.data.to_vec(), [c_out], x.device),
            None => Tensor1D::zeros([c_out], x.device),
        };
        let layer = crate::backbones::Conv1dLayer {
            weight: Tensor3D::from_floats(w.data.to_vec(), [c_out, c_in, k_size], x.device),
            bias: bias_1d,
            stride,
            padding: pad_left,
        };
        let x_3d = Tensor3D::from_floats(x.data.to_vec(), [batch_size, c_in, l_in], x.device);
        let out_3d = layer.forward(&x_3d);
        return Ok(DynTensor::from_tensor(&out_3d));
    }

    // General path supporting arbitrary dilation, groups, and asymmetric padding
    let eff_k = (k_size - 1) * dilation + 1;
    let padded_len = l_in + pad_left + pad_right;
    if padded_len < eff_k {
        bail!(
            "Conv1d padded length {} is smaller than dilated kernel {}",
            padded_len,
            eff_k
        );
    }
    let l_out = (padded_len - eff_k) / stride + 1;
    let c_out_per_group = c_out / groups;
    let mut out = vec![0.0f32; batch_size * c_out * l_out];

    for b in 0..batch_size {
        for g in 0..groups {
            for co_g in 0..c_out_per_group {
                let co = g * c_out_per_group + co_g;
                let b_val = bias.map(|bt| bt.data[co]).unwrap_or(0.0);
                for o in 0..l_out {
                    let in_start = (o * stride) as isize - (pad_left as isize);
                    let mut acc = b_val;
                    for ci_g in 0..c_in_per_group {
                        let ci = g * c_in_per_group + ci_g;
                        let x_off = (b * c_in + ci) * l_in;
                        let w_off = (co * c_in_per_group + ci_g) * k_size;
                        for k in 0..k_size {
                            let pos = in_start + (k * dilation) as isize;
                            if pos >= 0 && (pos as usize) < l_in {
                                acc += x.data[x_off + (pos as usize)] * w.data[w_off + k];
                            }
                        }
                    }
                    out[(b * c_out + co) * l_out + o] = acc;
                }
            }
        }
    }

    DynTensor::new(vec![batch_size, c_out, l_out], out, x.device)
}

pub(crate) fn exec_avg_pool1d(x: &DynTensor, kernel_size: usize, stride: usize) -> Result<DynTensor> {
    if x.shape.len() != 3 {
        bail!("AveragePool1d expects 3D [N, C, T], got {:?}", x.shape);
    }
    let [n, c, t] = [x.shape[0], x.shape[1], x.shape[2]];
    if t < kernel_size || kernel_size == 0 || stride == 0 {
        return Ok(x.clone());
    }
    let t_out = (t - kernel_size) / stride + 1;
    let inv_k = 1.0 / (kernel_size as f32);
    let mut out = vec![0.0f32; n * c * t_out];

    for b in 0..n {
        for ch in 0..c {
            let in_off = (b * c + ch) * t;
            let out_off = (b * c + ch) * t_out;
            for o in 0..t_out {
                let s = o * stride;
                let sum: f32 = x.data[in_off + s..in_off + s + kernel_size].iter().sum();
                out[out_off + o] = sum * inv_k;
            }
        }
    }
    DynTensor::new(vec![n, c, t_out], out, x.device)
}
