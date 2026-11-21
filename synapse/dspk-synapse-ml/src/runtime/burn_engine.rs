//! Burn tensor execution engine bridged to [`dspk_core::ComputeTarget`].
//!
//! Executes pretrained weights (from `.safetensors`, `.npy`, or `.onnx` initializers)
//! using Burn's wgpu (CubeCL) or Flex (CPU) backend without synthetic random initialization or
//! hardcoded tensor dimensions.

use burn_tensor::module::conv1d;
use burn_tensor::ops::ConvOptions;
use burn_tensor::{Device, Shape, Tensor, TensorData};
use dspk_core::{ComputeTarget, DspError, DspResult};

/// The Burn device for `target`: wgpu for [`ComputeTarget::Wgpu`], Flex (CPU) otherwise.
pub(crate) fn burn_device(target: ComputeTarget) -> DspResult<Device> {
    match target {
        #[cfg(feature = "wgpu")]
        ComputeTarget::Wgpu => Ok(Device::wgpu(burn_tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "flex")]
        _ => Ok(Device::flex()),
        #[cfg(not(feature = "flex"))]
        other => Err(DspError::InvalidConfig(format!("Burn backend not compiled for compute target {other}"))),
    }
}

/// Pre-uploaded `f32` linear layer weights stored already transposed `[in_features, out_features]` on
/// the active device.
#[derive(Clone, Debug)]
pub(crate) struct PreparedLinear2d {
    pub(crate) weight_t: Tensor<2>,
    pub(crate) bias: Option<Tensor<2>>,
}

impl PreparedLinear2d {
    /// Uploads `weight` (`[out_features, in_features]` row-major) and optional `bias` (`[out_features]`)
    /// to `dev`, transposing `weight` once on upload.
    pub(crate) fn new(
        dev: &Device,
        weight: &[f32],
        out_features: usize,
        in_features: usize,
        bias: Option<&[f32]>,
    ) -> DspResult<Self> {
        if weight.len() != out_features * in_features {
            return Err(DspError::InvalidConfig(format!(
                "PreparedLinear2d: weight len {} != out_features ({out_features}) * in_features ({in_features})",
                weight.len()
            )));
        }
        if let Some(b) = bias {
            if b.len() != out_features {
                return Err(DspError::InvalidConfig(format!(
                    "PreparedLinear2d: bias len {} != out_features ({out_features})",
                    b.len()
                )));
            }
        }
        let weight_t = Tensor::<2>::from_data(
            TensorData::new(weight.to_vec(), Shape::new([out_features, in_features])),
            dev,
        )
        .transpose();
        let bias = bias.map(|b| {
            Tensor::<2>::from_data(TensorData::new(b.to_vec(), Shape::new([1, out_features])), dev)
        });
        Ok(Self { weight_t, bias })
    }

    /// Creates from a pre-transposed `[in_features, out_features]` weight slice.
    pub(crate) fn from_pretransposed(
        dev: &Device,
        weight_t: &[f32],
        in_features: usize,
        out_features: usize,
        bias: Option<&[f32]>,
    ) -> DspResult<Self> {
        if weight_t.len() != in_features * out_features {
            return Err(DspError::InvalidConfig(format!(
                "PreparedLinear2d: weight_t len {} != [{in_features}, {out_features}]",
                weight_t.len()
            )));
        }
        if let Some(b) = bias {
            if b.len() != out_features {
                return Err(DspError::InvalidConfig(format!(
                    "PreparedLinear2d: bias len {} != out_features ({out_features})",
                    b.len()
                )));
            }
        }
        let weight_t = Tensor::<2>::from_data(
            TensorData::new(weight_t.to_vec(), Shape::new([in_features, out_features])),
            dev,
        );
        let bias = bias.map(|b| {
            Tensor::<2>::from_data(TensorData::new(b.to_vec(), Shape::new([1, out_features])), dev)
        });
        Ok(Self { weight_t, bias })
    }

    pub(crate) fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.weight_t.clone());
        match &self.bias {
            Some(b) => y + b.clone(),
            None => y,
        }
    }
}

/// Pre-uploaded `f32` 1D convolution kernel `[out_channels, in_channels / groups, kernel_size]` and bias.
#[derive(Clone, Debug)]
pub(crate) struct PreparedConv1d {
    pub(crate) weight: Tensor<3>,
    pub(crate) bias: Option<Tensor<1>>,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) dilation: usize,
    pub(crate) groups: usize,
}

impl PreparedConv1d {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: &Device,
        weight: &[f32],
        out_channels: usize,
        in_per_group: usize,
        kernel_size: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> DspResult<Self> {
        let expected_w = out_channels * in_per_group * kernel_size;
        if weight.len() != expected_w {
            return Err(DspError::InvalidConfig(format!(
                "PreparedConv1d: weight len {} != [{out_channels}, {in_per_group}, {kernel_size}]",
                weight.len()
            )));
        }
        let weight = Tensor::<3>::from_data(
            TensorData::new(weight.to_vec(), Shape::new([out_channels, in_per_group, kernel_size])),
            dev,
        );
        let bias = bias.map(|b_slice| {
            Tensor::<1>::from_data(TensorData::new(b_slice.to_vec(), Shape::new([out_channels])), dev)
        });
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let options = ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups);
        conv1d(x, self.weight.clone(), self.bias.clone(), options)
    }
}

/// Executes a 2D linear / GEMM projection `Y = X * W^T + b` on Burn tensors.
///
/// - `input`: `[batch, in_features]`
/// - `weight`: `[out_features, in_features]`
/// - `bias`: optional `[out_features]`
/// - Returns: flat `[batch, out_features]`
pub fn burn_linear_2d(
    target: ComputeTarget,
    input: &[f32],
    batch: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
) -> DspResult<Vec<f32>> {
    if input.len() != batch * in_features {
        return Err(DspError::InvalidConfig(format!(
            "burn_linear_2d: input len {} != batch ({batch}) * in_features ({in_features})",
            input.len()
        )));
    }
    if weight.len() != out_features * in_features {
        return Err(DspError::InvalidConfig(format!(
            "burn_linear_2d: weight len {} != out_features ({out_features}) * in_features ({in_features})",
            weight.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(DspError::InvalidConfig(format!(
                "burn_linear_2d: bias len {} != out_features ({out_features})",
                b.len()
            )));
        }
    }

    Ok(linear_on_device(&burn_device(target)?, input, batch, in_features, weight, out_features, bias))
}

fn linear_on_device(
    dev: &Device,
    input: &[f32],
    batch: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let x = Tensor::<2>::from_data(
        TensorData::new(input.to_vec(), Shape::new([batch, in_features])),
        dev,
    );
    let w = Tensor::<2>::from_data(
        TensorData::new(weight.to_vec(), Shape::new([out_features, in_features])),
        dev,
    );
    let mut y = x.matmul(w.transpose());
    if let Some(b) = bias {
        let b_tensor = Tensor::<2>::from_data(
            TensorData::new(b.to_vec(), Shape::new([1, out_features])),
            dev,
        );
        y = y + b_tensor;
    }
    y.into_data().try_to_vec::<f32>().expect("f32 tensor output")
}

/// Executes a 1D convolution on `[batch, in_channels, length]` with weights
/// `[out_channels, in_channels / groups, kernel_size]` using Burn's `conv1d` primitive.
///
/// Returns `(flat_output, out_length)`.
#[allow(clippy::too_many_arguments)]
pub fn burn_conv1d(
    target: ComputeTarget,
    input: &[f32],
    batch: usize,
    in_channels: usize,
    length: usize,
    weight: &[f32],
    out_channels: usize,
    kernel_size: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> DspResult<(Vec<f32>, usize)> {
    let expected_in = batch * in_channels * length;
    if input.len() != expected_in {
        return Err(DspError::InvalidConfig(format!(
            "burn_conv1d: input len {} != [{batch}, {in_channels}, {length}]",
            input.len()
        )));
    }
    let in_per_group = in_channels / groups.max(1);
    let expected_w = out_channels * in_per_group * kernel_size;
    if weight.len() != expected_w {
        return Err(DspError::InvalidConfig(format!(
            "burn_conv1d: weight len {} != [{out_channels}, {in_per_group}, {kernel_size}]",
            weight.len()
        )));
    }

    let dev = burn_device(target)?;
    Ok(conv1d_on_device(
        &dev, input, batch, in_channels, length, weight, out_channels, in_per_group, kernel_size, bias, stride, padding,
        dilation, groups,
    ))
}

#[allow(clippy::too_many_arguments)]
fn conv1d_on_device(
    dev: &Device,
    input: &[f32],
    batch: usize,
    in_channels: usize,
    length: usize,
    weight: &[f32],
    out_channels: usize,
    in_per_group: usize,
    kernel_size: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> (Vec<f32>, usize) {
    let x = Tensor::<3>::from_data(
        TensorData::new(input.to_vec(), Shape::new([batch, in_channels, length])),
        dev,
    );
    let w = Tensor::<3>::from_data(
        TensorData::new(
            weight.to_vec(),
            Shape::new([out_channels, in_per_group, kernel_size]),
        ),
        dev,
    );
    let b = bias.map(|b_slice| {
        Tensor::<1>::from_data(
            TensorData::new(b_slice.to_vec(), Shape::new([out_channels])),
            dev,
        )
    });
    let options = ConvOptions::new([stride], [padding], [dilation], groups);
    let y = conv1d(x, w, b, options);
    let out_len = y.dims()[2];
    let data = y.into_data().try_to_vec::<f32>().expect("f32 tensor output");
    (data, out_len)
}
