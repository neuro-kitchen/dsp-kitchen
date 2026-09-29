//! 1D Convolutional layers (`im2col` + BLAS `matmul`), 1D Batch Normalization,
//! and 1D Residual Blocks (`ResBlock1D`).

use anyhow::Result;
use crate::backend::{SynapseMlDevice, Tensor1D, Tensor2D, Tensor3D};
use crate::hub::SafetensorsMap;

/// 1D Convolutional layer operating on `[batch, in_channels, length]`.
#[derive(Debug, Clone)]
pub struct Conv1dLayer {
    /// Filter weights of shape `[out_channels, in_channels, kernel_size]`.
    pub weight: Tensor3D,
    /// Bias of shape `[out_channels]`.
    pub bias: Tensor1D,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1dLayer {
    pub fn new_initialized(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let fan_out = out_channels * kernel_size;
        let weight = Tensor3D::xavier_uniform(
            [out_channels, in_channels, kernel_size],
            fan_in,
            fan_out,
            seed,
            device,
        );
        let bias = Tensor1D::zeros([out_channels], device);
        Self {
            weight,
            bias,
            stride: stride.max(1),
            padding,
        }
    }

    pub fn out_channels(&self) -> usize {
        self.weight.shape[0]
    }

    pub fn in_channels(&self) -> usize {
        self.weight.shape[1]
    }

    pub fn kernel_size(&self) -> usize {
        self.weight.shape[2]
    }

    /// Forward pass `[N, C_in, L_in] -> [N, C_out, L_out]` using `im2col` + `Tensor2D::matmul`.
    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let [batch_size, c_in, l_in] = input.shape;
        let [c_out, w_c_in, k_size] = self.weight.shape;
        assert_eq!(
            c_in, w_c_in,
            "Conv1d in_channels mismatch: input has {}, weight expects {}",
            c_in, w_c_in
        );

        let padded_len = l_in + 2 * self.padding;
        assert!(
            padded_len >= k_size,
            "Padded input length {} smaller than kernel_size {}",
            padded_len,
            k_size
        );
        let l_out = (padded_len - k_size) / self.stride + 1;

        // Build im2col matrix of shape [batch_size * l_out, c_in * k_size]
        let col_rows = batch_size * l_out;
        let col_cols = c_in * k_size;
        let mut col_data = vec![0.0f32; col_rows * col_cols];

        for b in 0..batch_size {
            for o in 0..l_out {
                let row_idx = b * l_out + o;
                let row_off = row_idx * col_cols;
                let in_start_signed = (o * self.stride) as isize - (self.padding as isize);

                for ch in 0..c_in {
                    let in_ch_off = (b * c_in + ch) * l_in;
                    let col_ch_off = row_off + ch * k_size;
                    for k in 0..k_size {
                        let pos = in_start_signed + (k as isize);
                        if pos >= 0 && (pos as usize) < l_in {
                            col_data[col_ch_off + k] = input.data[in_ch_off + (pos as usize)];
                        }
                    }
                }
            }
        }

        // Weight matrix reshaped to [c_out, c_in * k_size], transposed to [c_in * k_size, c_out]
        let mut w_t = vec![0.0f32; col_cols * c_out];
        for co in 0..c_out {
            for ck in 0..col_cols {
                w_t[ck * c_out + co] = self.weight.data[co * col_cols + ck];
            }
        }

        let col_mat = Tensor2D::from_floats(col_data, [col_rows, col_cols], input.device);
        let w_mat = Tensor2D::from_floats(w_t, [col_cols, c_out], input.device);
        // [batch_size * l_out, c_out]
        let prod = col_mat.matmul(&w_mat).add_bias_1d(&self.bias);

        // Permute [batch_size, l_out, c_out] -> [batch_size, c_out, l_out]
        let mut out_data = vec![0.0f32; batch_size * c_out * l_out];
        for b in 0..batch_size {
            for o in 0..l_out {
                let src_off = (b * l_out + o) * c_out;
                for co in 0..c_out {
                    out_data[(b * c_out + co) * l_out + o] = prod.data[src_off + co];
                }
            }
        }

        Tensor3D::from_floats(out_data, [batch_size, c_out, l_out], input.device)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        map.insert_tensor(format!("{}.weight", prefix), &self.weight);
        map.insert_tensor(format!("{}.bias", prefix), &self.bias);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.weight = map.get_tensor(&format!("{}.weight", prefix), device)?;
        self.bias = map.get_tensor(&format!("{}.bias", prefix), device)?;
        Ok(())
    }
}

/// 1D Batch Normalization layer over `[batch, channels, length]`.
#[derive(Debug, Clone)]
pub struct BatchNorm1dLayer {
    pub gamma: Tensor1D,
    pub beta: Tensor1D,
    pub running_mean: Tensor1D,
    pub running_var: Tensor1D,
    pub eps: f32,
}

impl BatchNorm1dLayer {
    pub fn new(channels: usize, device: SynapseMlDevice) -> Self {
        Self {
            gamma: Tensor1D::ones([channels], device),
            beta: Tensor1D::zeros([channels], device),
            running_mean: Tensor1D::zeros([channels], device),
            running_var: Tensor1D::ones([channels], device),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let [n, c, l] = input.shape;
        assert_eq!(c, self.gamma.shape[0], "BatchNorm1d channel mismatch");
        let mut out = vec![0.0f32; n * c * l];

        for ch in 0..c {
            let mean = self.running_mean.data[ch];
            let inv_std = 1.0 / (self.running_var.data[ch] + self.eps).sqrt();
            let scale = self.gamma.data[ch] * inv_std;
            let shift = self.beta.data[ch] - mean * scale;

            for b in 0..n {
                let off = (b * c + ch) * l;
                for i in 0..l {
                    out[off + i] = input.data[off + i] * scale + shift;
                }
            }
        }

        Tensor3D::from_floats(out, [n, c, l], input.device)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        map.insert_tensor(format!("{}.weight", prefix), &self.gamma);
        map.insert_tensor(format!("{}.bias", prefix), &self.beta);
        map.insert_tensor(format!("{}.running_mean", prefix), &self.running_mean);
        map.insert_tensor(format!("{}.running_var", prefix), &self.running_var);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.gamma = map.get_tensor(&format!("{}.weight", prefix), device)?;
        self.bias_or_beta_load(prefix, map, device)?;
        self.running_mean = map.get_tensor(&format!("{}.running_mean", prefix), device)?;
        self.running_var = map.get_tensor(&format!("{}.running_var", prefix), device)?;
        Ok(())
    }

    fn bias_or_beta_load(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.beta = map.get_tensor(&format!("{}.bias", prefix), device)?;
        Ok(())
    }
}

/// 1D Residual Convolutional Block (`Conv1d -> BN -> GELU -> Conv1d -> BN + Skip -> GELU`).
#[derive(Debug, Clone)]
pub struct ResBlock1D {
    pub conv1: Conv1dLayer,
    pub bn1: BatchNorm1dLayer,
    pub conv2: Conv1dLayer,
    pub bn2: BatchNorm1dLayer,
    pub shortcut: Option<Conv1dLayer>,
}

impl ResBlock1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let padding = kernel_size / 2;
        let conv1 = Conv1dLayer::new_initialized(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            seed,
            device,
        );
        let bn1 = BatchNorm1dLayer::new(out_channels, device);
        let conv2 = Conv1dLayer::new_initialized(
            out_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            seed.wrapping_add(101),
            device,
        );
        let bn2 = BatchNorm1dLayer::new(out_channels, device);
        let shortcut = if in_channels != out_channels {
            Some(Conv1dLayer::new_initialized(
                in_channels,
                out_channels,
                1,
                1,
                0,
                seed.wrapping_add(202),
                device,
            ))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        }
    }

    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let h1 = self.bn1.forward(&self.conv1.forward(input)).gelu();
        let h2 = self.bn2.forward(&self.conv2.forward(&h1));
        let res = match &self.shortcut {
            Some(proj) => proj.forward(input),
            None => input.clone(),
        };
        let matched_h2 = h2.match_temporal_length(res.shape[2]);
        matched_h2.add(&res).gelu()
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        self.conv1.save_weights(&format!("{}.conv1", prefix), map);
        self.bn1.save_weights(&format!("{}.bn1", prefix), map);
        self.conv2.save_weights(&format!("{}.conv2", prefix), map);
        self.bn2.save_weights(&format!("{}.bn2", prefix), map);
        if let Some(sc) = &self.shortcut {
            sc.save_weights(&format!("{}.shortcut", prefix), map);
        }
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.conv1
            .load_weights(&format!("{}.conv1", prefix), map, device)?;
        self.bn1
            .load_weights(&format!("{}.bn1", prefix), map, device)?;
        self.conv2
            .load_weights(&format!("{}.conv2", prefix), map, device)?;
        self.bn2
            .load_weights(&format!("{}.bn2", prefix), map, device)?;
        if let Some(sc) = &mut self.shortcut {
            sc.load_weights(&format!("{}.shortcut", prefix), map, device)?;
        }
        Ok(())
    }
}
