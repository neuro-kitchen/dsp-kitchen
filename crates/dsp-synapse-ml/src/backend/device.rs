//! Burn-idiomatic compute device and multi-dimensional `Tensor<const D: usize>` backend
//! powered by `ndarray` (matrixmultiply BLAS) and `cubecl` (WGPU/CPU).

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

/// Target execution device for `dsp-synapse-ml` neural inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynapseMlDevice {
    /// Cross-platform GPU execution via WebGPU / Vulkan / Metal (`cubecl-wgpu`).
    Wgpu(usize),
    /// Portable multi-threaded CPU execution via `ndarray` / `matrixmultiply`.
    Cpu,
}

impl Default for SynapseMlDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Multi-dimensional contiguous `f32` Tensor of rank `D` (`D = 1, 2, 3`).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<const D: usize> {
    pub data: Vec<f32>,
    pub shape: [usize; D],
    pub device: SynapseMlDevice,
}

pub type Tensor1D = Tensor<1>;
pub type Tensor2D = Tensor<2>;
pub type Tensor3D = Tensor<3>;

impl<const D: usize> Tensor<D> {
    /// Constructs a tensor from a contiguous `Vec<f32>` without copying.
    pub fn from_floats(data: Vec<f32>, shape: [usize; D], device: SynapseMlDevice) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Tensor shape {:?} expects {} elements, got {}",
            shape,
            expected,
            data.len()
        );
        Self { data, shape, device }
    }

    /// Allocates a zero-filled tensor of `shape`.
    pub fn zeros(shape: [usize; D], device: SynapseMlDevice) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; len],
            shape,
            device,
        }
    }

    /// Allocates a one-filled tensor of `shape`.
    pub fn ones(shape: [usize; D], device: SynapseMlDevice) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![1.0f32; len],
            shape,
            device,
        }
    }

    /// Deterministic Xavier/Glorot uniform initialization seeded by `seed`.
    pub fn xavier_uniform(
        shape: [usize; D],
        fan_in: usize,
        fan_out: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let len: usize = shape.iter().product();
        let limit = (6.0f32 / (fan_in.max(1) + fan_out.max(1)) as f32).sqrt();
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            // SplitMix64 PRNG
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z = z ^ (z >> 31);
            let u = (z >> 40) as f32 / ((1u64 << 24) as f32); // [0, 1)
            data.push((u * 2.0 - 1.0) * limit);
        }
        Self { data, shape, device }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Rectified Linear Unit: $\max(0, x)$.
    pub fn relu(&self) -> Self {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Gaussian Error Linear Unit (Hendrycks & Gimpel tanh approximation):
    /// $\text{GELU}(x) \approx 0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$.
    pub fn gelu(&self) -> Self {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        let data = self
            .data
            .iter()
            .map(|&x| {
                let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Logistic sigmoid activation: $1 / (1 + e^{-x})$.
    pub fn sigmoid(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Hyperbolic tangent activation.
    pub fn tanh(&self) -> Self {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Elementwise tensor addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Elementwise add shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Elementwise tensor subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Elementwise sub shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }

    /// Scales all tensor elements by a scalar constant.
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Self {
            data,
            shape: self.shape,
            device: self.device,
        }
    }
}

impl Tensor2D {
    /// Matrix multiplication `[M, K] x [K, N] -> [M, N]` accelerated by `ndarray` (`matrixmultiply`).
    pub fn matmul(&self, rhs: &Tensor2D) -> Tensor2D {
        let [m, k1] = self.shape;
        let [k2, n] = rhs.shape;
        assert_eq!(k1, k2, "matmul inner dimension mismatch: {} vs {}", k1, k2);

        if m == 0 || n == 0 || k1 == 0 {
            return Tensor2D::zeros([m, n], self.device);
        }

        let a = ArrayView2::from_shape((m, k1), &self.data).expect("Valid LHS shape");
        let b = ArrayView2::from_shape((k2, n), &rhs.data).expect("Valid RHS shape");
        let c = a.dot(&b);
        let (vec, _offset) = c.into_raw_vec_and_offset();
        Tensor2D::from_floats(vec, [m, n], self.device)
    }

    /// Adds a 1D bias `[N]` across rows of `[M, N]`.
    pub fn add_bias_1d(&self, bias: &Tensor1D) -> Tensor2D {
        let [m, n] = self.shape;
        assert_eq!(bias.shape[0], n, "Bias dimension mismatch");
        let mut out = self.data.clone();
        for r in 0..m {
            let row_off = r * n;
            for c in 0..n {
                out[row_off + c] += bias.data[c];
            }
        }
        Tensor2D::from_floats(out, [m, n], self.device)
    }

    /// Row-wise numerically stable Softmax over the last dimension `[M, N]`.
    pub fn softmax(&self) -> Tensor2D {
        let [m, n] = self.shape;
        let mut out = vec![0.0f32; m * n];
        if n == 0 {
            return Tensor2D::from_floats(out, [m, n], self.device);
        }
        for r in 0..m {
            let row = &self.data[r * n..(r + 1) * n];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..n {
                let e = (row[c] - max_v).exp();
                out[r * n + c] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for c in 0..n {
                out[r * n + c] *= inv;
            }
        }
        Tensor2D::from_floats(out, [m, n], self.device)
    }

    /// Row-wise $L_2$ unit-hypersphere normalization (for Contrastive SimCLR embeddings).
    pub fn l2_normalize(&self, eps: f32) -> Tensor2D {
        let [m, n] = self.shape;
        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            let row = &self.data[r * n..(r + 1) * n];
            let norm_sq: f32 = row.iter().map(|&v| v * v).sum();
            let inv_norm = 1.0 / (norm_sq.sqrt().max(eps));
            for c in 0..n {
                out[r * n + c] = row[c] * inv_norm;
            }
        }
        Tensor2D::from_floats(out, [m, n], self.device)
    }

    /// Reshapes `[N, K * T]` into a 3D tensor `[N, K, T]`.
    pub fn unflatten_channels_time(self, k: usize, t: usize) -> Tensor3D {
        let [n, kt] = self.shape;
        assert_eq!(kt, k * t, "Cannot unflatten {} into {} x {}", kt, k, t);
        Tensor3D::from_floats(self.data, [n, k, t], self.device)
    }
}

impl Tensor3D {
    /// Flattens `[N, K, T]` into a 2D matrix `[N, K * T]` without copying.
    pub fn flatten_channels_time(self) -> Tensor2D {
        let [n, k, t] = self.shape;
        Tensor2D::from_floats(self.data, [n, k * t], self.device)
    }

    /// Global Average Pooling over temporal dimension `T`: `[N, C, T] -> [N, C]`.
    pub fn global_avg_pool_1d(&self) -> Tensor2D {
        let [n, c, t] = self.shape;
        let mut out = vec![0.0f32; n * c];
        if t == 0 {
            return Tensor2D::from_floats(out, [n, c], self.device);
        }
        let inv_t = 1.0 / (t as f32);
        for b in 0..n {
            for ch in 0..c {
                let slice = &self.data[(b * c + ch) * t..(b * c + ch + 1) * t];
                let sum: f32 = slice.iter().sum();
                out[b * c + ch] = sum * inv_t;
            }
        }
        Tensor2D::from_floats(out, [n, c], self.device)
    }

    /// Global Max Pooling over temporal dimension `T`: `[N, C, T] -> [N, C]`.
    pub fn global_max_pool_1d(&self) -> Tensor2D {
        let [n, c, t] = self.shape;
        let mut out = vec![0.0f32; n * c];
        if t == 0 {
            return Tensor2D::from_floats(out, [n, c], self.device);
        }
        for b in 0..n {
            for ch in 0..c {
                let slice = &self.data[(b * c + ch) * t..(b * c + ch + 1) * t];
                let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                out[b * c + ch] = max_v;
            }
        }
        Tensor2D::from_floats(out, [n, c], self.device)
    }

    /// 1D Max Pooling along the temporal axis: `[N, C, T] -> [N, C, T_out]`.
    pub fn max_pool1d(&self, kernel_size: usize, stride: usize) -> Tensor3D {
        let [n, c, t] = self.shape;
        assert!(kernel_size > 0 && stride > 0);
        if t < kernel_size {
            return self.clone();
        }
        let t_out = (t - kernel_size) / stride + 1;
        let mut out = vec![0.0f32; n * c * t_out];
        for b in 0..n {
            for ch in 0..c {
                let in_off = (b * c + ch) * t;
                let out_off = (b * c + ch) * t_out;
                for o in 0..t_out {
                    let s = o * stride;
                    let mut m = f32::NEG_INFINITY;
                    for k in 0..kernel_size {
                        let v = self.data[in_off + s + k];
                        if v > m {
                            m = v;
                        }
                    }
                    out[out_off + o] = m;
                }
            }
        }
        Tensor3D::from_floats(out, [n, c, t_out], self.device)
    }

    /// Nearest-neighbor 1D temporal upsampling by `scale_factor`: `[N, C, T] -> [N, C, T * scale_factor]`.
    pub fn upsample_nearest_1d(&self, scale_factor: usize) -> Tensor3D {
        let [n, c, t] = self.shape;
        let t_out = t * scale_factor;
        let mut out = vec![0.0f32; n * c * t_out];
        for b in 0..n {
            for ch in 0..c {
                let in_off = (b * c + ch) * t;
                let out_off = (b * c + ch) * t_out;
                for i in 0..t {
                    let val = self.data[in_off + i];
                    for s in 0..scale_factor {
                        out[out_off + i * scale_factor + s] = val;
                    }
                }
            }
        }
        Tensor3D::from_floats(out, [n, c, t_out], self.device)
    }

    /// Crops or zero-pads the temporal axis `T` to match `target_len`.
    pub fn match_temporal_length(&self, target_len: usize) -> Tensor3D {
        let [n, c, t] = self.shape;
        if t == target_len {
            return self.clone();
        }
        let mut out = vec![0.0f32; n * c * target_len];
        let copy_len = t.min(target_len);
        for b in 0..n {
            for ch in 0..c {
                let in_off = (b * c + ch) * t;
                let out_off = (b * c + ch) * target_len;
                out[out_off..out_off + copy_len]
                    .copy_from_slice(&self.data[in_off..in_off + copy_len]);
            }
        }
        Tensor3D::from_floats(out, [n, c, target_len], self.device)
    }

    /// Concatenates two 3D tensors along the channel axis `C` (`dim = 1`):
    /// `[N, C1, T]` + `[N, C2, T]` -> `[N, C1 + C2, T]`.
    pub fn concat_channels(&self, other: &Tensor3D) -> Tensor3D {
        let [n1, c1, t1] = self.shape;
        let [n2, c2, t2] = other.shape;
        assert_eq!(n1, n2, "Batch size mismatch in concat_channels");
        assert_eq!(t1, t2, "Temporal length mismatch in concat_channels");

        let c_out = c1 + c2;
        let mut out = vec![0.0f32; n1 * c_out * t1];
        for b in 0..n1 {
            let dst_b = b * c_out * t1;
            let src1_b = b * c1 * t1;
            let src2_b = b * c2 * t1;
            out[dst_b..dst_b + c1 * t1].copy_from_slice(&self.data[src1_b..src1_b + c1 * t1]);
            out[dst_b + c1 * t1..dst_b + c_out * t1]
                .copy_from_slice(&other.data[src2_b..src2_b + c2 * t1]);
        }
        Tensor3D::from_floats(out, [n1, c_out, t1], self.device)
    }
}
