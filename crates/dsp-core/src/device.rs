use serde::{Deserialize, Serialize};
use std::fmt;

/// Compute device target for DSP kernel acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DeviceTarget {
    /// Automatically select highest-performance discrete GPU, fallback to CPU JIT.
    #[default]
    Auto,

    /// Bare-metal CPU vectorization via CubeCL JIT / SIMD.
    Cpu,

    /// WebGPU / Vulkan / Metal / DX12 runtime via WGPU.
    Wgpu(usize),

    /// Dedicated NVIDIA CUDA runtime.
    Cuda(usize),

    /// Dedicated AMD ROCm / HIP runtime.
    Hip(usize),
}

impl fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "Auto (Fastest Available)"),
            Self::Cpu => write!(f, "CPU (Hardware-Aware JIT)"),
            Self::Wgpu(idx) => write!(f, "WGPU Adapter {idx}"),
            Self::Cuda(idx) => write!(f, "CUDA Device {idx}"),
            Self::Hip(idx) => write!(f, "HIP Device {idx}"),
        }
    }
}
