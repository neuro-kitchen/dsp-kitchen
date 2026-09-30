//! Compute runtime selection at the application edge.
//!
//! Library code is generic over `R: cubecl::Runtime`; the CLI, Python bindings and the app pick a
//! [`ComputeTarget`] once (flag, argument, setting or the `DSP_KITCHEN_RUNTIME` environment
//! variable) and run a [`ComputeTask`] on it. Which targets exist depends on the enabled cargo
//! features (`wgpu`, `cpu`, `cuda`, `hip`).

use cubecl::prelude::*;
use thiserror::Error;

/// Environment variable that overrides [`ComputeTarget::default`] (`wgpu`, `cpu`, `cuda`, `hip`).
pub const RUNTIME_ENV: &str = "DSP_KITCHEN_RUNTIME";

/// A CubeCL runtime and its default device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeTarget {
    /// WebGPU (Vulkan / Metal / DX12).
    Wgpu,
    /// Multi-threaded CPU (MLIR JIT).
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm HIP.
    Hip,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ComputeError {
    #[error("unknown compute runtime '{0}' (expected wgpu, cpu, cuda or hip)")]
    Unknown(String),
    #[error("compute runtime {0:?} is not compiled in (enable the `{1}` feature)")]
    Unavailable(ComputeTarget, &'static str),
    #[error("no compute runtime is compiled in (enable one of the wgpu, cpu, cuda, hip features)")]
    NoneAvailable,
}

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output;
}

impl ComputeTarget {
    pub const ALL: [ComputeTarget; 4] = [Self::Wgpu, Self::Cuda, Self::Hip, Self::Cpu];

    pub fn name(self) -> &'static str {
        match self {
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }

    pub fn parse(name: &str) -> Result<Self, ComputeError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "wgpu" | "webgpu" | "vulkan" | "metal" | "gpu" => Ok(Self::Wgpu),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "hip" | "rocm" => Ok(Self::Hip),
            other => Err(ComputeError::Unknown(other.to_string())),
        }
    }

    /// Whether this runtime is compiled in.
    pub fn is_available(self) -> bool {
        match self {
            Self::Wgpu => cfg!(feature = "wgpu"),
            Self::Cpu => cfg!(feature = "cpu"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Hip => cfg!(feature = "hip"),
        }
    }

    /// Compiled-in runtimes in preference order (GPU runtimes first).
    pub fn available() -> Vec<Self> {
        Self::ALL.into_iter().filter(|t| t.is_available()).collect()
    }

    /// `DSP_KITCHEN_RUNTIME` when set, else the first compiled-in runtime.
    pub fn from_env() -> Result<Self, ComputeError> {
        match std::env::var(RUNTIME_ENV) {
            Ok(name) if !name.trim().is_empty() => Self::parse(&name)?.checked(),
            _ => Self::available().first().copied().ok_or(ComputeError::NoneAvailable),
        }
    }

    /// `self` if it is compiled in, else [`ComputeError::Unavailable`].
    pub fn checked(self) -> Result<Self, ComputeError> {
        if self.is_available() { Ok(self) } else { Err(ComputeError::Unavailable(self, self.name())) }
    }

    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        match self.checked()? {
            #[cfg(feature = "wgpu")]
            Self::Wgpu => {
                use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
                Ok(task.run(WgpuRuntime::client(&WgpuDevice::default())))
            }
            #[cfg(feature = "cpu")]
            Self::Cpu => {
                use cubecl::cpu::{CpuDevice, CpuRuntime};
                Ok(task.run(CpuRuntime::client(&CpuDevice)))
            }
            #[cfg(feature = "cuda")]
            Self::Cuda => {
                use cubecl::cuda::{CudaDevice, CudaRuntime};
                Ok(task.run(CudaRuntime::client(&CudaDevice::default())))
            }
            #[cfg(feature = "hip")]
            Self::Hip => {
                use cubecl::hip::{AmdDevice, HipRuntime};
                Ok(task.run(HipRuntime::client(&AmdDevice::default())))
            }
            #[allow(unreachable_patterns)]
            other => Err(ComputeError::Unavailable(other, other.name())),
        }
    }
}

impl std::fmt::Display for ComputeTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for ComputeTarget {
    type Err = ComputeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeName;
    impl ComputeTask for RuntimeName {
        type Output = String;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> String {
            R::name(&client).to_string()
        }
    }

    #[test]
    fn parses_and_runs_on_available_targets() {
        assert_eq!(ComputeTarget::parse("ROCm").unwrap(), ComputeTarget::Hip);
        assert!(ComputeTarget::parse("tpu").is_err());
        for target in ComputeTarget::available() {
            assert!(!target.run(RuntimeName).unwrap().is_empty());
        }
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.run(RuntimeName), Err(ComputeError::Unavailable(..))));
        }
    }
}
