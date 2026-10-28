//! Which compute runtime runs the DSP kernels.
//!
//! [`ComputeTarget`] is the one name for a runtime across the workspace: the CLI flag, Python
//! argument, app setting and the `DSPK_RUNTIME` environment variable all resolve to it.
//! Choosing a target needs no GPU libraries; running work on it is [`crate::compute`] (the
//! `compute` feature). Which targets can run depends on the enabled runtime features (`wgpu`,
//! `cpu`, `cuda`, `hip`).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Environment variable that overrides [`ComputeTarget::from_env`]'s default (`wgpu`, `cpu`,
/// `cuda`, `hip`).
pub const RUNTIME_ENV: &str = "DSPK_RUNTIME";

/// Environment variable that limits the runtimes tests run on ([`ComputeTarget::for_tests`]):
/// comma-separated names, e.g. `cpu` on a CI machine that has CUDA compiled in but no GPU.
pub const TEST_RUNTIMES_ENV: &str = "DSPK_TEST_RUNTIMES";

/// A CubeCL runtime and its default device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

    /// The runtimes tests run on: [`Self::available`], limited to the names in `DSPK_TEST_RUNTIMES`
    /// when it is set. Compiled in is not the same as usable: a build with CUDA can run on a
    /// machine without an NVIDIA driver.
    ///
    /// # Panics
    /// If `DSPK_TEST_RUNTIMES` names an unknown runtime (a typo would silently skip tests).
    pub fn for_tests() -> Vec<Self> {
        match std::env::var(TEST_RUNTIMES_ENV) {
            Ok(list) if !list.trim().is_empty() => {
                let wanted: Vec<Self> = list
                    .split(',')
                    .map(|name| Self::parse(name).unwrap_or_else(|e| panic!("{TEST_RUNTIMES_ENV}: {e}")))
                    .collect();
                Self::available().into_iter().filter(|t| wanted.contains(t)).collect()
            }
            _ => Self::available(),
        }
    }

    /// `DSPK_RUNTIME` when set, else the first compiled-in runtime.
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

    #[test]
    fn parses_names_and_reports_missing_runtimes() {
        assert_eq!(ComputeTarget::parse("ROCm").unwrap(), ComputeTarget::Hip);
        assert!(ComputeTarget::parse("tpu").is_err());
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.checked(), Err(ComputeError::Unavailable(..))));
        }
    }
}
