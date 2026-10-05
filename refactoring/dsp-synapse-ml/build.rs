//! Build script for `dsp-synapse-ml`.
//!
//! Monitors the `onnx/` schema directory and emits compile-time metadata for
//! pre-exported SpikeInterface / PyTorch `.onnx` models.

use std::path::Path;

fn main() {
    let onnx_dir = Path::new("onnx");
    println!("cargo:rerun-if-changed=onnx");
    if onnx_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(onnx_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}
