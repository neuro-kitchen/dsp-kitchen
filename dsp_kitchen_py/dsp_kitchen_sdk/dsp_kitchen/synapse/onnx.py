"""
Burn-ONNX (`onnx-ir`) external spike-sorter bridge (`OnnxModelRunner`) supporting
Kilosort4, DARTsort, CEBRA, and Bombcell / UnitMatch `.onnx` graphs.
"""

from .._bindings import OnnxModelRunner

__all__ = [
    "OnnxModelRunner",
]
