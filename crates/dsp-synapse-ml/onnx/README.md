# SpikeInterface / PyTorch Model Schemas (`onnx/` & `.safetensors`)

This directory houses reference `.onnx` model graphs and `.safetensors` weights exported from `spikeinterface` and PyTorch (`SpikeDeeptector`, `YASS`, `Dartsort`, `SingleChannelDenoiser`).

## Exporting Weights to `.safetensors`
```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```
Load weights dynamically at runtime via `dsp_synapse_ml::hub::SafetensorsMap::from_bytes` or `from_file`.
