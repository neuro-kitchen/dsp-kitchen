# 08 · Detection

Threshold detection, template matching and waveform features on the synthetic recording (its
spikes are known, so detection can be scored). Waits for the synapse Python bindings
(`.tasks/27`): the detectors exist in Rust (`dspk-synapse`) and the CLI (`dspk detect`), not yet in
Python.
