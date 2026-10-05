# Myomatrix & EMUsort: Introduction, Architecture & Citations

---

## 1. Overview & Scientific Motivation

**Myomatrix** arrays are microfabricated, flexible, high-density electrode arrays engineered specifically to record single-motor-unit activity from muscles in behaving animals and humans. Unlike rigid silicon probes designed for brain tissue, Myomatrix arrays flex with contracting muscle fibers, providing high-yield, stable electromyographic (HD-EMG) recordings across diverse motor behaviors.

Extracting single motor units from high-density EMG recordings presents biological challenges distinct from cortical electrophysiology:
1. **Prolonged Waveform Durations**: Motor Unit Action Potentials (MUAPs) typically span **$5\text{--}10\,\text{ms}$** (150–300 samples at 30 kHz), compared to the brief $1\text{--}2\,\text{ms}$ duration of cortical action potentials.
2. **Propagating Action Potentials**: MUAPs conduct along muscle fibers at velocities of **$2\text{--}6\,\text{m/s}$**, creating systematic inter-electrode conduction delays across adjacent array contacts.
3. **Complex Polyphasic Shapes**: Motor units consist of multiple muscle fibers innervated by a single motor neuron, yielding polyphasic waveforms that violate classic biphasic neuronal models.
4. **Severe Signal Overlap (Superposition)**: During sustained force generation, multiple motor units fire quasi-synchronously, leading to constructive and destructive waveform interference.

To solve this, **EMUsort**—developed by Samuel J. Sober's laboratory at Emory University and the SNEL group—extends the Kilosort4 template-matching framework to handle the spatial and temporal dynamics of muscle recordings.

---

## 2. Core Algorithmic Principles in `dsp-synapse`

```mermaid
flowchart LR
    A["Raw EMG Signal\n[Channels, Samples]"] --> B["Bandpass Filter\n(100–2000 Hz)"]
    B --> C["Common Average Ref\n(Or Differential)"]
    C --> D["150-Sample Universal\nMUAP Filter (wTEMP_EMG)"]
    D --> E["Conduction Latency\nAlignment (MFCV)"]
    E --> F["12-PC Spatiotemporal\nBasis (wPCA_EMG)"]
    F --> G["cBSS / Density\nClustering"]
    G --> H["Motor Unit Pulse\nTrains & PNR"]
```

* **150-Sample Universal MUAP Templates (`wTEMP_EMG`)**: Extended-length matched-filter kernels centered at the primary trough to capture triphasic and polyphasic muscle potentials.
* **Conduction Velocity & Latency Compensation**: Measures muscle fiber conduction velocity ($v \approx 2\text{--}6\,\text{m/s}$) across array rows to align propagating wavefronts before feature extraction.
* **12-Component Temporal Muscle Basis (`wPCA_EMG`)**: Preserves polyphasic details across prolonged MUAP windows ($T=150$ samples) without losing repolarization dynamics.
* **Motor Unit Pulse Trains & PNR Metric**: Quantifies motor unit isolation using the Pulse-to-Noise Ratio ($\text{PNR} \ge 20\,\text{dB}$) and Coefficient of Variation of Inter-Spike Intervals ($\text{CoV}_{\text{ISI}} \le 0.35$).

---

## 3. Official References & Academic Citations

### Primary Array Publication
* **Title**: *Myomatrix arrays for high-definition muscle recording*
* **Authors**: Bryce Chung, Megan M. Sellers, Claire M. Salinger, Bryce D. Schuman, Kyle F. Kovach, Michael A. Mendoza, and Samuel J. Sober
* **Journal**: *eLife*, Vol. 12, e88551 (2023)
* **DOI**: [10.7554/eLife.88551](https://doi.org/10.7554/eLife.88551)

### Sorting Software Reference (EMUsort)
* **Software**: *EMUsort: High-density intramuscular motor unit spike sorting*
* **Repository**: [https://github.com/snel-repo/EMUsort](https://github.com/snel-repo/EMUsort)
* **Associated Laboratory**: Sober Laboratory, Emory University ([https://sober-lab.org](https://sober-lab.org))

### BibTeX Entry
```bibtex
@article{chung2023myomatrix,
  title     = {{Myomatrix} arrays for high-definition muscle recording},
  author    = {Chung, Bryce and Sellers, Megan M and Salinger, Claire M and Schuman, Bryce D and Kovach, Kyle F and Mendoza, Michael A and Sober, Samuel J},
  journal   = {eLife},
  volume    = {12},
  pages     = {e88551},
  year      = {2023},
  publisher = {eLife Sciences Publications Limited},
  doi       = {10.7554/eLife.88551},
  url       = {https://doi.org/10.7554/eLife.88551}
}
```
