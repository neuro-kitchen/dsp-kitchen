---
title: Validation
description: "How the numbers are checked, and the results."
sidebar:
  order: 0
---

A GPU kernel can be fast and wrong. These pages say what we compare dsp-kitchen against, and what
came out.

| Page | What it shows |
|---|---|
| [Against SciPy](/dsp-kitchen/validation/scipy/) | Each signal operation against SciPy and NumPy: time and agreement, per runtime |
| [What the tests check](/dsp-kitchen/validation/tests/) | The rules the test suite enforces |
| [Speed on real recordings](/dsp-kitchen/validation/throughput/) | Multiples of real time for whole files |
| [Sorters](/dsp-kitchen/plugins/synapse/validation/) | Each sorter against Kilosort4's saved results |

## What is and isn't claimed

- Signal operations are checked against reference implementations on every runtime we have.
- GPU claims rest on the hardware we have: an NVIDIA RTX 2070 Max-Q, an Intel UHD 630, and x86-64 CPUs.
  HIP (AMD) compiles but is **untested**.
- Sorters are compared with Kilosort4's output on one Neuropixels recording, which measures agreement,
  not accuracy against ground truth. Ground-truth accuracy isn't done yet.
