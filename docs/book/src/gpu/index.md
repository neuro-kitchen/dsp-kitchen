# GPU engineering

These chapters explain how the device code of dsp-kitchen is built and **why** it is built that
way, using the changes we made as worked examples. They are meant to be read by someone who wants
to write fast, correct GPU code for signal processing, not only by someone who wants to call it.

- [Why matrix multiplication matters](matmul.md): which DSP operations are matrix products, why a
  tiled product is so much faster than the obvious kernel, and how `dsp_base::linalg::matmul` is
  built on cubek.
- [Case study: covariance](covariance.md): from one thread per channel pair to split matrix
  products; the precision, alignment and layout problems we met.
- [Case study: HDBSCAN](hdbscan.md): a quadratic algorithm on 500 000 waveforms, a kernel that
  hung after a compiler upgrade, and exact shortcuts that make it finish in under a minute.
- [Case study: exact median](selection.md): radix select on the bits of |x|, exact and 4–10×
  faster than bisection.
- [Case study: when the review was wrong](eigen.md): the eigensolver's convergence test,
  checked against the literature and measured.
- [Case study: Kilosort4 detection, measured](detection.md): 477 → 123 ms per window, and why the
  planned fix was not the one that mattered.
- [Case study: Kilosort4 clustering and matching](clustering.md): graph clustering, merging trees
  and matching pursuit as matrix products and parallel kernels; 158 s → 5 s once the real cost (long
  serial loops in few threads, not host round trips) was measured.
- [Moving data to the device](data-movement.md): one crossing per window, in its stored integer
  form, scaled on the device.
- [Pitfalls](pitfalls.md): the traps that cost us a failing test or a frozen run.

## How we work

Every change follows the same loop. It is slower than "write it and try it", and it is the
reason the measurements in these chapters can be trusted.

1. **Read before running.** Find out from the code what a change touches and what can go wrong;
   run a test to check a conclusion, not to find one.
2. **Baseline first.** Measure the current code with the same benchmark that will measure the new
   code (`crates/dsp-base/tests/bench_linalg.rs`, `crates/dsp-synapse/tests/bench_hdbscan.rs`),
   on every device class: a discrete GPU, an integrated GPU and the CPU runtime. A rewrite can
   win on one and lose on another.
3. **A host reference in `f64`.** Device results are checked against a plain host computation in
   double precision, never against the previous kernel (which may be wrong too).
4. **Odd sizes and `f64`.** Tests use odd lengths (a layout bug hid behind even ones) and, where
   the runtime supports `f64`, check that the element type is kept end to end: an `f32` step
   anywhere shows up as a 1e-7 error where 1e-12 is expected.
5. **No silent constants.** Every tuning number is a named, documented `const` that says what it
   trades; extreme values come from the type (`F::max_value()`, `positive_infinity::<F>()`).
6. **Short runs.** A single test or benchmark gets at most five minutes; a run that needs more is
   a sign the problem size or the algorithm is wrong.

## Generic over the device and the element type

Algorithms take a CubeCL `Client` (the device was chosen by the caller) and are generic over the
float type `F: DspFloat`. The same source runs on CUDA, ROCm, Vulkan / Metal / DX12 (wgpu) and
the CPU (MLIR JIT); launch sizes come from the device's own properties
(`dsp_core::compute::LaunchGeometry`), and settings with no device-independent best value are
chosen by CubeCL's autotuner, which times the candidates once per device and problem class.
