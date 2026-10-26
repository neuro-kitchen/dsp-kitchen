# Pitfalls

Traps that each cost us a failing test or a frozen run, with the rule we keep.

## WGSL has no infinity literal

`F::new(f32::INFINITY)` compiles to `Inf` in WGSL, which the shader compiler rejects; CUDA and the
CPU accept it, so only a wgpu run catches it. Build ±∞ from its IEEE-754 bits:
`dsp_core::compute::{positive_infinity, negative_infinity}` (`f32::reinterpret(bits)`, cast to `F`).

## Buffer bindings must be aligned

A buffer bound at an offset must start at a multiple of the device's offset alignment
(`min_storage_buffer_offset_alignment` on wgpu: 32 bytes on an RTX 2070, 256 on some GPUs; read it
from `client.properties().memory.alignment`). A column range of a channel-major window starts at
an arbitrary sample. Kernels we write take the element offset as a parameter and bind at the
buffer's start; library calls get an aligned view (`MatrixView::aligned`).

## 32-bit indices

Kernels index with `u32`: WGSL has no 64-bit integers and one kernel source serves every runtime.
A buffer holds at most `dsp_core::compute::MAX_DEVICE_ELEMENTS` (2³² − 1) values; past that an
index wraps silently (on CUDA, an out-of-range write corrupts memory without an error). Sizes a
user chooses are checked where they are chosen (`device_elements` in the sorter's schedule and
the detector), and every device allocation checks again as a backstop.

## Batched views for cubek

Give cubek batches as the outermost stride. A view whose batches interleave inside its rows is
computed wrongly when the row stride is odd ([covariance](covariance.md)).

## Loops on flags

A kernel loop whose exit depends on a flag set inside a branch can turn into an endless loop after
a compiler change, and an endless kernel freezes the program. Bound every loop by a counter or a
compile-time constant ([HDBSCAN](hdbscan.md)).

## Tensor cores change the precision

Tensor-core matrix routines round `f32` to `tf32`. `dsp_base::linalg::matmul` only uses routines
that compute in the requested type and checks it after every launch.

## The first call is slow

Autotuning compiles and times every candidate the first time a device sees a problem class:
seconds per class, not saved between runs unless CubeCL's cache is enabled. Benchmarks warm up
before timing; `CUBECL_DEBUG_LOG=<file>` shows what the tuner decided.

## A failing test is not yet a bug

A filter test failed for weeks as "the IIR from-rest bug": a 0.5 Hz high-pass started at rest
reported `max error 0.0702`, against a limit of `2e-5 · scale`. Taken apart (a host replay of the
kernel's arithmetic in `f32` and `f64`, then the device beside it):

- **Read the units.** 0.0702 was an *absolute* error on a signal whose peak is ~650: 1.08·10⁻⁴ of
  the peak, not 7 %. The tests now report relative errors.
- **Separate the kernel from the design.** The kernel matched its own arithmetic to ~6·10⁻⁶. The
  gap to scipy came from rounding the coefficients to `f32`: poles this close to `z = 1` move a
  little, and a start at rest feeds the signal's −500 DC level in as a step whose slow response
  exposes it. The `f64` state-variable filter with the same rounded coefficients is just as far from
  scipy. The tight limit had been measured with steady-state starts, which never excite that
  transient; from-rest at very low cutoffs now has its own, named, measured limit.
- **Regenerate fixtures when their generator changes.** The scipy reference script had learned to
  write `forward_rest` outputs, but the fixture file was never regenerated, so two more tests failed
  on a missing key (`uv run crates/dsp-base/tests/scipy_reference.py` declares its own numpy/scipy
  and leaves the project untouched).
- **An undersized buffer fails silently on a GPU.** While probing, a too-small scratch buffer made a
  forward-backward filter return zeros: writes past a buffer's end are dropped, reads return zero.
  `DeviceFilter::apply` now checks its `state` and `scratch` sizes.

## CubeCL 0.11 surprises inside kernels

- `comptime!(T::size_bits())` does not give the element width (192 for `u32`): pass widths from
  the host as `#[comptime]` parameters.
- Call atomics through a reference, `Atomic::fetch_add(&shared[i], v)`; `shared[i].fetch_add(v)`
  works on a copy.
- A mutable variable initialised from a constant expression (`let mut b = BINS - 1;`) becomes a
  comptime value and cannot take a runtime assignment later: initialise it from a runtime value or
  a literal.

## CubeCL 0.11 changes (from 0.10)

- No runtime generic: a `Client` is one type, chosen when made (`ComputeTarget::client()`).
- Kernel buffers are slices (`&[F]`, `BufferArg::from_raw_parts`); `Array::<F>::new(n)` is a local
  array; shared memory is `Shared::<[F]>::new_slice(n)`.
- A float scalar argument needs `F: Float + CubeElement + LaunchArg`.
- Annotated mutable locals (`let mut x: u32 = 0u32;`) do not expand in kernels.
- `LocalTuner::init(&id, …)` takes the device id.
- With `default-features = false`, enable cubecl's `std` feature.
