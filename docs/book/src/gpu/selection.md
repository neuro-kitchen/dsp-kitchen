# Case study: exact median on the device

Spike detection thresholds are set in units of each channel's noise, estimated robustly as
`median(|x|) / 0.6745` (`dsp_base::math::execute_channel_noise_std`). On the device this is a
per-row **selection**: the `k`-th smallest `|x|` of every channel (`core::reduce::row_abs_kth`).

## Before: bisection on values

The first kernel (one cube per channel) bracketed the answer between −1 and `max |x|` and halved
the bracket 64 times, counting at each step how many samples fall below the midpoint. Each halving
is a full pass over the row: 66 passes in all, and the answer was only exact to the bracket width
near zero.

## The idea: sort keys, not values

Libraries that select on GPUs (PyTorch's `kthvalue`, NVIDIA RAFT's `select_k` / AIR top-k) do
not compare values: they select on **bit patterns**. For a non-negative IEEE float, the bits read as
an unsigned integer of the same width sort exactly like the value (subnormals and zero included;
NaN above +∞). `|x|` is non-negative, so its key is simply `u32::reinterpret(|x|)` (`u64` for
`f64`).

Selection on integer keys is a **radix select**, digit by digit from the most significant:

1. Each pass counts, in a histogram of 2⁸ = 256 bins in shared memory (atomic adds), the keys
   that agree with the digits fixed so far, by their next 8-bit digit.
2. One thread walks the 256 bins to the one that holds rank `k`, fixes that digit, and subtracts
   the keys in lower bins from `k`.
3. After 4 passes (32-bit keys; 8 for `f64`) every digit is fixed: the key is the exact bit
   pattern of the answer.

PyTorch uses 2-bit digits (16 passes); RAFT uses 8 to 11. With 8 bits the histogram fits easily in
shared memory on every device we target, and 4 passes over a row replace 66.

## Results

Exact (tests require equality with the host's `select_nth`, in `f32` and `f64`), and faster:

| median \|x\|, 60 000 samples | RTX 2070 | Intel UHD 630 |
|---|---|---|
| 384 channels | 31.3 → **3.22** ms | 193 → **23.6** ms |
| 64 channels | 6.78 → **1.12** ms | 35.2 → **4.87** ms |
| 4 channels | 3.36 → **0.92** ms | 5.07 → **0.93** ms |

## What went wrong on the way

The first version returned 0 everywhere. A small probe kernel, testing each primitive alone,
found two CubeCL 0.11 behaviours (now in [Pitfalls](pitfalls.md)):

- `comptime!(K::size_bits())` inside a kernel gave 192 for `u32`, so the kernel ran 24 passes
  with shifts past the key width. The width now comes from the host as a comptime parameter.
- Methods called on an indexed shared atomic (`hist[i].fetch_add(1)`) act on a copy; the
  reference form `Atomic::fetch_add(&hist[i], 1)` updates shared memory.

The lesson is general: when a kernel is wrong in a way reading cannot explain, **probe the
primitives one at a time** with a tiny kernel rather than changing the algorithm.
