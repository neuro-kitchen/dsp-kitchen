# Case study: covariance

`dsp_base::linalg::{covariance, SecondMomentAccumulator}` compute `C = (X − μ)(X − μ)ᵀ / n` and
the uncentred `X Xᵀ / n`. The second moment, summed over windows, is the whitening statistic of
Kilosort4 and EMUsort; the same product of the clips with themselves gives their `wPCA` basis.

## Before: one thread per channel pair

The first kernel gave each pair `(i, j)` of the upper triangle a thread per 4 096-sample split and
summed `x_i[t] · x_j[t]`; a second kernel added the splits. It was easy to verify and very slow at
384 channels (706 ms per window on an RTX 2070): every sample is read once per pair it belongs to,
and neighbouring threads read different rows, so the loads do not combine (see
[Why matrix multiplication matters](matmul.md)).

## After: split products

The product is now computed by [`matmul`](matmul.md), keeping the one good idea of the old kernel,
the **splits**:

1. `split_rows_kernel` copies the samples into a **split-major** buffer
   `[splits, channels, 4096]` in one pass. The same pass subtracts the channel means (covariance),
   applies the window's column offset (so the input binds at its start: no alignment problem),
   and zero-pads the end of the last split (zero columns add nothing to `X Xᵀ`).
2. One **batched** product computes every split's `X_s X_sᵀ` (`[splits, C, C]`).
3. `sum_slices_kernel` adds the splits, divides by the true sample count and writes, or adds to,
   the result.

Why splits? **Precision**: no single sum adds more than 4 096 terms, so the rounding error of a
60 000-sample window stays close to that of a short one (`COVARIANCE_SPLIT_SAMPLES`). And
**parallelism**: with 4 channels the product is only 4 × 4, too small to keep a GPU busy; the
batch dimension gives it more independent work.

## The bug a test with an odd length found

The first version did not copy: it described the splits as a strided view of the original rows
(batch stride 4 096, row stride = the window length). With an **odd** window length the result
was wrong (`[1, 0] = −611` instead of `169`), while every test with even lengths passed.
cubek's batched kernels pick a vector width from the batch stride (4 096, divisible by 4) without
checking the row stride; with an odd row stride, vector loads straddle rows. Single products with
any stride are fine.

The fix is the copy above: batches are now the outermost stride, as cubek expects. Two rules came
out of it: **only give a library batched views whose batches are outermost**, and **test odd sizes**.
A matrix of clips (`[61, n_clips]`) has an odd row length half the time, so `wPCA` would have been
silently wrong.

## Results

At 384 channels × 60 000 samples, 706 → 18.9 ms (covariance) and 706 → 18.4 ms (second moment
over a window's interior) on the RTX 2070; 64 channels 7.4 → 2.2 ms. Tests compare against an
`f64` host covariance with odd and even lengths, a misaligned column range, and an `f64` run that
must agree to 1e-12.
