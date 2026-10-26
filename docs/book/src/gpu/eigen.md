# Case study: when the review was wrong

Whitening needs the eigenvectors and eigenvalues of the channel covariance
(`dsp_base::linalg::symmetric_eigen_batched`, a parallel cyclic Jacobi solver). A code review
predicted a problem, and checking it taught more than the fix did.

## The prediction

The solver stopped when `‖offdiag(A)‖_F ≤ 10⁻⁶ · ‖A‖_F`. In `f32`, rounding leaves off-diagonal
entries of about `√n · ε` relative to the matrix (≈ 1.2 · 10⁻⁶ at `n = 384`), so, the review
argued, the threshold could never be met and every solve would run all 30 sweeps.

## What the literature does

- cuSOLVER and rocSOLVER (`syevj`) stop on the off-diagonal norm with a tolerance that defaults to
  the machine precision of the type: not a fixed number.
- Demmel and Veselić ("Jacobi's method is more accurate than QR", SIAM J. Matrix Anal. Appl. 1992)
  test each pair: `(p, q)` has converged when `|a_pq| ≤ tol · √|a_pp · a_qq|`, and converged pairs
  are not rotated. With that test Jacobi computes the small eigenvalues of a positive definite
  matrix to high relative accuracy, where QR only guarantees accuracy relative to the largest.
  Whitening divides by `√λ`, so small eigenvalues are the ones that matter.

The solver now uses the pair test, with `tol = √n · ε` of the float type by default (the scaling
of LAPACK's one-sided Jacobi), and skips rotations of converged pairs (`linalg/kernels/eigen.rs`).
A requested tolerance below that floor is raised to it, since it could never be reached.

## What the measurement said

Counting sweeps on a 384 × 384 covariance-like matrix (eigenvalues over three decades):

```text
sweep 0  worst |a_pq|/√(a_pp a_qq) = 0.86
sweep 5                              0.022
sweep 6                              0.0041
sweep 7                              3.6e-5
sweep 8                              2.3e-6   ← converged (tol 2.3e-6)
```

Eight sweeps, with the quadratic convergence of Jacobi in the last three, and the old criterion
took about the same time (80 ms against 87 ms on an RTX 2070). **The prediction was wrong**: the
large eigenvalues dominate `‖A‖_F`, so `10⁻⁶` of it is well above the rounding floor. Accuracy on
this test is at the `f32` floor either way (worst relative residual ~2 · 10⁻⁴); the pair test's
advantage shows on matrices whose diagonal spans many magnitudes, which this one does not.

The change stays, for three reasons that do not depend on speed: the tolerance now follows the
element type instead of a constant tuned for `f32`; the criterion is the one with an accuracy
guarantee for small eigenvalues; and the shared-memory kernel's sweep loop, which looped on a
`done` flag, is now bounded by the sweep count.

**Lesson: a review's performance claim is a hypothesis.** Instrument the quantity it is about (here
the sweep count) before believing it, and report the result even when it is "no change".
