use super::FilterError;

/// Relative impulse-response amplitude below which a filter is considered settled.
pub const DEFAULT_SETTLING_TOLERANCE: f64 = 1e-3;

/// One second-order section `b0 + b1 z⁻¹ + b2 z⁻² / (1 + a1 z⁻¹ + a2 z⁻²)`, `a[0] = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Section {
    pub b: [f64; 3],
    pub a: [f64; 3],
}

impl Section {
    /// Largest pole radius of the section.
    pub fn pole_radius(&self) -> f64 {
        let (a1, a2) = (self.a[1], self.a[2]);
        let disc = a1 * a1 - 4.0 * a2;
        if disc < 0.0 {
            a2.sqrt()
        } else {
            let sq = disc.sqrt();
            ((-a1 + sq) / 2.0).abs().max(((-a1 - sq) / 2.0).abs())
        }
    }

    /// Steady-state direct-form-II-transposed state for a unit step input (`lfilter_zi`).
    fn step_zi(&self) -> [f64; 2] {
        let [b0, b1, b2] = self.b;
        let [_, a1, a2] = self.a;
        let r0 = b1 - a1 * b0;
        let r1 = b2 - a2 * b0;
        let z0 = (r0 + r1) / (1.0 + a1 + a2);
        [z0, r1 - a2 * z0]
    }

    fn dc_gain(&self) -> f64 {
        self.b.iter().sum::<f64>() / self.a.iter().sum::<f64>()
    }

    /// The same transfer function as a trapezoidal state-variable filter (Cytomic SVF):
    /// `[a1, a2, a3, m0, m1, m2]` with `g = tan(ω/2)`-style normalization.
    ///
    /// Substituting `z⁻¹ = (1 − s)/(1 + s)` gives `N(s) / D(s)`; normalizing `D` to
    /// `g²(s'² + k·s' + 1)` with `s = g·s'` yields the SVF's `g`, `k`, and output mix `m0·v0 + m1·v1
    /// + m2·v2`. Valid for any stable section (`D` then has positive coefficients).
    pub fn svf(&self) -> [f64; 6] {
        let [b0, b1, b2] = self.b;
        let [_, a1, a2] = self.a;
        let (n0, n1, n2) = (b0 + b1 + b2, 2.0 * (b0 - b2), b0 - b1 + b2);
        let (d0, d1, d2) = (1.0 + a1 + a2, 2.0 * (1.0 - a2), 1.0 - a1 + a2);
        let g = (d0 / d2).sqrt();
        let k = d1 / (d2 * g);
        let m0 = n2 / d2;
        let m1 = n1 / (d2 * g) - m0 * k;
        let m2 = n0 / (d2 * g * g) - m0;
        let c1 = 1.0 / (1.0 + g * (g + k));
        let c2 = g * c1;
        [c1, c2, g * c2, m0, m1, m2]
    }
}

/// Cascade of second-order sections.
#[derive(Debug, Clone, PartialEq)]
pub struct Sos {
    pub sections: Vec<Section>,
}

impl Sos {
    pub fn new(sections: Vec<Section>) -> Self {
        Self { sections }
    }

    pub fn len(&self) -> usize {
        self.sections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Checks that sections are finite, normalized (`a0 = 1`) and stable.
    pub fn validate(&self) -> Result<(), FilterError> {
        let ok = !self.sections.is_empty()
            && self.sections.iter().all(|s| {
                s.b.iter().chain(s.a.iter()).all(|c| c.is_finite())
                    && (s.a[0] - 1.0).abs() < 1e-12
                    && s.pole_radius() < 1.0
            });
        if ok { Ok(()) } else { Err(FilterError::InvalidSections) }
    }

    /// Per-section steady-state initial conditions for a unit step (`sosfilt_zi`).
    /// Multiply by the first input sample to start a pass without a transient.
    pub fn zi(&self) -> Vec<[f64; 2]> {
        let mut scale = 1.0;
        self.sections
            .iter()
            .map(|s| {
                let [z0, z1] = s.step_zi();
                let zi = [scale * z0, scale * z1];
                scale *= s.dc_gain();
                zi
            })
            .collect()
    }

    /// Settling length in samples: the first `t` after which the impulse response tail holds at
    /// most `tol` of its total L1 mass (`Σ_{k≥t}|h[k]| ≤ tol · Σ|h[k]|`). For any input bounded by
    /// `A`, a chunk edge then perturbs samples `t` or more away by at most `tol · A · Σ|h|`.
    ///
    /// Measured on the designed cascade in f64; simulated up to twice the pole-radius bound
    /// ([`Self::pole_settling_bound`]), which is always longer.
    pub fn settling_samples(&self, tol: f64) -> usize {
        let horizon = 2 * self.pole_settling_bound(tol) + 64;
        let mut impulse = vec![0.0; horizon];
        impulse[0] = 1.0;
        let h = self.filter(&impulse, false);
        let total: f64 = h.iter().map(|v| v.abs()).sum();
        let mut tail = 0.0;
        for (t, v) in h.iter().enumerate().rev() {
            tail += v.abs();
            if tail > tol * total {
                return t + 1;
            }
        }
        0
    }

    /// Conservative settling bound from pole radii: `ceil(ln(tol) / ln(r))` per section (largest
    /// pole radius `r`), summed over the cascade.
    pub fn pole_settling_bound(&self, tol: f64) -> usize {
        self.sections
            .iter()
            .map(|s| {
                let r = s.pole_radius();
                if r < 1e-12 {
                    2
                } else {
                    (tol.ln() / r.ln()).ceil().max(2.0) as usize
                }
            })
            .sum()
    }

    /// DC gain of the whole cascade.
    pub fn dc_gain(&self) -> f64 {
        self.sections.iter().map(Section::dc_gain).product()
    }

    /// Kernel coefficient layout: [`Section::svf`] per section, then the cascade DC gain, f32.
    pub fn svf_coeffs_f32(&self) -> Vec<f32> {
        let mut c: Vec<f32> =
            self.sections.iter().flat_map(|s| s.svf()).map(|c| c as f32).collect();
        c.push(self.dc_gain() as f32);
        c
    }

    /// Host f64 reference of a forward pass (`sosfilt`) starting from `zi · x[0]` when
    /// `steady_state` is set, else from rest.
    pub fn filter(&self, x: &[f64], steady_state: bool) -> Vec<f64> {
        let x0 = x.first().copied().unwrap_or(0.0);
        let mut state: Vec<[f64; 2]> = if steady_state {
            self.zi().into_iter().map(|[a, b]| [a * x0, b * x0]).collect()
        } else {
            vec![[0.0; 2]; self.sections.len()]
        };
        x.iter()
            .map(|&xi| {
                let mut v = xi;
                for (s, z) in self.sections.iter().zip(state.iter_mut()) {
                    let y = s.b[0] * v + z[0];
                    z[0] = s.b[1] * v - s.a[1] * y + z[1];
                    z[1] = s.b[2] * v - s.a[2] * y;
                    v = y;
                }
                v
            })
            .collect()
    }

    /// Host f64 reference of zero-phase filtering (`sosfiltfilt` with `padtype="odd"`).
    /// `padlen` is clamped to `x.len() - 1`.
    pub fn filtfilt(&self, x: &[f64], padlen: usize) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let p = padlen.min(n - 1);
        let mut ext = Vec::with_capacity(n + 2 * p);
        ext.extend((1..=p).rev().map(|k| 2.0 * x[0] - x[k]));
        ext.extend_from_slice(x);
        ext.extend((1..=p).map(|k| 2.0 * x[n - 1] - x[n - 1 - k]));
        let mut y = self.filter(&ext, true);
        y.reverse();
        let mut y = self.filter(&y, true);
        y.reverse();
        y[p..p + n].to_vec()
    }
}
