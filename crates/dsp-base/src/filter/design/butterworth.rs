//! Butterworth design: analog prototype → band transform → bilinear transform → SOS pairing,
//! following `scipy.signal.butter(N, Wn, btype, fs=fs, output="sos")` step by step.

use num_complex::Complex64 as C;
use std::f64::consts::PI;

use super::sos::{Section, Sos};
use super::{FilterBand, FilterError, check_cutoff, check_sample_rate};

/// Designs a Butterworth filter of `order` (per band edge) as second-order sections.
pub fn butterworth_sos(order: usize, band: FilterBand, sample_rate: f64) -> Result<Sos, FilterError> {
    check_sample_rate(sample_rate)?;
    if order == 0 {
        return Err(FilterError::InvalidOrder);
    }
    let edges = match band {
        FilterBand::Lowpass(f) | FilterBand::Highpass(f) => vec![f],
        FilterBand::Bandpass(lo, hi) | FilterBand::Bandstop(lo, hi) => vec![lo, hi],
    };
    for &f in &edges {
        check_cutoff(f, sample_rate)?;
    }
    if let [lo, hi] = edges[..]
        && lo >= hi
    {
        return Err(FilterError::InvalidBand { low_hz: lo, high_hz: hi });
    }

    // Design on scipy's normalized grid (fs = 2) with prewarped edges.
    let fs2 = 4.0;
    let warped: Vec<f64> = edges.iter().map(|f| fs2 * (PI * f / sample_rate).tan()).collect();

    let (z, p, k) = prototype(order);
    let (z, p, k) = match band {
        FilterBand::Lowpass(_) => lp2lp(&z, &p, k, warped[0]),
        FilterBand::Highpass(_) => lp2hp(&z, &p, k, warped[0]),
        FilterBand::Bandpass(..) => {
            lp2bp(&z, &p, k, (warped[0] * warped[1]).sqrt(), warped[1] - warped[0])
        }
        FilterBand::Bandstop(..) => {
            lp2bs(&z, &p, k, (warped[0] * warped[1]).sqrt(), warped[1] - warped[0])
        }
    };
    let (z, p, k) = bilinear(&z, &p, k, fs2);
    Ok(zpk2sos(z, p, k))
}

type Zpk = (Vec<C>, Vec<C>, f64);

/// Analog Butterworth prototype (`buttap`): no zeros, poles on the unit circle, gain 1.
fn prototype(n: usize) -> Zpk {
    let n_f = n as f64;
    let p = (0..n)
        .map(|i| {
            let m = -(n_f) + 1.0 + 2.0 * i as f64;
            -(C::i() * PI * m / (2.0 * n_f)).exp()
        })
        .collect();
    (Vec::new(), p, 1.0)
}

fn prod_neg(v: &[C]) -> C {
    v.iter().fold(C::new(1.0, 0.0), |acc, x| acc * -x)
}

fn lp2lp(z: &[C], p: &[C], k: f64, wo: f64) -> Zpk {
    let degree = (p.len() - z.len()) as i32;
    (z.iter().map(|x| x * wo).collect(), p.iter().map(|x| x * wo).collect(), k * wo.powi(degree))
}

fn lp2hp(z: &[C], p: &[C], k: f64, wo: f64) -> Zpk {
    let degree = p.len() - z.len();
    let mut z_hp: Vec<C> = z.iter().map(|x| wo / x).collect();
    let p_hp = p.iter().map(|x| wo / x).collect();
    z_hp.extend(std::iter::repeat_n(C::new(0.0, 0.0), degree));
    let k_hp = k * (prod_neg(z) / prod_neg(p)).re;
    (z_hp, p_hp, k_hp)
}

fn lp2bp(z: &[C], p: &[C], k: f64, wo: f64, bw: f64) -> Zpk {
    let degree = p.len() - z.len();
    let split = |v: &[C]| -> Vec<C> {
        let lp: Vec<C> = v.iter().map(|x| x * bw / 2.0).collect();
        let plus = lp.iter().map(|x| x + (x * x - wo * wo).sqrt());
        let minus = lp.iter().map(|x| x - (x * x - wo * wo).sqrt());
        plus.chain(minus).collect()
    };
    let mut z_bp = split(z);
    z_bp.extend(std::iter::repeat_n(C::new(0.0, 0.0), degree));
    (z_bp, split(p), k * bw.powi(degree as i32))
}

fn lp2bs(z: &[C], p: &[C], k: f64, wo: f64, bw: f64) -> Zpk {
    let degree = p.len() - z.len();
    let split = |v: &[C]| -> Vec<C> {
        let hp: Vec<C> = v.iter().map(|x| (bw / 2.0) / x).collect();
        let plus = hp.iter().map(|x| x + (x * x - wo * wo).sqrt());
        let minus = hp.iter().map(|x| x - (x * x - wo * wo).sqrt());
        plus.chain(minus).collect()
    };
    let mut z_bs = split(z);
    z_bs.extend(std::iter::repeat_n(C::new(0.0, wo), degree));
    z_bs.extend(std::iter::repeat_n(C::new(0.0, -wo), degree));
    let k_bs = k * (prod_neg(z) / prod_neg(p)).re;
    (z_bs, split(p), k_bs)
}

fn bilinear(z: &[C], p: &[C], k: f64, fs2: f64) -> Zpk {
    let degree = p.len() - z.len();
    let map = |x: &C| (fs2 + x) / (fs2 - x);
    let mut z_z: Vec<C> = z.iter().map(map).collect();
    z_z.extend(std::iter::repeat_n(C::new(-1.0, 0.0), degree));
    let num = z.iter().fold(C::new(1.0, 0.0), |acc, x| acc * (fs2 - x));
    let den = p.iter().fold(C::new(1.0, 0.0), |acc, x| acc * (fs2 - x));
    (z_z, p.iter().map(map).collect(), k * (num / den).re)
}

fn is_real(x: C) -> bool {
    x.im.abs() <= 100.0 * f64::EPSILON * x.norm().max(1.0)
}

/// `_cplxreal`: keep one member (positive imaginary part) of each conjugate pair, then reals.
fn cplxreal(v: Vec<C>) -> Vec<C> {
    let mut complex: Vec<C> = v.iter().copied().filter(|x| !is_real(*x) && x.im > 0.0).collect();
    let mut real: Vec<C> =
        v.iter().copied().filter(|x| is_real(*x)).map(|x| C::new(x.re, 0.0)).collect();
    complex.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.abs().total_cmp(&b.im.abs())));
    real.sort_by(|a, b| a.re.total_cmp(&b.re));
    complex.extend(real);
    complex
}

#[derive(Clone, Copy, PartialEq)]
enum Which {
    Any,
    Real,
    Complex,
}

fn nearest_idx(from: &[C], to: C, which: Which) -> usize {
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by(|&a, &b| (from[a] - to).norm().total_cmp(&(from[b] - to).norm()));
    match which {
        Which::Any => order[0],
        Which::Real => *order.iter().find(|&&i| is_real(from[i])).expect("real zero"),
        Which::Complex => *order.iter().find(|&&i| !is_real(from[i])).expect("complex zero"),
    }
}

/// Real polynomial coefficients (highest power first) from roots.
fn poly(roots: &[C]) -> Vec<f64> {
    let mut c = vec![C::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![C::new(0.0, 0.0); c.len() + 1];
        for (i, ci) in c.iter().enumerate() {
            next[i] += ci;
            next[i + 1] -= ci * r;
        }
        c = next;
    }
    c.into_iter().map(|x| x.re).collect()
}

/// `_single_zpksos`: coefficients right-aligned in the 3-tap section.
fn section(z: &[C], p: &[C]) -> Section {
    let (bz, ap) = (poly(z), poly(p));
    let mut b = [0.0; 3];
    let mut a = [0.0; 3];
    b[3 - bz.len()..].copy_from_slice(&bz);
    a[3 - ap.len()..].copy_from_slice(&ap);
    Section { b, a }
}

/// `zpk2sos` with `pairing="nearest"`: poles closest to the unit circle are paired first with
/// their nearest zeros and placed last; the gain goes into the first section.
fn zpk2sos(z: Vec<C>, p: Vec<C>, k: f64) -> Sos {
    let mut z = z;
    let mut p = p;
    let zero = C::new(0.0, 0.0);
    if p.len() < z.len() {
        p.resize(z.len(), zero);
    }
    if z.len() < p.len() {
        z.resize(p.len(), zero);
    }
    let n_sections = p.len().div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(zero);
        z.push(zero);
    }
    let mut p = cplxreal(p);
    let mut z = cplxreal(z);

    let mut sections = Vec::with_capacity(n_sections);
    for _ in 0..n_sections {
        let p1_idx = (0..p.len())
            .min_by(|&a, &b| (1.0 - p[a].norm()).abs().total_cmp(&(1.0 - p[b].norm()).abs()))
            .expect("pole");
        let p1 = p.remove(p1_idx);
        let reals_left = p.iter().filter(|x| is_real(**x)).count();

        if is_real(p1) && reals_left == 0 {
            let zi = nearest_idx(&z, p1, Which::Real);
            let z1 = z.remove(zi);
            sections.push(section(&[z1, zero], &[p1, zero]));
        } else if p.len() + 1 == z.len()
            && !is_real(p1)
            && reals_left == 1
            && z.iter().filter(|x| is_real(**x)).count() == 1
        {
            let zi = nearest_idx(&z, p1, Which::Complex);
            let z1 = z.remove(zi);
            sections.push(section(&[z1, z1.conj()], &[p1, p1.conj()]));
        } else {
            let p2 = if is_real(p1) {
                let p2_idx = (0..p.len())
                    .filter(|&i| is_real(p[i]))
                    .min_by(|&a, &b| (p[a].norm() - 1.0).abs().total_cmp(&(p[b].norm() - 1.0).abs()))
                    .expect("real pole");
                p.remove(p2_idx)
            } else {
                p1.conj()
            };
            if z.is_empty() {
                sections.push(section(&[], &[p1, p2]));
            } else {
                let zi = nearest_idx(&z, p1, Which::Any);
                let z1 = z.remove(zi);
                if !is_real(z1) {
                    sections.push(section(&[z1, z1.conj()], &[p1, p2]));
                } else if !z.is_empty() {
                    let zi2 = nearest_idx(&z, p1, Which::Real);
                    let z2 = z.remove(zi2);
                    sections.push(section(&[z1, z2], &[p1, p2]));
                } else {
                    sections.push(section(&[z1], &[p1, p2]));
                }
            }
        }
    }
    sections.reverse();
    for c in sections[0].b.iter_mut() {
        *c *= k;
    }
    Sos::new(sections)
}
