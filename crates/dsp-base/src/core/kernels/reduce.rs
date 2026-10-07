use cubecl::prelude::*;
use dsp_core::compute::row_position;

/// Merges unit `b`'s `(count, mean, m2)` into unit `a`'s (Chan et al. parallel variance).
#[cube]
fn merge_moments<F: Float>(count: &mut Shared<[F]>, mean: &mut Shared<[F]>, m2: &mut Shared<[F]>, a: usize, b: usize) {
    let (na, nb) = (count[a], count[b]);
    let n = na + nb;
    if nb > F::new(0.0f32) {
        let delta = mean[b] - mean[a];
        let m2_b = m2[b];
        mean[a] += delta * nb / n;
        m2[a] += m2_b + delta * delta * na * nb / n;
        count[a] = n;
    }
}

/// Mean and population standard deviation (`/ cols`) of every row. `units` is the cube's x size
/// (a power of two).
#[cube(launch)]
pub fn row_mean_std_kernel<F: Float>(
    input: &[F],
    out_mean: &mut [F],
    out_std: &mut [F],
    rows: u32,
    cols: u32,
    #[comptime] units: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let base = (row * cols) as usize;

        // Welford over this unit's strided share
        let mut n = F::new(0.0f32);
        let mut mean = F::new(0.0f32);
        let mut m2 = F::new(0.0f32);
        let mut col = unit;
        while col < cols {
            let x = input[base + col as usize];
            n += F::new(1.0f32);
            let delta = x - mean;
            mean += delta / n;
            m2 += delta * (x - mean);
            col += units;
        }

        let mut count_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        let mut mean_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        let mut m2_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        count_s[unit as usize] = n;
        mean_s[unit as usize] = mean;
        m2_s[unit as usize] = m2;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(units / 2u32);
        while stride.read() > 0u32 {
            let s = stride.read();
            if unit < s {
                merge_moments::<F>(&mut count_s, &mut mean_s, &mut m2_s, unit as usize, (unit + s) as usize);
            }
            sync_cube();
            stride.store(s / 2u32);
        }

        if unit == 0u32 {
            out_mean[row as usize] = mean_s[0];
            let total = F::max(count_s[0], F::new(1.0f32));
            out_std[row as usize] = F::sqrt(m2_s[0] / total);
        }
    }
}
