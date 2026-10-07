//! Matrix products on the device, `out[b] = lhs[b] · rhs[b]`, through
//! [cubek-matmul](https://github.com/tracel-ai/cubek) (CubeCL 0.11 has no matmul of its own).
//!
//! Operands are [`MatrixView`]s: any batch / row / column strides and a start offset over a device
//! buffer, so transposes, column ranges and sample splits are views, never copies.
//!
//! **Precision.** Only routines that keep every operand in `F` take part: the unit routines
//! (`SimpleUnit`, `DoubleUnit`; every runtime), `Gemm` and the CPU register-blocked `CpuGemm`. The
//! tensor-core routines are left out because they round `f32` operands to `tf32` (~3 significant
//! digits), too coarse for covariances and whitening. A routine that changes the register types is
//! rejected even if it is in the list.
//!
//! **Selection.** cubek's own `Strategy::Auto` does not tune (it tries tensor cores, then
//! `SimpleUnit`), so on the CPU runtime it would never reach `CpuGemm`. CubeCL's autotuner measures
//! the candidates once per device and problem class instead; candidates a device or layout cannot
//! run report `Unavailable` and drop out. The CPU routines are offered only where lanes run on
//! their own (plane width 1, the CPU runtime), as burn does. One more candidate is ours:
//! [`direct_matmul_kernel`], one unit per output element, which wins on products with little to
//! reuse (a few rows, a short inner dimension), where a tiled routine's fixed per-call cost and
//! unused tile space dominate. The tuner keeps whichever is fastest: no size threshold.
//!
//! **Reproducibility.** Routines sum in different orders, so the tuner's pick can change the last
//! bits; while [`dsp_core::compute::pin_tuned_choices`] is held, one fixed routine runs instead.

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::{Shape, Strides};
use cubecl::tune::{local_tuner, LocalTuner, Tunable, TunableSet};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::launch::launch_ref;
use cubek_matmul::multi_level::routines::batch::double_unit::DoubleUnitSelectionArgs;
use cubek_matmul::multi_level::routines::batch::simple_unit::SimpleUnitSelectionArgs;
use cubek_matmul::multi_level::routines::TileSizeSelection;
use cubek_matmul::multi_level::Strategy as LevelStrategy;
use cubek_matmul::routine::BlueprintStrategy;
use cubek_matmul::strategy::Strategy;
use cubek_matmul::tiled::Strategy as TiledStrategy;
use cubek_std::InputBinding;
use dsp_core::compute::tune::{size_class, tune_id};
use dsp_core::compute::LaunchGeometry;

use super::kernels::{direct_matmul_kernel, gather_view_kernel};
use crate::core::{buffer, DspFloat};

/// A `[batches, rows, cols]` matrix stack over a device buffer of `F`: element `(b, r, c)` is at
/// `offset + b · batch_stride + r · row_stride + c · col_stride` (in elements).
#[derive(Debug, Clone)]
pub struct MatrixView {
    handle: Handle,
    /// Elements the buffer holds (bounds checks).
    len: usize,
    offset: usize,
    shape: [usize; 3],
    strides: [usize; 3],
}

impl MatrixView {
    /// A row-major `[rows, cols]` matrix filling the start of `handle` (`len` elements).
    pub fn row_major(handle: &Handle, len: usize, rows: usize, cols: usize) -> Self {
        let view = Self { handle: handle.clone(), len, offset: 0, shape: [1, rows, cols], strides: [rows * cols, cols, 1] };
        view.check();
        view
    }

    /// `batches` contiguous row-major `[rows, cols]` matrices, one after the other, filling the
    /// start of `handle` (`len` elements).
    pub fn batched_row_major(handle: &Handle, len: usize, batches: usize, rows: usize, cols: usize) -> Self {
        let view =
            Self { handle: handle.clone(), len, offset: 0, shape: [batches, rows, cols], strides: [rows * cols, cols, 1] };
        view.check();
        view
    }

    pub fn batches(&self) -> usize {
        self.shape[0]
    }

    pub fn rows(&self) -> usize {
        self.shape[1]
    }

    pub fn cols(&self) -> usize {
        self.shape[2]
    }

    /// The same matrices transposed (no copy).
    pub fn transposed(&self) -> Self {
        let mut view = self.clone();
        view.shape.swap(1, 2);
        view.strides.swap(1, 2);
        view
    }

    /// Columns `cols` of every matrix (no copy).
    pub fn columns(&self, cols: std::ops::Range<usize>) -> Self {
        assert!(cols.start <= cols.end && cols.end <= self.cols(), "columns {cols:?} outside {} columns", self.cols());
        let mut view = self.clone();
        view.offset += cols.start * self.strides[2];
        view.shape[2] = cols.len();
        view
    }

    /// Whether the view can bind at its offset: devices require buffer bindings to start at a
    /// multiple of their offset alignment (`min_storage_buffer_offset_alignment` on wgpu).
    pub fn is_aligned<F: DspFloat>(&self, client: &Client) -> bool {
        let alignment = client.properties().memory.alignment.max(1);
        (self.offset * size_of::<F>()) as u64 % alignment == 0
    }

    /// This view if it can bind as it is, else a contiguous copy of it (one gather pass). Build
    /// transposes and splits of one operand from the aligned view to copy it once.
    pub fn aligned<F: DspFloat>(&self, client: &Client) -> Self {
        if self.is_aligned::<F>(client) {
            return self.clone();
        }
        let [batches, rows, cols] = self.shape;
        let total = batches * rows * cols;
        let out = buffer::empty::<F>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: `check` keeps every element the view reaches inside `len`; `out` holds `total`
        unsafe {
            gather_view_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(self.handle.clone(), self.len),
                BufferArg::from_raw_parts(out.clone(), total),
                self.offset as u32,
                rows as u32,
                cols as u32,
                self.strides[0] as u32,
                self.strides[1] as u32,
                self.strides[2] as u32,
                total as u32,
            );
        }
        Self { handle: out, len: total, offset: 0, shape: self.shape, strides: [rows * cols, cols, 1] }
    }

    /// The highest element index the view reaches must lie in the buffer, and every index must fit
    /// the 32-bit device indexing.
    fn check(&self) {
        let last = self.offset
            + self.shape.iter().zip(&self.strides).map(|(&n, &s)| n.saturating_sub(1) * s).sum::<usize>();
        let empty = self.shape.contains(&0);
        assert!(empty || last < self.len, "matrix view reaches element {last} of a {}-element buffer", self.len);
        assert!(u32::try_from(self.len).is_ok(), "{} elements exceed 32-bit device indexing", self.len);
    }

    fn binding<F: DspFloat>(&self) -> InputBinding {
        self.check();
        let handle = self.handle.clone().offset_start((self.offset * size_of::<F>()) as u64);
        // SAFETY: `check` keeps every element the shape and strides reach inside the buffer
        let tensor = unsafe { TensorBinding::from_raw_parts(handle, Strides::new(&self.strides), Shape::new(self.shape)) };
        InputBinding::new(tensor, F::elem_type_native())
    }
}

/// One product's operands (cloned per autotune candidate).
#[derive(Clone)]
struct MatmulInputs {
    client: Client,
    lhs: MatrixView,
    rhs: MatrixView,
    out: Handle,
}

impl MatmulInputs {
    fn run<F: DspFloat>(&self, strategy: &Strategy) -> Result<(), String> {
        let [b, m, _] = self.lhs.shape;
        let n = self.rhs.cols();
        // SAFETY: `matmul` checked that `out` holds `b · m · n` elements of `F`
        let out = unsafe { TensorBinding::from_raw_parts(self.out.clone(), Strides::new(&[m * n, n, 1]), Shape::new([b, m, n])) };
        let elem = F::elem_type_native();
        let mut dtypes = MatmulElems::from_single_dtype(elem);
        launch_ref(strategy, &self.client, self.lhs.binding::<F>(), self.rhs.binding::<F>(), out, &mut dtypes)
            .map_err(|e| format!("{e:?}"))?;
        // A routine that rounded the operands below `F` is not exact: never select it
        let registers = [dtypes.lhs_register, dtypes.rhs_register, dtypes.acc_register];
        if registers.iter().any(|&r| r != elem) {
            return Err(format!("{strategy} computes in {registers:?}, not {elem:?}"));
        }
        Ok(())
    }

    /// [`direct_matmul_kernel`] over the views (offsets applied in the kernel).
    fn run_direct<F: DspFloat>(&self) {
        let [b, m, k] = self.lhs.shape;
        let n = self.rhs.cols();
        let geom = LaunchGeometry::channels_samples(&self.client, b * m, n);
        let (l, r) = (&self.lhs, &self.rhs);
        // SAFETY: both views passed `check` (every element they reach lies in `len`); `matmul`
        // checked that `out` holds `b · m · n` elements
        unsafe {
            direct_matmul_kernel::launch::<F>(
                &self.client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(l.handle.clone(), l.len),
                BufferArg::from_raw_parts(r.handle.clone(), r.len),
                BufferArg::from_raw_parts(self.out.clone(), b * m * n),
                b as u32,
                m as u32,
                n as u32,
                k as u32,
                l.offset as u32,
                l.strides[0] as u32,
                l.strides[1] as u32,
                l.strides[2] as u32,
                r.offset as u32,
                r.strides[0] as u32,
                r.strides[1] as u32,
                r.strides[2] as u32,
            );
        }
    }
}

/// The full-precision cubek candidates; `cpu_lanes`: the device's lanes run on their own (plane
/// width 1, the CPU runtime).
fn exact_strategies(cpu_lanes: bool) -> Vec<Strategy> {
    let mut all = Vec::new();
    for tile_size in [TileSizeSelection::MaxTileSize, TileSizeSelection::MinTileSize] {
        all.push(LevelStrategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs { tile_size })).into());
        all.push(LevelStrategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs { tile_size })).into());
    }
    // Written for lanes that run on their own: timed on a GPU they only waste the tuning run
    if cpu_lanes {
        all.push(LevelStrategy::Gemm(BlueprintStrategy::Inferred(Default::default())).into());
        all.push(TiledStrategy::CpuGemm(BlueprintStrategy::Inferred(Default::default())).into());
    }
    all
}

/// The routine of every product while tuned choices are pinned
/// ([`dsp_core::compute::pin_tuned_choices`]): `SimpleUnit` with its largest tiles, exact and
/// available on every runtime; [`direct_matmul_kernel`] where a layout rules it out.
fn pinned_strategy() -> Strategy {
    LevelStrategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs { tile_size: TileSizeSelection::MaxTileSize })).into()
}

/// Whether a view's columns are its contiguous axis (row-major) or its rows are.
fn layout(view: &MatrixView) -> &'static str {
    if view.strides[2] == 1 { "row" } else { "col" }
}

/// `out[b] = lhs[b] · rhs[b]` for `lhs` `[batches, m, k]` and `rhs` `[batches, k, n]` views of `F`,
/// into the contiguous `[batches, m, n]` buffer `out` (overwritten), with the fastest exact routine
/// for this device and problem class. Operands whose offset the device cannot bind are copied
/// first ([`MatrixView::aligned`]).
///
/// # Panics
/// If the shapes do not chain, `out` is too small, or no exact routine can run the product.
pub fn matmul<F: DspFloat>(client: &Client, lhs: &MatrixView, rhs: &MatrixView, out: &Handle, out_len: usize) {
    assert_eq!(lhs.batches(), rhs.batches(), "batch counts differ");
    assert_eq!(lhs.cols(), rhs.rows(), "inner dimensions differ");
    let need = lhs.batches() * lhs.rows() * rhs.cols();
    assert!(need <= out_len, "output of {need} elements does not fit {out_len}");
    if need == 0 {
        return;
    }
    let (lhs, rhs) = (lhs.aligned::<F>(client), rhs.aligned::<F>(client));
    let inputs = MatmulInputs { client: client.clone(), lhs, rhs, out: out.clone() };
    // Routines sum in different orders: pinned runs use one fixed, exact routine
    if dsp_core::compute::tuned_choices_pinned() {
        if inputs.run::<F>(&pinned_strategy()).is_err() {
            inputs.run_direct::<F>();
        }
        return;
    }

    static TUNER: LocalTuner<String, String> = local_tuner!("matmul-exact");
    let id = tune_id(client);
    let cpu_lanes = LaunchGeometry::plane_lanes(client) == 1;
    let set = TUNER.init(&id, move || {
        let key = |i: &MatmulInputs| {
            format!(
                "{}-b{}-m{}-n{}-k{}-{}{}",
                F::type_name(),
                size_class(i.lhs.batches()),
                size_class(i.lhs.rows()),
                size_class(i.rhs.cols()),
                size_class(i.lhs.cols()),
                layout(&i.lhs),
                layout(&i.rhs)
            )
        };
        let set: TunableSet<String, MatmulInputs, ()> = TunableSet::new_cloning_inputs(key)
            .with(Tunable::new("direct", |i: MatmulInputs| {
                i.run_direct::<F>();
                Ok::<_, String>(())
            }));
        exact_strategies(cpu_lanes).into_iter().fold(set, |set, strategy| {
            set.with(Tunable::new(&strategy.to_string(), move |i: MatmulInputs| i.run::<F>(&strategy)))
        })
    });
    TUNER.execute(&id, client, set, inputs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    /// Host `f64` product of row-major `a` (`[m, k]`) and `b` (`[k, n]`).
    fn host(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        (0..m * n).map(|e| (0..k).map(|j| a[(e / n) * k + j] * b[j * n + e % n]).sum()).collect()
    }

    fn values(len: usize, seed: usize) -> Vec<f32> {
        (0..len).map(|i| (((i + seed) * 7919) % 211) as f32 / 21.1 - 5.0).collect()
    }

    fn products(client: &Client) {
        // Square, tall-skinny, a long inner dimension, odd sizes
        for (m, k, n) in [(16usize, 16usize, 16usize), (5, 3, 7), (3, 4_099, 3), (33, 65, 17)] {
            let (a, b) = (values(m * k, 1), values(k * n, 2));
            let (ha, hb) = (buffer::upload(client, &a), buffer::upload(client, &b));
            let out = buffer::empty::<f32>(client, m * n);
            matmul::<f32>(client, &MatrixView::row_major(&ha, a.len(), m, k), &MatrixView::row_major(&hb, b.len(), k, n), &out, m * n);
            let got = buffer::download::<f32>(client, out);
            let want = host(&a.iter().map(|v| *v as f64).collect::<Vec<_>>(), &b.iter().map(|v| *v as f64).collect::<Vec<_>>(), m, k, n);
            for (e, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() < 1e-4 * w.abs().max(k as f64), "{} {m}x{k}x{n} [{e}]: {g} vs {w}", client.name());
            }
        }
    }
    runtime_test!(test_matmul_matches_host, products);

    /// `X[:, cols] · X[:, cols]ᵀ` from views (offset columns, a transpose), against the host.
    fn views(client: &Client) {
        // Unaligned starts (gathered) and aligned ones bound as they are, even and odd row lengths
        for (row_len, cols) in [(1_000usize, 13..987usize), (1_001, 13..987), (1_001, 0..1_001), (1_001, 0..987)] {
            views_with(client, row_len, cols);
        }
    }

    fn views_with(client: &Client, row_len: usize, cols: std::ops::Range<usize>) {
        let rows = 6usize;
        let x = values(rows * row_len, 3);
        let hx = buffer::upload(client, &x);
        let view = MatrixView::row_major(&hx, x.len(), rows, row_len).columns(cols.clone());
        let host_gram: Vec<f64> =
            (0..rows * rows).map(|e| cols.clone().map(|t| x[(e / rows) * row_len + t] as f64 * x[(e % rows) * row_len + t] as f64).sum()).collect();

        let out = buffer::empty::<f32>(client, rows * rows);
        matmul::<f32>(client, &view, &view.transposed(), &out, rows * rows);
        let got = buffer::download::<f32>(client, out);
        for (g, w) in got.iter().zip(&host_gram) {
            assert!((*g as f64 - w).abs() < 1e-4 * w.abs().max(cols.len() as f64), "{} row_len {row_len} cols {cols:?} gram: {g} vs {w}", client.name());
        }
    }
    runtime_test!(test_matmul_views_match_host, views);

    /// Contiguous batches (`batched_row_major`, odd sizes) times their transposes, against the host.
    fn batches(client: &Client) {
        let (count, rows, cols) = (7usize, 5usize, 1_001usize);
        let x = values(count * rows * cols, 4);
        let hx = buffer::upload(client, &x);
        let view = MatrixView::batched_row_major(&hx, x.len(), count, rows, cols);
        let out = buffer::empty::<f32>(client, count * rows * rows);
        matmul::<f32>(client, &view, &view.transposed(), &out, count * rows * rows);
        let got = buffer::download::<f32>(client, out);
        for b in 0..count {
            let m = &x[b * rows * cols..(b + 1) * rows * cols];
            for e in 0..rows * rows {
                let want: f64 = (0..cols).map(|t| m[(e / rows) * cols + t] as f64 * m[(e % rows) * cols + t] as f64).sum();
                let g = got[b * rows * rows + e] as f64;
                assert!((g - want).abs() < 1e-4 * want.abs().max(cols as f64), "{} batch {b} [{e}]: {g} vs {want}", client.name());
            }
        }
    }
    runtime_test!(test_matmul_batches_match_host, batches);

    /// On runtimes with `f64`, the product is computed in `f64` (an `f32` step would leave ~1e-7
    /// relative error, far above the bound).
    fn keeps_f64(client: &Client) {
        if !client.properties().supports_type(f64::elem_type_native()) {
            return;
        }
        let (m, k, n) = (7usize, 513usize, 5usize);
        let a: Vec<f64> = (0..m * k).map(|i| ((i * 7919) % 1013) as f64 / 7.0 + 1e3).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i * 104_729) % 997) as f64 / 13.0).collect();
        let (ha, hb) = (buffer::upload(client, &a), buffer::upload(client, &b));
        let out = buffer::empty::<f64>(client, m * n);
        matmul::<f64>(client, &MatrixView::row_major(&ha, a.len(), m, k), &MatrixView::row_major(&hb, b.len(), k, n), &out, m * n);
        let got = buffer::download::<f64>(client, out);
        for (e, (g, w)) in got.iter().zip(host(&a, &b, m, k, n)).enumerate() {
            assert!((g - w).abs() < 1e-12 * w.abs(), "{} f64 [{e}]: {g} vs {w}", client.name());
        }
    }
    runtime_test!(test_matmul_keeps_f64, keeps_f64);
}
