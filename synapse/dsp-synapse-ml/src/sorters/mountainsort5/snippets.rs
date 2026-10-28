//! MountainSort 5 snippets (`core/extract_snippets.py`): `T1` samples before each event to `T2`
//! after, on the channels within `mask_radius` µm of the event's channel (all channels when
//! `None`); upstream stores them dense (`[L, T, M]`, zeros off the mask), here only the masked
//! channels are stored (`[L, T, max_neighbours]`): the same numbers in less memory. The dense
//! form is rebuilt on the device a batch at a time for PCA ([`SnippetRows`], flattened `s · M + m`
//! as upstream's `reshape((L, T * M))`).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::RowSource;
use dsp_core::compute::LaunchGeometry;
use dsp_synapse::features::ChannelNeighbourhoods;

use super::kernels::{gather_snippets_kernel, scatter_snippets_kernel, NO_CHANNEL};

/// Masked snippets of many events (module docs).
#[derive(Debug, Clone)]
pub struct MaskedSnippets {
    /// `T1`.
    pub n_before: usize,
    /// `T1 + T2`.
    pub width: usize,
    pub channels: usize,
    pub mask: ChannelNeighbourhoods,
    pub event_channels: Vec<u32>,
    /// `[events, width, max_neighbours]`, slot `k` the `k`-th mask channel (0 in padding).
    pub values: Vec<f32>,
}

impl MaskedSnippets {
    /// No events yet. `positions`: `(x, y)` µm per channel.
    pub fn new(positions: &[[f32; 2]], mask_radius_um: Option<f32>, n_before: usize, n_after: usize) -> Self {
        Self {
            n_before,
            width: n_before + n_after,
            channels: positions.len(),
            mask: ChannelNeighbourhoods::within_radius(positions, mask_radius_um.unwrap_or(f32::INFINITY)),
            event_channels: Vec::new(),
            values: Vec::new(),
        }
    }

    /// No events yet, with explicit channel rows: an event's "channel" is the index of its row
    /// (e.g. one row per unit). Rows list channels in ascending order.
    pub fn with_rows(rows: &[Vec<u32>], channels: usize, n_before: usize, n_after: usize) -> Self {
        let max_neighbours = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut table = vec![NO_CHANNEL; rows.len().max(1) * max_neighbours];
        for (r, row) in rows.iter().enumerate() {
            table[r * max_neighbours..r * max_neighbours + row.len()].copy_from_slice(row);
        }
        Self {
            n_before,
            width: n_before + n_after,
            channels,
            mask: ChannelNeighbourhoods { table, max_neighbours },
            event_channels: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.event_channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.event_channels.is_empty()
    }

    fn row_len(&self) -> usize {
        self.width * self.mask.max_neighbours
    }

    /// Appends the snippets of events at local samples `samples` on `channels` of the `[channels,
    /// n_samples]` device trace.
    ///
    /// # Panics
    ///
    /// If a window leaves the trace, or the arrays differ in length.
    pub fn extract(&mut self, client: &Client, trace: &Handle, n_samples: usize, samples: &[u32], channels: &[u32]) {
        if samples.is_empty() {
            return;
        }
        let out = self.gather(client, trace, n_samples, samples, channels);
        self.values.extend(buffer::download::<f32>(client, out));
        self.event_channels.extend_from_slice(channels);
    }

    /// The masked snippets of events at local samples `samples` on `channels` (rows of the mask
    /// table), left on the device: `[events, width, max_neighbours]` `f32`. Nothing is stored.
    ///
    /// # Panics
    ///
    /// If a window leaves the trace, the arrays differ in length, or there are no events.
    pub fn gather(&self, client: &Client, trace: &Handle, n_samples: usize, samples: &[u32], channels: &[u32]) -> Handle {
        assert_eq!(samples.len(), channels.len(), "one channel per event");
        let n = samples.len();
        assert!(n > 0, "no events to gather");
        assert!(
            samples.iter().all(|&t| t as usize >= self.n_before && t as usize + self.width - self.n_before <= n_samples),
            "snippet windows must lie inside the trace"
        );
        let total = n * self.row_len();
        let out = buffer::empty::<f32>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            gather_snippets_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), self.channels * n_samples),
                BufferArg::from_raw_parts(buffer::upload(client, samples), n),
                BufferArg::from_raw_parts(buffer::upload(client, channels), n),
                BufferArg::from_raw_parts(buffer::upload(client, &self.mask.table), self.mask.table.len()),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                n_samples as u32,
                self.width as u32,
                self.mask.max_neighbours as u32,
                self.n_before as u32,
            );
        }
        out
    }

    /// Event `i`'s masked snippet, `[width, max_neighbours]`.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.row_len()..(i + 1) * self.row_len()]
    }

    /// Event `i`'s snippet in upstream's dense form, `[width, channels]` (zeros off the mask).
    pub fn dense(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.width * self.channels];
        let (m, row, ch) = (self.mask.max_neighbours, self.row(i), self.event_channels[i] as usize);
        for (k, nb) in self.mask.of(ch).enumerate() {
            for s in 0..self.width {
                out[s * self.channels + nb] = row[s * m + k];
            }
        }
        out
    }

    /// Upstream `np.roll(snippet, shift, axis=time)` of event `i`: sample `s` takes sample
    /// `s − shift` (circularly).
    pub fn roll(&mut self, i: usize, shift: i32) {
        let (w, m) = (self.width, self.mask.max_neighbours);
        let shift = shift.rem_euclid(w as i32) as usize;
        if shift == 0 {
            return;
        }
        let row = &mut self.values[i * w * m..(i + 1) * w * m];
        row.rotate_right(shift * m);
    }

    /// The dense rows of `subset` (all events when `None`) for [`RowSource`] consumers. The masked
    /// values are uploaded once here; each batch is then built on the device.
    pub fn rows<'a>(&'a self, client: &Client, subset: Option<&'a [usize]>) -> SnippetRows<'a> {
        let upload = |v: &[f32]| buffer::upload(client, if v.is_empty() { &[0.0f32][..] } else { v });
        SnippetRows {
            snippets: self,
            subset,
            table: buffer::upload(client, &self.mask.table),
            compact: upload(&self.values),
            compact_len: self.values.len().max(1),
        }
    }
}

/// Dense snippet rows, `width · channels` features each (module docs).
pub struct SnippetRows<'a> {
    snippets: &'a MaskedSnippets,
    subset: Option<&'a [usize]>,
    table: Handle,
    /// Every masked value, on the device.
    compact: Handle,
    compact_len: usize,
}

impl RowSource for SnippetRows<'_> {
    fn rows(&self) -> usize {
        self.subset.map_or(self.snippets.len(), <[usize]>::len)
    }

    fn dim(&self) -> usize {
        self.snippets.width * self.snippets.channels
    }

    fn batch(&self, client: &Client, range: std::ops::Range<usize>) -> Handle {
        let s = self.snippets;
        let b = range.len();
        let rows: Vec<u32> = match self.subset {
            None => range.map(|i| i as u32).collect(),
            Some(sub) => sub[range].iter().map(|&i| i as u32).collect(),
        };
        let row_channels: Vec<u32> = rows.iter().map(|&i| s.event_channels[i as usize]).collect();
        let dense = buffer::zeros::<f32>(client, b * self.dim());
        let total = b * s.row_len();
        if total > 0 {
            let geom = LaunchGeometry::elementwise(client, total);
            // SAFETY: every array is passed with the length it was created with
            unsafe {
                scatter_snippets_kernel::launch::<f32>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(self.compact.clone(), self.compact_len),
                    BufferArg::from_raw_parts(buffer::upload(client, &rows), b),
                    BufferArg::from_raw_parts(buffer::upload(client, &row_channels), b),
                    BufferArg::from_raw_parts(self.table.clone(), s.mask.table.len()),
                    BufferArg::from_raw_parts(dense.clone(), b * self.dim()),
                    total as u32,
                    s.width as u32,
                    s.mask.max_neighbours as u32,
                    s.channels as u32,
                );
            }
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    #[test]
    fn masked_snippets_match_upstream_dense_ones() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (m, n) = (5usize, 400usize);
        let pos: Vec<[f32; 2]> = (0..m).map(|c| [0.0, 20.0 * c as f32]).collect();
        let x: Vec<f32> = (0..m * n).map(|i| ((i * 7919) % 1000) as f32 / 100.0 - 5.0).collect();
        let trace = buffer::upload(&client, &x);
        let (t1, t2) = (6usize, 9usize);
        let mut sn = MaskedSnippets::new(&pos, Some(25.0), t1, t2);
        let (times, chans) = (vec![10u32, 200, 390], vec![0u32, 2, 4]);
        sn.extract(&client, &trace, n, &times, &chans);
        assert_eq!(sn.len(), 3);
        for j in 0..3 {
            let d = sn.dense(j);
            for s in 0..t1 + t2 {
                for c in 0..m {
                    let near = (pos[c][1] - pos[chans[j] as usize][1]).abs() <= 25.0;
                    let want = if near { x[c * n + times[j] as usize - t1 + s] } else { 0.0 };
                    assert_eq!(d[s * m + c], want, "event {j} sample {s} channel {c}");
                }
            }
        }
        // Device dense rows equal the host dense form, also for a subset
        let subset = [2usize, 0];
        let rows = sn.rows(&client, Some(&subset));
        assert_eq!((rows.rows(), rows.dim()), (2, (t1 + t2) * m));
        let got = buffer::download::<f32>(&client, rows.batch(&client, 0..2));
        assert_eq!(&got[..rows.dim()], &sn.dense(2)[..]);
        assert_eq!(&got[rows.dim()..], &sn.dense(0)[..]);
        // np.roll by +2: sample s takes s − 2
        let before = sn.dense(1);
        sn.roll(1, 2);
        let after = sn.dense(1);
        let w = t1 + t2;
        for s in 0..w {
            assert_eq!(after[s * m..(s + 1) * m], before[((s + w - 2) % w) * m..((s + w - 2) % w + 1) * m]);
        }
    }
}
