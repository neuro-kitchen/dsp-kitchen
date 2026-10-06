//! Pipelined reading of a recording for envelope building: the next block is read while the
//! current one is folded.

use std::ops::Range;

use dsp_core::{DspResult, MemoryOrder, RecordingSource};

use crate::envelope::fold::Block;

/// Reads `ranges` of every channel of `source` in its native order and hands each block to
/// `fold`, reading the next block while `fold` runs. Returns `Ok(false)` when `stop` returned
/// true before a read (blocks already read are still folded).
pub(crate) fn read_pipelined(
    source: &dyn RecordingSource,
    ranges: impl IntoIterator<Item = Range<u64>>,
    stop: impl Fn() -> bool + Sync,
    mut fold: impl FnMut(&Block) + Send,
) -> DspResult<bool> {
    let channels = source.info().channel_count();
    let read = |r: Range<u64>| -> DspResult<(Vec<f32>, MemoryOrder, Range<u64>)> {
        let mut data = vec![0.0f32; channels * (r.end - r.start) as usize];
        let order = source.read_native(r.clone(), &mut data)?;
        Ok((data, order, r))
    };
    let mut ranges = ranges.into_iter();
    let Some(first) = ranges.next() else { return Ok(true) };
    if stop() {
        return Ok(false);
    }
    let mut current = read(first)?;
    loop {
        let (data, order, range) = &current;
        let block = Block { data, order: *order, channels, samples: (range.end - range.start) as usize, first: range.start };
        let next = match ranges.next() {
            Some(r) if !stop() => {
                let (next, ()) = rayon::join(|| read(r), || fold(&block));
                next?
            }
            Some(_) => {
                fold(&block);
                return Ok(false);
            }
            None => {
                fold(&block);
                return Ok(true);
            }
        };
        current = next;
    }
}
