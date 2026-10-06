//! Pipelined reading of a recording for envelope building: the next block is read while the
//! current one is folded.

use std::ops::Range;

use dsp_core::{DspResult, MemoryOrder, RecordingSource};

use std::cell::RefCell;

use crate::envelope::fold::Block;

thread_local! {
    /// The two block buffers of [`read_channel_blocks`], kept per thread so a viewer's frames
    /// reuse them.
    static BLOCKS: RefCell<[Vec<f32>; 2]> = const { RefCell::new([Vec::new(), Vec::new()]) };
}

/// Reads `ranges` of `channels` of `source` (channel-major) and hands each block to `fold`,
/// reading the next block while `fold` runs. The two block buffers are reused across calls on
/// one thread.
pub(crate) fn read_channel_blocks(
    source: &dyn RecordingSource,
    channels: &[usize],
    ranges: &[Range<u64>],
    mut fold: impl FnMut(&Block) + Send,
) -> DspResult<()> {
    let Some(first) = ranges.first() else { return Ok(()) };
    let rows = channels.len();
    let read = |r: &Range<u64>, buf: &mut Vec<f32>| -> DspResult<()> {
        buf.resize(rows * (r.end - r.start) as usize, 0.0);
        source.read(channels, r.clone(), buf)
    };
    BLOCKS.with_borrow_mut(|[current, next]| {
        read(first, current)?;
        for (i, range) in ranges.iter().enumerate() {
            let block = Block { data: current, order: MemoryOrder::ChannelMajor, channels: rows, samples: (range.end - range.start) as usize, first: range.start };
            match ranges.get(i + 1) {
                Some(r) => {
                    let (read_next, ()) = rayon::join(|| read(r, next), || fold(&block));
                    read_next?;
                }
                None => fold(&block),
            }
            std::mem::swap(current, next);
        }
        Ok(())
    })
}

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
