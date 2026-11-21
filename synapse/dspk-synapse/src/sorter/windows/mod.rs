//! Windows of a recording on their way to the device ([`DeviceWindows`]), shared by every sorter's
//! passes: read on a background thread, uploaded once in their compact stored form, preprocessed
//! on the device and, for sorters that need it, aligned ([`WindowAligner`]).

use std::collections::HashMap;

use cubecl::server::Handle;
use dspk_core::{DspResult, HaloWindow, RecordingSource, SampleFormat, WindowLoader};
use dspk_ops::core::buffer;
use dspk_ops::pipeline::PipelineWorkspace;

/// Maximum bytes of exact `f32` preprocessed windows retained on device across passes.
const MAX_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// A step applied to every preprocessed window on the device before a sorter sees it (in
/// `dspk-sorter-kilosort`: EMUsort's per-channel delay alignment, Kilosort4's drift correction).
pub trait WindowAligner {
    /// The aligned `[channels, samples]` window for the device window `window` (`samples` per
    /// channel), which is `at` of the schedule.
    fn align(&mut self, window: &Handle, at: &HaloWindow, samples: usize) -> Handle;
}

/// Several steps, applied in order.
pub struct AlignerChain<'a>(pub Vec<Box<dyn WindowAligner + 'a>>);

impl WindowAligner for AlignerChain<'_> {
    fn align(&mut self, window: &Handle, at: &HaloWindow, samples: usize) -> Handle {
        let mut handle = window.clone();
        for step in &mut self.0 {
            handle = step.align(&handle, at, samples);
        }
        handle
    }
}

/// Windows of a recording on their way to the device: read on a background thread into a buffer
/// the upload then takes over (no copy on the calling thread), uploaded as the source's **stored**
/// values when it can provide them (integers: `int16` moves half the bytes of `f32`, and the gain /
/// offset are applied on the device, not on the host thread), else as scaled `f32`; preprocessed
/// by `workspace` and, with an aligner, aligned. Every pass of a run goes through it, so a window
/// crosses to the device once, in its compact form.
pub struct DeviceWindows<'a> {
    loader: WindowLoader<'a>,
    stored: Option<SampleFormat>,
    workspace: PipelineWorkspace<f32>,
    aligner: Option<Box<dyn WindowAligner + 'a>>,
    cache: HashMap<(usize, u64, u64), Handle>,
    cached_bytes: usize,
}

impl<'a> DeviceWindows<'a> {
    pub fn new(source: &'a dyn RecordingSource, mut workspace: PipelineWorkspace<f32>, aligner: Option<Box<dyn WindowAligner + 'a>>) -> Self {
        let info = source.info();
        // Stored reads only pay off for integer formats, and not every source implements them
        let probe = |format: SampleFormat| source.read_stored(&[0], 0..1, &mut vec![0u8; format.bytes()]).is_ok();
        let stored = (info.format != SampleFormat::F32 && info.samples > 0 && info.channel_count() > 0 && probe(info.format))
            .then_some(info.format);
        if stored.is_some() {
            let (gains, offsets): (Vec<f32>, Vec<f32>) = info.channels.iter().map(|c| (c.gain, c.offset)).unzip();
            workspace.set_stored_scaling(&gains, &offsets);
        }
        Self { loader: WindowLoader::new(source), stored, workspace, aligner, cache: HashMap::new(), cached_bytes: 0 }
    }

    /// Adds `step` after the current per-window steps, for the passes still to come (a correction
    /// estimated from an earlier pass: drift).
    pub fn push_aligner(&mut self, step: Box<dyn WindowAligner + 'a>) {
        self.cache.clear();
        self.cached_bytes = 0;
        self.aligner = Some(match self.aligner.take() {
            Some(first) => Box::new(AlignerChain(vec![first, step])),
            None => step,
        });
    }

    /// Calls `f(window, prepared)` for every window of `windows` in order (`prepared`: the
    /// `[channels, window.read_len()]` preprocessed window on the device, valid until the next
    /// call), until `f` returns `false`.
    pub fn for_each_while(&mut self, windows: &[HaloWindow], mut f: impl FnMut(&HaloWindow, &Handle) -> DspResult<bool>) -> DspResult<()> {
        let Self { loader, stored, workspace, aligner, cache, cached_bytes } = self;
        let key_of = |w: &HaloWindow| (w.index, w.read_global.start, w.read_global.end);
        let mut idx = 0usize;
        while idx < windows.len() {
            while idx < windows.len() {
                if let Some(hit) = cache.get(&key_of(&windows[idx])) {
                    if !f(&windows[idx], hit)? {
                        return Ok(());
                    }
                    idx += 1;
                } else {
                    break;
                }
            }
            if idx >= windows.len() {
                break;
            }
            let start = idx;
            while idx < windows.len() && !cache.contains_key(&key_of(&windows[idx])) {
                idx += 1;
            }
            let slice = &windows[start..idx];
            let mut keep = true;
            // Owned windows: the reading thread allocates and fills each one, and it moves into the
            // upload without a copy here, so this thread's time goes to queueing device work
            match *stored {
                Some(format) => loader.stream_stored_owned_while(slice, |window, bytes| {
                    let handle = workspace.process_owned_stored_chunk_in_vram(bytes, format, window.read_len())?;
                    keep = finish_window(window, handle, workspace.client(), workspace.channels(), aligner, cache, cached_bytes, &mut f)?;
                    Ok(keep)
                })?,
                None => loader.stream_owned_while(slice, |window, raw| {
                    let handle = workspace.process_owned_chunk_in_vram(raw, window.read_len());
                    keep = finish_window(window, handle, workspace.client(), workspace.channels(), aligner, cache, cached_bytes, &mut f)?;
                    Ok(keep)
                })?,
            }
            if !keep {
                break;
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_window(
    window: &HaloWindow,
    handle: Handle,
    client: &cubecl::prelude::Client,
    channels: usize,
    aligner: &mut Option<Box<dyn WindowAligner + '_>>,
    cache: &mut HashMap<(usize, u64, u64), Handle>,
    cached_bytes: &mut usize,
    f: &mut impl FnMut(&HaloWindow, &Handle) -> DspResult<bool>,
) -> DspResult<bool> {
    let handle = match aligner.as_mut() {
        Some(aligner) => aligner.align(&handle, window, window.read_len()),
        None => handle,
    };
    let k = (window.index, window.read_global.start, window.read_global.end);
    if !cache.contains_key(&k) {
        let elems = channels * window.read_len();
        let bytes = elems * size_of::<f32>();
        if *cached_bytes + bytes <= MAX_CACHE_BYTES {
            let copy = buffer::empty::<f32>(client, elems);
            buffer::copy::<f32>(client, &handle, &copy, elems);
            cache.insert(k, copy);
            *cached_bytes += bytes;
        }
    }
    f(window, &handle)
}

/// `count` evenly spaced windows of `len` samples, read with `halos`.
pub fn spread_windows(total: u64, count: usize, len: u64, halos: (u64, u64)) -> Vec<HaloWindow> {
    let len = len.clamp(1, total);
    let n = count.max(1) as u64;
    (0..n)
        .map(|i| {
            let start = if n > 1 { i * (total - len) / (n - 1) } else { 0 };
            HaloWindow::around(i as usize, start..start + len, halos.0, halos.1, total)
        })
        .collect()
}

/// Local samples of a window's own range whose snippet `[s − before, s + after)` fits the buffer.
pub fn snippet_fits(s: usize, before: usize, after: usize, len: usize) -> bool {
    s >= before && s + after <= len
}
