//! Windows of a recording on their way to the device ([`DeviceWindows`]), shared by every sorter's
//! passes: read on a background thread, uploaded once in their compact stored form, preprocessed
//! on the device and, for sorters that need it, aligned ([`WindowAligner`]).

use cubecl::server::Handle;
use dspk_core::{DspResult, HaloWindow, RecordingSource, SampleFormat, WindowLoader};
use dspk_ops::pipeline::PipelineWorkspace;

/// A step applied to every preprocessed window on the device before a sorter sees it (EMUsort's
/// per-channel delay alignment, in `dspk-ks4`).
pub trait WindowAligner {
    /// The aligned `[channels, samples]` window for `window` (`samples` per channel).
    fn align(&mut self, window: &Handle, samples: usize) -> Handle;
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
        Self { loader: WindowLoader::new(source), stored, workspace, aligner }
    }

    /// Calls `f(window, prepared)` for every window of `windows` in order (`prepared`: the
    /// `[channels, window.read_len()]` preprocessed window on the device, valid until the next
    /// call), until `f` returns `false`.
    pub fn for_each_while(&mut self, windows: &[HaloWindow], mut f: impl FnMut(&HaloWindow, &Handle) -> DspResult<bool>) -> DspResult<()> {
        let Self { loader, stored, workspace, aligner } = self;
        let mut finish = |window: &HaloWindow, handle: Handle| {
            let handle = match aligner.as_mut() {
                Some(aligner) => aligner.align(&handle, window.read_len()),
                None => handle,
            };
            f(window, &handle)
        };
        // Owned windows: the reading thread allocates and fills each one, and it moves into the
        // upload without a copy here, so this thread's time goes to queueing device work
        match *stored {
            Some(format) => loader.stream_stored_owned_while(windows, |window, bytes| {
                let handle = workspace.process_owned_stored_chunk_in_vram(bytes, format, window.read_len())?;
                finish(window, handle)
            }),
            None => loader.stream_owned_while(windows, |window, raw| {
                let handle = workspace.process_owned_chunk_in_vram(raw, window.read_len());
                finish(window, handle)
            }),
        }
    }
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
