# Moving data to the device

A spike sorter reads hours of recording, window by window. Whatever the kernels do, every window
must cross from disk to host memory to the device, so **how** it crosses sets a floor under the
run time. Two rules shape the code:

1. **A window crosses each way at most once.** It is uploaded, every stage (filters, whitening,
   delays, detection) runs on the device copy, and only results come back: spikes, or the clips
   template learning needs.
2. **It crosses in its most compact form.** Recordings are usually stored as 16-bit integers with a
   gain and an offset per channel. Converting them to `f32` on the host doubles the bytes on the bus
   and spends a host thread on arithmetic the device does for free.

## How the Kilosort4 / EMUsort runner does it

`DeviceWindows` (in `sorters/kilosort4/runner.rs`) is the one path every pass uses (preprocessing
fit, clip collection, detection):

- A background thread reads the next window while the device works on the current one
  (`dsp_core::WindowLoader`, two recycled buffers: host memory stays at two windows whatever the
  recording length).
- At start it asks the source once whether it can return **stored** values
  (`RecordingSource::read_stored`; the NWB, raw binary, Zarr and mtscomp readers can). If so, the
  window is uploaded as raw integer bytes and `PipelineWorkspace::process_stored_chunk_in_vram`
  unpacks and scales them on the device (WGSL has no 16-bit integers, so the kernel reads 32-bit
  words and sign-extends). If not, it uploads scaled `f32`.
- The window is preprocessed in device buffers the workspace keeps (no allocation per window),
  aligned for channel delays when EMUsort estimated them, and handed to the pass.

The test `stored_upload_matches_f32_upload` runs Kilosort4 on an int16 recording both ways (the
second through a wrapper that hides stored reads) and requires the same spikes.

## What it saves

For 384 channels in 60 000-sample windows, an `f32` window is 92 MB and an int16 window 46 MB.
Over PCIe 3.0 ×16 (~12 GB/s in practice) that is about 4 ms saved per window, plus the host-side
conversion it no longer does, and half the pinned staging memory. It matters most where the
device work per window is small (detection on a fast GPU), because then the transfer is the run
time.
