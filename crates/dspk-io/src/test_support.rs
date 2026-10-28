//! Assertions shared by the format tests of this crate and of crates that add formats
//! (feature `test-support`).

use dspk_core::{MemoryOrder, RecordingSource};

/// `read_stored` decoded with each channel's gain and offset equals `read`.
pub fn assert_stored_matches(rec: &dyn RecordingSource, channels: &[usize], samples: std::ops::Range<u64>) {
    let info = rec.info();
    let n = (samples.end - samples.start) as usize;
    let mut values = vec![0.0f32; channels.len() * n];
    rec.read(channels, samples.clone(), &mut values).unwrap();
    let mut bytes = vec![0u8; channels.len() * n * info.format.bytes()];
    rec.read_stored(channels, samples, &mut bytes).unwrap();
    let mut decoded = vec![0.0f32; values.len()];
    for (i, &c) in channels.iter().enumerate() {
        let ch = &info.channels[c];
        let b = info.format.bytes();
        info.format.decode(&bytes[i * n * b..(i + 1) * n * b], &mut decoded[i * n..(i + 1) * n], ch.gain, ch.offset);
    }
    assert_eq!(decoded, values, "{}", info.name);
}

/// `read_native`, put back in channel-major order, equals `read` of every channel.
pub fn assert_native_matches(rec: &dyn RecordingSource, samples: std::ops::Range<u64>) {
    let (nch, n) = (rec.info().channel_count(), (samples.end - samples.start) as usize);
    let all: Vec<usize> = (0..nch).collect();
    let mut values = vec![0.0f32; nch * n];
    rec.read(&all, samples.clone(), &mut values).unwrap();
    let mut native = vec![0.0f32; nch * n];
    let order = rec.read_native(samples, &mut native).unwrap();
    if order == MemoryOrder::TimeMajor {
        native = (0..nch).flat_map(|c| (0..n).map(move |t| (c, t))).map(|(c, t)| native[t * nch + c]).collect();
    }
    assert_eq!(native, values, "{} {order:?}", rec.info().name);
}
