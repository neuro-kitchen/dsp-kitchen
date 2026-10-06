//! Conversions between wire messages and dsp-core / dsp-view types. Values are exact: rates and
//! times travel as fractions, samples as stored.

use dsp_core::{ChannelInfo, MemoryOrder, RationalTime, RecordingInfo, SampleFormat, SampleRate, SignalUnit};
use dsp_view::{Envelope, View};

use super::wire;
use super::PROTOCOL_VERSION;
use crate::error::{StreamError, StreamResult};

/// Values per column in `Columns.min_max`.
const PAIR: usize = 2;

fn format_to_wire(format: SampleFormat) -> wire::SampleFormat {
    match format {
        SampleFormat::I8 => wire::SampleFormat::I8,
        SampleFormat::I16 => wire::SampleFormat::I16,
        SampleFormat::U16 => wire::SampleFormat::U16,
        SampleFormat::I32 => wire::SampleFormat::I32,
        SampleFormat::F32 => wire::SampleFormat::F32,
        SampleFormat::F64 => wire::SampleFormat::F64,
    }
}

fn format_from_wire(format: i32) -> StreamResult<SampleFormat> {
    Ok(match wire::SampleFormat::try_from(format) {
        Ok(wire::SampleFormat::I8) => SampleFormat::I8,
        Ok(wire::SampleFormat::I16) => SampleFormat::I16,
        Ok(wire::SampleFormat::U16) => SampleFormat::U16,
        Ok(wire::SampleFormat::I32) => SampleFormat::I32,
        Ok(wire::SampleFormat::F32) => SampleFormat::F32,
        Ok(wire::SampleFormat::F64) => SampleFormat::F64,
        _ => return Err(StreamError::protocol(format!("unknown sample format {format}"))),
    })
}

fn unit_to_wire(unit: &SignalUnit) -> (wire::Unit, String) {
    let u = match unit {
        SignalUnit::Volt => wire::Unit::Volt,
        SignalUnit::Millivolt => wire::Unit::Millivolt,
        SignalUnit::Microvolt => wire::Unit::Microvolt,
        SignalUnit::Ampere => wire::Unit::Ampere,
        SignalUnit::Milliampere => wire::Unit::Milliampere,
        SignalUnit::Microampere => wire::Unit::Microampere,
        SignalUnit::Dimensionless => wire::Unit::Dimensionless,
        SignalUnit::Other(symbol) => return (wire::Unit::Other, symbol.clone()),
    };
    (u, String::new())
}

fn unit_from_wire(unit: i32, symbol: &str) -> StreamResult<SignalUnit> {
    Ok(match wire::Unit::try_from(unit) {
        Ok(wire::Unit::Volt) => SignalUnit::Volt,
        Ok(wire::Unit::Millivolt) => SignalUnit::Millivolt,
        Ok(wire::Unit::Microvolt) => SignalUnit::Microvolt,
        Ok(wire::Unit::Ampere) => SignalUnit::Ampere,
        Ok(wire::Unit::Milliampere) => SignalUnit::Milliampere,
        Ok(wire::Unit::Microampere) => SignalUnit::Microampere,
        Ok(wire::Unit::Dimensionless) => SignalUnit::Dimensionless,
        Ok(wire::Unit::Other) => SignalUnit::Other(symbol.to_string()),
        _ => return Err(StreamError::protocol(format!("unknown unit {unit}"))),
    })
}

fn ratio(numerator: &u64, denominator: &u64) -> wire::Ratio {
    wire::Ratio { numerator: *numerator, denominator: *denominator }
}

/// The `Header` describing `info` (what the server streams).
pub fn header_from_info(info: &RecordingInfo) -> wire::Header {
    wire::Header {
        protocol_version: PROTOCOL_VERSION,
        name: info.name.clone(),
        channels: info
            .channels
            .iter()
            .map(|c| {
                let (unit, unit_symbol) = unit_to_wire(&c.unit);
                wire::Channel { name: c.name.clone(), gain: c.gain, offset: c.offset, unit: unit as i32, unit_symbol }
            })
            .collect(),
        sample_rate_hz: Some({
            let r = info.sample_rate.as_ratio();
            ratio(r.numer(), r.denom())
        }),
        format: format_to_wire(info.format) as i32,
        samples: info.samples,
        start_time_s: Some({
            let r = info.start_time.as_ratio();
            ratio(r.numer(), r.denom())
        }),
        metadata: info.metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    }
}

/// The recording a `Header` describes, as received: samples arrive channel-major.
pub fn info_from_header(header: &wire::Header) -> StreamResult<RecordingInfo> {
    if header.protocol_version != PROTOCOL_VERSION {
        return Err(StreamError::protocol(format!("server speaks protocol {}, this client {PROTOCOL_VERSION}", header.protocol_version)));
    }
    let rate = header.sample_rate_hz.as_ref().ok_or_else(|| StreamError::protocol("header without a sample rate"))?;
    let start_time = match header.start_time_s {
        Some(t) => RationalTime::new(t.numerator, t.denominator)?,
        None => RationalTime::ZERO,
    };
    let channels = header
        .channels
        .iter()
        .map(|c| Ok(ChannelInfo { name: c.name.clone(), gain: c.gain, offset: c.offset, unit: unit_from_wire(c.unit, &c.unit_symbol)? }))
        .collect::<StreamResult<Vec<_>>>()?;
    Ok(RecordingInfo {
        name: header.name.clone(),
        channels,
        samples: header.samples,
        sample_rate: SampleRate::from_ratio(rate.numerator, rate.denominator)?,
        format: format_from_wire(header.format)?,
        order: MemoryOrder::ChannelMajor,
        start_time,
        metadata: header.metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    })
}

/// Scaled values (`stored · gain + offset`, each channel in its unit) of a signal frame,
/// channel-major (`channels × frame.samples`).
pub fn decode_signal(frame: &wire::SignalFrame, info: &RecordingInfo) -> StreamResult<Vec<f32>> {
    let n = frame.samples as usize;
    let row_bytes = n * info.format.bytes();
    let expected = info.channel_count() * row_bytes;
    if frame.data.len() != expected {
        return Err(StreamError::protocol(format!("signal frame of {} bytes, expected {expected}", frame.data.len())));
    }
    let mut out = vec![0.0f32; info.channel_count() * n];
    for ((row, bytes), c) in out.chunks_exact_mut(n.max(1)).zip(frame.data.chunks_exact(row_bytes.max(1))).zip(&info.channels) {
        info.format.decode(bytes, row, c.gain, c.offset);
    }
    Ok(out)
}

pub fn request_from_view(view: &View) -> StreamResult<wire::ViewRequest> {
    let channels = view.channels.iter().map(|&c| u32::try_from(c)).collect::<Result<_, _>>().map_err(|_| StreamError::protocol("channel index beyond 32 bits"))?;
    let width = u32::try_from(view.width).map_err(|_| StreamError::protocol("view width beyond 32 bits"))?;
    Ok(wire::ViewRequest { channels, start: view.start, end: view.end, width })
}

pub fn view_from_request(request: &wire::ViewRequest) -> View {
    View { channels: request.channels.iter().map(|&c| c as usize).collect(), start: request.start, end: request.end, width: request.width as usize }
}

pub fn frame_from_envelope(envelope: StreamResult<Envelope>) -> wire::EnvelopeFrame {
    use wire::envelope_frame::Content;
    let content = match envelope {
        Ok(Envelope::Samples(values)) => Content::Samples(wire::Samples { values }),
        Ok(Envelope::Columns { values, complete }) => Content::Columns(wire::Columns { min_max: values.into_iter().flatten().collect(), complete }),
        Err(e) => Content::Error(e.to_string()),
    };
    wire::EnvelopeFrame { content: Some(content) }
}

pub fn envelope_from_frame(frame: wire::EnvelopeFrame) -> StreamResult<Envelope> {
    use wire::envelope_frame::Content;
    match frame.content {
        Some(Content::Samples(s)) => Ok(Envelope::Samples(s.values)),
        Some(Content::Columns(c)) => {
            if c.min_max.len() % PAIR != 0 {
                return Err(StreamError::protocol("columns with an odd number of values"));
            }
            Ok(Envelope::Columns { values: c.min_max.as_chunks::<PAIR>().0.to_vec(), complete: c.complete })
        }
        Some(Content::Error(message)) => Err(StreamError::Protocol(format!("server: {message}"))),
        None => Err(StreamError::protocol("empty envelope frame")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_exactly() {
        let mut info = RecordingInfo::new("rec", 3, 1_000, SampleRate::new(30_000.12).unwrap(), SampleFormat::I16, MemoryOrder::TimeMajor)
            .with_gain(0.195, SignalUnit::Microvolt);
        info.channels[2].unit = SignalUnit::Other("Pa".into());
        info.start_time = RationalTime::new(7, 3).unwrap();
        info.metadata.insert("device".into(), "probe".into());
        let back = info_from_header(&header_from_info(&info)).unwrap();
        assert_eq!(back, RecordingInfo { order: MemoryOrder::ChannelMajor, ..info });
    }

    #[test]
    fn signal_frames_decode_to_scaled_rows() {
        let info = RecordingInfo::new("rec", 2, 4, SampleRate::new(1000.0).unwrap(), SampleFormat::I16, MemoryOrder::ChannelMajor).with_gain(0.5, SignalUnit::Microvolt);
        let stored: Vec<i16> = vec![2, 4, 6, 8, -2, -4, -6, -8];
        let frame = wire::SignalFrame { sequence: 0, first_sample: 0, samples: 4, data: stored.iter().flat_map(|v| v.to_le_bytes()).collect(), sent_unix_ns: 0 };
        assert_eq!(decode_signal(&frame, &info).unwrap(), vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]);
        let short = wire::SignalFrame { data: vec![0; 3], ..frame };
        assert!(decode_signal(&short, &info).is_err());
    }

    #[test]
    fn views_and_envelopes_round_trip() {
        let view = View { channels: vec![4, 1], start: 10, end: 1_000, width: 64 };
        assert_eq!(view_from_request(&request_from_view(&view).unwrap()), view);
        let columns = Envelope::Columns { values: vec![[-1.0, 2.0], [f32::NAN, f32::NAN]], complete: false };
        let back = envelope_from_frame(frame_from_envelope(Ok(columns))).unwrap();
        let Envelope::Columns { values, complete } = back else { panic!() };
        assert!(!complete && values[0] == [-1.0, 2.0] && values[1][0].is_nan());
        let samples = Envelope::Samples(vec![1.0, 2.0]);
        assert_eq!(envelope_from_frame(frame_from_envelope(Ok(samples.clone()))).unwrap(), samples);
        assert!(envelope_from_frame(frame_from_envelope(Err(StreamError::protocol("out of range")))).is_err());
    }
}
