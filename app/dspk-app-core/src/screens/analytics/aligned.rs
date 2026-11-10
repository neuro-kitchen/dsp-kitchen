//! A signal aligned to events: every channel's samples from `pre` before to `post` after each of
//! its events (or of every event), averaged with the spread across events (mean ± SD, the
//! template look). Computed off the UI thread with `dspk_ops::math::RunningMoments`; drawn into a
//! [`Frame`] per channel.

use dspk_core::DspResult;
use dspk_ops::math::RunningMoments;

use crate::data::events::Event;
use crate::ui::plot::canvas::{Canvas, Frame, Pixel};

/// Which events each channel is aligned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignTo {
    /// A channel to the events found on it (e.g. each muscle to its own contractions).
    #[default]
    OwnEvents,
    /// Every channel to every event of the set.
    AllEvents,
}

/// One channel's average around the events.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelAverage {
    pub channel: usize,
    /// Events averaged (those whose window lies inside the recording).
    pub trials: usize,
    pub mean: Vec<f32>,
    /// Across-event standard deviation (sample, ddof 1).
    pub sd: Vec<f32>,
    /// The first events' own windows (as many as asked for), to draw under the mean.
    pub snippets: Vec<Vec<f32>>,
}

/// The average of every channel, on a common time axis.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedAverage {
    /// Time of each sample from the event, in milliseconds.
    pub time_ms: Vec<f64>,
    pub channels: Vec<ChannelAverage>,
}

/// Averages `channels` around `events` (seconds) of a signal of `total` samples at `rate`, reading
/// with `read(channel, samples, out)`: windows `[−pre_s, +post_s)`; events whose window leaves the
/// recording are skipped.
pub fn aligned_average(
    read: impl Fn(usize, std::ops::Range<u64>, &mut [f32]) -> DspResult<()>,
    total: u64,
    rate: f64,
    channels: &[usize],
    events: &[Event],
    align: AlignTo,
    pre_s: f64,
    post_s: f64,
    keep: usize,
) -> DspResult<AlignedAverage> {
    let pre = (pre_s * rate).round().max(0.0) as u64;
    let post = (post_s * rate).round().max(1.0) as u64;
    let len = (pre + post) as usize;
    let time_ms = (0..len).map(|i| (i as f64 - pre as f64) * 1e3 / rate).collect();
    let mut buf = vec![0.0f32; len];
    let mut out = Vec::with_capacity(channels.len());
    for &ch in channels {
        let mut m = RunningMoments::new(len);
        let mut snippets = Vec::new();
        for e in events.iter().filter(|e| align == AlignTo::AllEvents || e.channel as usize == ch) {
            let at = (e.onset_s * rate).round() as i64;
            let (a, b) = (at - pre as i64, at + post as i64);
            if a < 0 || b as u64 > total {
                continue;
            }
            read(ch, a as u64..b as u64, &mut buf)?;
            m.push(&buf);
            if snippets.len() < keep {
                snippets.push(buf.clone());
            }
        }
        let trials = m.count() as usize;
        out.push(ChannelAverage {
            channel: ch,
            trials,
            mean: m.mean().iter().map(|&v| v as f32).collect(),
            sd: m.std(1).into_iter().map(|v| v as f32).collect(),
            snippets,
        });
    }
    Ok(AlignedAverage { time_ms, channels: out })
}

/// One channel's average drawn into `width × height`: the band (mean ± SD) faint, the mean on top,
/// a vertical line at the event (`event_frac` of the width) and a horizontal one at zero; y fits
/// the band.
pub fn draw(avg: &ChannelAverage, event_frac: f32, width: u32, height: u32, background: Pixel, grid: Pixel, colour: Pixel, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut c = frame.canvas();
    c.pixels.fill(background);
    let n = avg.mean.len();
    if n < 2 || avg.trials == 0 {
        return frame;
    }
    let snip = avg.snippets.iter().flatten().copied();
    let lo = avg.mean.iter().zip(&avg.sd).map(|(m, s)| m - s).chain(snip.clone()).fold(f32::INFINITY, f32::min).min(0.0);
    let hi = avg.mean.iter().zip(&avg.sd).map(|(m, s)| m + s).chain(snip).fold(f32::NEG_INFINITY, f32::max).max(0.0);
    let span = (hi - lo).max(1e-12);
    let (w, h) = (width as f32, height as f32);
    let pad = 4.0 * scale;
    let y = |v: f32| pad + (1.0 - (v - lo) / span) * (h - 2.0 * pad);
    let x = |i: usize| i as f32 / (n - 1) as f32 * (w - 1.0);
    let zero_y = y(0.0);
    c.rect(0.0, zero_y - 0.5, w, zero_y + 0.5, grid, 0.8);
    let ex = event_frac.clamp(0.0, 1.0) * (w - 1.0);
    c.rect(ex - 0.5 * scale, 0.0, ex + 0.5 * scale, h, grid, 1.0);
    // Each event's own window, faint, under the band and the mean
    let faint = (0.6 / (avg.snippets.len() as f32).sqrt().max(1.0)).clamp(0.08, 0.35);
    for s in &avg.snippets {
        let points: Vec<(f32, f32)> = (0..n).map(|i| (x(i), y(s[i]))).collect();
        polyline(&mut c, &points, colour, faint, 1.0 * scale);
    }
    for col in 0..width as usize {
        let i = ((col as f32 / (w - 1.0).max(1.0)) * (n - 1) as f32).round() as usize;
        let (m, s) = (avg.mean[i], avg.sd[i]);
        c.span_aa_alpha(col, y(m + s), y(m - s), colour, 0.22);
    }
    let points: Vec<(f32, f32)> = (0..n).map(|i| (x(i), y(avg.mean[i]))).collect();
    polyline(&mut c, &points, colour, 1.0, 1.5 * scale);
    frame
}

/// A line through `points` (pixels), one vertical span per column between consecutive points.
pub fn polyline(canvas: &mut Canvas, points: &[(f32, f32)], c: Pixel, alpha: f32, width: f32) {
    for w in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        let (xa, xb) = (x0.round() as i64, x1.round() as i64);
        for x in xa.max(0)..=xb.max(xa) {
            if x as usize >= canvas.width {
                break;
            }
            let f = if xb > xa { (x - xa) as f32 / (xb - xa) as f32 } else { 0.0 };
            let f2 = if xb > xa { ((x + 1 - xa) as f32 / (xb - xa) as f32).min(1.0) } else { 1.0 };
            let (ya, yb) = (y0 + (y1 - y0) * f, y0 + (y1 - y0) * f2);
            canvas.span_aa_alpha(x as usize, ya.min(yb) - width * 0.5, ya.max(yb) + width * 0.5, c, alpha);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 kHz, 2 channels: a ramp (value = sample) on channel 0, a constant 5 on channel 1.
    fn read(ch: usize, r: std::ops::Range<u64>, out: &mut [f32]) -> DspResult<()> {
        for (o, s) in out.iter_mut().zip(r) {
            *o = if ch == 0 { s as f32 } else { 5.0 };
        }
        Ok(())
    }

    #[test]
    fn averages_each_channel_around_its_events() {
        let ev = |ch: u32, t: f64| Event { channel: ch, onset_s: t, offset_s: None, peak_value: 0.0 };
        let events = [ev(0, 0.1), ev(0, 0.3), ev(1, 0.2), ev(0, 0.001)];
        let a = aligned_average(read, 1000, 1000.0, &[0, 1], &events, AlignTo::OwnEvents, 0.01, 0.02, 1).unwrap();
        assert_eq!(a.channels[0].snippets.len(), 1, "one kept");
        assert_eq!(a.time_ms.len(), 30);
        assert_eq!((a.time_ms[0], a.time_ms[10]), (-10.0, 0.0));
        // Channel 0: events at 100 and 300 (the one at 1 ms has no room before it): mean at t = 0 is 200
        let c0 = &a.channels[0];
        assert_eq!(c0.trials, 2);
        assert_eq!(c0.mean[10], 200.0);
        assert!((c0.sd[10] - 141.42136).abs() < 1e-3);
        // Channel 1 to its own event only; to all events: 3 usable
        assert_eq!(a.channels[1].trials, 1);
        let all = aligned_average(read, 1000, 1000.0, &[1], &events, AlignTo::AllEvents, 0.01, 0.02, 0).unwrap();
        assert_eq!((all.channels[0].trials, all.channels[0].mean[0], all.channels[0].sd[0]), (3, 5.0, 0.0));
        let f = draw(&c0.clone(), 1.0 / 3.0, 120, 40, Pixel::rgb(0, 0, 0), Pixel::rgb(60, 60, 60), Pixel::rgb(250, 200, 50), 1.0);
        assert!(f.pixels().iter().any(|p| p.r > 200 && p.g > 150), "the mean is drawn");
    }
}
