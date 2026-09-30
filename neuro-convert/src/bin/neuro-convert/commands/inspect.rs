use std::path::Path;
use std::time::Instant;

use neuro_convert::OpenOptions;

pub fn run(path: &Path, read_sec: Option<f64>, options: &OpenOptions) -> anyhow::Result<()> {
    let detections = neuro_convert::detect(path);
    let t = Instant::now();
    let s = neuro_convert::open(path, options)?;
    let opened = t.elapsed();

    let p = &s.provenance;
    println!("=== {} ===", path.display());
    println!("Format:     {} ({})", p.format, p.version.as_deref().unwrap_or("version unknown"));
    if detections.len() > 1 {
        println!("Also matched: {:?}", detections.iter().skip(1).map(|d| d.format).collect::<Vec<_>>());
    }
    println!("Opened in:  {:.2} s", opened.as_secs_f64());
    let total: u64 = p.files.iter().map(|f| f.bytes).sum();
    println!("Files:      {} ({:.2} GB)", p.files.len(), total as f64 / 1e9);

    let m = &s.metadata;
    println!("\n-- Session");
    let show = |k: &str, v: Option<&str>| {
        if let Some(v) = v {
            println!("{k:12}{v}");
        }
    };
    show("Experiment:", m.experiment.as_deref());
    show("Subject:", m.subject.id.as_deref());
    show("User:", m.experimenters.first().map(String::as_str));
    show("Start:", m.start_time.as_deref());
    show("Stop:", m.stop_time.as_deref());
    println!("Duration:   {:.1} s ({:.2} h)", s.duration(), s.duration() / 3600.0);
    for d in &m.devices {
        println!("Device:     {} — {}", d.name, d.description);
    }
    for n in &m.notes {
        println!("Note:       {n}");
    }

    println!("\n-- Streams ({})", s.recordings.len());
    println!("  {:6} {:>4} {:>11} {:>12} {:>9} {:>8} {:>6} {:8}  description", "name", "ch", "rate (Hz)", "samples", "duration", "stored", "unit", "storage");
    for r in &s.recordings {
        let i = r.info();
        // Source-specific storage tag (e.g. TDT `tev` / `sev v3`), when the input records one
        let storage = match (i.metadata.get("tdt_storage"), i.metadata.get("sev_version")) {
            (Some(s), Some(v)) => format!("{s} v{v}"),
            (Some(s), None) => s.clone(),
            _ => String::new(),
        };
        println!(
            "  {:6} {:>4} {:>11.4} {:>12} {:>8.1}s {:>8} {:>6} {storage:8}  {}",
            i.name,
            i.channel_count(),
            i.sample_rate,
            i.samples,
            i.duration(),
            i.stored_as.name(),
            i.unit,
            i.description
        );
    }

    println!("\n-- Events ({})", s.events.len());
    for e in &s.events {
        let span = match (e.onsets.first(), e.onsets.last()) {
            (Some(a), Some(b)) => format!("{a:.3}–{b:.3} s"),
            _ => "empty".into(),
        };
        let kind = if e.channels > 1 { format!("scalar x{}", e.channels) } else if e.offsets.is_some() { "interval".into() } else { "event".into() };
        let values: Vec<String> = e.values.iter().take(e.channels.min(4)).map(|v| format!("{v:.4}")).collect();
        println!("  {:6} {:>6} {:10} {:>22}  first values [{}]  {}", e.name, e.len(), kind, span, values.join(", "), e.description);
    }
    if !s.snippets.is_empty() {
        println!("\n-- Snippets ({})", s.snippets.len());
        for sn in &s.snippets {
            println!("  {:6} {:>6} snippets x {} samples @ {:.1} Hz", sn.name, sn.len(), sn.samples_per_snippet, sn.sample_rate);
        }
    }
    if !s.tables.is_empty() {
        println!("\n-- Tables ({})", s.tables.len());
        for t in &s.tables {
            println!("  {:16} {} rows x {} columns — {}", t.name, t.rows.len(), t.columns.len(), t.description);
        }
    }
    if !p.warnings.is_empty() {
        println!("\n-- Warnings ({})", p.warnings.len());
        for w in &p.warnings {
            println!("  ! {w}");
        }
    }

    if let Some(sec) = read_sec {
        let Some(r) = s.recordings.iter().max_by_key(|r| r.info().stored_bytes()) else { return Ok(()) };
        let i = r.info();
        let chunk = (i.sample_rate as u64).clamp(1, i.samples.max(1));
        let chunks = ((sec * i.sample_rate) as u64 / chunk).max(1);
        let start = (i.samples / 2).saturating_sub(chunks * chunk / 2);
        let channels: Vec<usize> = (0..i.channel_count()).collect();
        let mut buf = vec![0.0f32; channels.len() * chunk as usize];
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let t = Instant::now();
        let mut s0 = start;
        for _ in 0..chunks {
            let s1 = (s0 + chunk).min(i.samples);
            let len = channels.len() * (s1 - s0) as usize;
            r.read(&channels, s0..s1, &mut buf[..len])?;
            for &v in &buf[..(s1 - s0) as usize] {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            s0 = s1;
        }
        let secs = t.elapsed().as_secs_f64();
        let read = (s0 - start) as f64 / i.sample_rate;
        println!(
            "\nRead {}: {read:.1} s x {} ch in {secs:.2} s ({:.0} Msamples/s, {:.0}x real time); ch0 range {lo:.6} .. {hi:.6} {}",
            i.name,
            channels.len(),
            (s0 - start) as f64 * channels.len() as f64 / secs / 1e6,
            read / secs,
            i.unit
        );
    }
    Ok(())
}

/// JSON summary: per store the shape and the first values (for comparisons with other readers).
/// `at_sec` picks where stream samples are taken (default: the start).
pub fn json(path: &Path, at_sec: Option<f64>, options: &OpenOptions) -> anyhow::Result<()> {
    let s = neuro_convert::open(path, options)?;
    let recordings: Vec<serde_json::Value> = s
        .recordings
        .iter()
        .map(|r| {
            let i = r.info();
            let first = (at_sec.unwrap_or(0.0) * i.sample_rate).ceil() as u64;
            let n = 5.min(i.samples.saturating_sub(first));
            let mut v = vec![0.0f32; n as usize];
            let _ = r.read(&[0], first..first + n, &mut v);
            serde_json::json!({ "name": i.name, "channels": i.channel_count(), "rate": i.sample_rate, "samples": i.samples,
                "start_time": i.start_time, "first_sample": first, "values": v, "stored_as": i.stored_as.name() })
        })
        .collect();
    let events: Vec<serde_json::Value> = s
        .events
        .iter()
        .map(|e| serde_json::json!({ "name": e.name, "count": e.len(), "channels": e.channels,
            "onsets": &e.onsets[..3.min(e.len())], "values": &e.values[..3.min(e.values.len())],
            "offsets": e.offsets.as_ref().map(|o| o[..3.min(o.len())].to_vec()),
            "labels": &e.labels[..3.min(e.labels.len())] }))
        .collect();
    let snippets: Vec<serde_json::Value> = s
        .snippets
        .iter()
        .map(|sn| serde_json::json!({ "name": sn.name, "count": sn.len(), "samples_per_snippet": sn.samples_per_snippet,
            "rate": sn.sample_rate, "timestamps": &sn.timestamps[..3.min(sn.len())], "channels": &sn.channels[..3.min(sn.len())],
            "sort_codes": &sn.sort_codes[..3.min(sn.len())], "values": &sn.data[..3.min(sn.data.len())] }))
        .collect();
    let out = serde_json::json!({ "format": s.provenance.format, "version": s.provenance.version, "duration": s.duration(),
        "recordings": recordings, "events": events, "snippets": snippets, "warnings": s.provenance.warnings });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
