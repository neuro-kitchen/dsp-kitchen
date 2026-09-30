//! Minimal calendar arithmetic for ISO 8601 timestamps (no time zone handling).

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

/// Seconds since the epoch → `YYYY-MM-DDTHH:MM:SS[.ffffff]` (in whatever zone `secs` is in).
pub fn format_iso(secs: f64) -> String {
    let whole = secs.floor();
    let frac = secs - whole;
    let whole = whole as i64;
    let (y, m, d) = civil_from_days(whole.div_euclid(86_400));
    let t = whole.rem_euclid(86_400);
    let base = format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", t / 3600, t / 60 % 60, t % 60);
    let micros = (frac * 1e6).round() as i64;
    if micros > 0 && micros < 1_000_000 { format!("{base}.{micros:06}") } else { base }
}

/// Parses `YYYY-MM-DDTHH:MM:SS[.fff]` (any zone suffix is ignored) into epoch seconds.
pub fn parse_iso(s: &str) -> Option<f64> {
    let (date, time) = s.trim().split_once(['T', ' '])?;
    let mut dp = date.split('-').map(|v| v.parse::<i64>().ok());
    let (y, m, d) = (dp.next()??, dp.next()??, dp.next()??);
    // Drop a zone suffix: Z, +hh:mm or -hh:mm
    let time = time.trim_end_matches('Z');
    let time = time.rfind(['+', '-']).map_or(time, |i| &time[..i]);
    let mut tp = time.split(':');
    let h: i64 = tp.next()?.parse().ok()?;
    let mi: i64 = tp.next()?.parse().ok()?;
    let sec: f64 = tp.next().unwrap_or("0").parse().ok()?;
    Some((days_from_civil(y, m, d) * 86_400 + h * 3600 + mi * 60) as f64 + sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_and_known_dates() {
        assert_eq!(format_iso(0.0), "1970-01-01T00:00:00");
        // TSQ start marker of the test block (UTC)
        assert_eq!(format_iso(1_740_601_556.0), "2025-02-26T20:25:56");
        let t = parse_iso("2025-02-26T15:25:56").unwrap();
        assert_eq!(format_iso(t + 2833.5), "2025-02-26T16:13:09.500000");
        assert_eq!(parse_iso("2024-02-29T00:00:00").map(format_iso).unwrap(), "2024-02-29T00:00:00");
        assert!(parse_iso("garbage").is_none());
        assert_eq!(parse_iso("2025-01-01T12:00:00-05:00"), parse_iso("2025-01-01T12:00:00"));
        assert_eq!(parse_iso("2025-01-01T12:00:00+01:00"), parse_iso("2025-01-01T12:00:00Z"));
    }
}
