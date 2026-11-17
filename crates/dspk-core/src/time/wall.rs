//! Wall-clock time as files record it: seconds since 1970 (UTC) and RFC 3339, also in a browser.
//! No time library: days to the civil date by Howard Hinnant's method.

/// Seconds since 1970 (UTC).
pub fn unix_now() -> u64 {
    web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `unix_secs` as UTC RFC 3339 (`2026-11-17T10:52:03Z`).
pub fn rfc3339_utc(unix_secs: u64) -> String {
    let (days, rem) = ((unix_secs / 86_400) as i64, unix_secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Now, as UTC RFC 3339.
pub fn now_rfc3339() -> String {
    rfc3339_utc(unix_now())
}

/// A file time (modified, created) as UTC RFC 3339; `None` before 1970 or unknown.
pub fn file_time_rfc3339(t: std::io::Result<std::time::SystemTime>) -> Option<String> {
    let secs = t.ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    Some(rfc3339_utc(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_the_civil_ones() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_763_377_923), "2025-11-17T11:12:03Z");
        assert_eq!(rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
    }
}
