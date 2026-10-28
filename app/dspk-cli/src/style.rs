//! The CLI's colours, in one place: clap's help and every command's output use these. `anstream`
//! drops them when the output is not a terminal, or with `NO_COLOR` set.

use anstyle::{AnsiColor, Effects, Style};
use clap::builder::Styles;

/// Section headings and command names.
pub const HEADING: Style = AnsiColor::Green.on_default().effects(Effects::BOLD);
/// Values worth reading first: paths, counts, runtimes.
pub const VALUE: Style = AnsiColor::Cyan.on_default();
/// Labels and secondary detail.
pub const DIM: Style = Style::new().effects(Effects::DIMMED);
pub const OK: Style = AnsiColor::Green.on_default().effects(Effects::BOLD);
pub const WARN: Style = AnsiColor::Yellow.on_default().effects(Effects::BOLD);
pub const ERROR: Style = AnsiColor::Red.on_default().effects(Effects::BOLD);

/// clap's help in the same colours.
pub fn clap_styles() -> Styles {
    Styles::styled()
        .header(HEADING)
        .usage(HEADING)
        .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
        .placeholder(VALUE)
        .error(ERROR)
        .valid(OK)
        .invalid(WARN)
}

/// `text` in `style` (reset after).
pub fn paint(style: Style, text: impl std::fmt::Display) -> String {
    format!("{style}{text}{style:#}")
}

/// Bytes as `12.3 MB` (decimal units).
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1000.0 && unit + 1 < UNITS.len() {
        v /= 1000.0;
        unit += 1;
    }
    if unit == 0 { format!("{n} B") } else { format!("{v:.1} {}", UNITS[unit]) }
}

/// A count as `1.2M` / `30k` / `512`.
pub fn count(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", n as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1}M", n as f64 / 1e6),
        _ => format!("{:.1}G", n as f64 / 1e9),
    }
}

/// A rate as `30 kHz` / `2.5 kHz` / `500 Hz`.
pub fn hz(rate: f64) -> String {
    if rate >= 1000.0 { format!("{} kHz", trim(rate / 1000.0)) } else { format!("{} Hz", trim(rate)) }
}

fn trim(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_read_naturally() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(14_400_000), "14.4 MB");
        assert_eq!(count(1_234_567), "1.2M");
        assert_eq!(hz(30_000.0), "30 kHz");
        assert_eq!(hz(2_500.0), "2.5 kHz");
        assert_eq!(hz(500.0), "500 Hz");
    }
}
