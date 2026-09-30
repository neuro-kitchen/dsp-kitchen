//! RS4 / Synapse `*_log.txt` next to SEV files: where recording started and any gaps.
//! `<STORE>…_log.txt` holds lines like `recording started at sample: 2` and
//! `gap detected. last saved sample: 1000, new saved sample: 1100`.

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SevLog {
    pub store: String,
    pub hour: u32,
    pub start_sample: Option<u64>,
    /// (last saved sample, next saved sample) for every gap.
    pub gaps: Vec<(u64, u64)>,
}

pub fn parse(file_name: &str, text: &str) -> SevLog {
    let mut log = SevLog { store: file_name.chars().take(4).collect(), ..Default::default() };
    log.hour = file_name.rfind('-').and_then(|i| file_name[i + 1..].split('h').next()?.parse().ok()).unwrap_or(0);
    let number_after = |line: &str, key: &str| -> Option<u64> {
        let rest = &line[line.find(key)? + key.len()..];
        rest.trim_start().chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok()
    };
    for line in text.lines() {
        if let Some(n) = number_after(line, "recording started at sample:") {
            log.start_sample = Some(n);
        }
        if line.contains("gap detected") {
            if let (Some(a), Some(b)) = (number_after(line, "last saved sample:"), number_after(line, "new saved sample:")) {
                log.gaps.push((a, b));
            }
        }
    }
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_and_gaps() {
        let l = parse("RSn1_log.txt", "recording started at sample: 2\ngap detected. last saved sample: 1000, new saved sample: 1100\n");
        assert_eq!((l.store.as_str(), l.start_sample, l.gaps.clone()), ("RSn1", Some(2), vec![(1000, 1100)]));
    }
}
