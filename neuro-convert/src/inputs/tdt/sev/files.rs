//! Finding SEV files and grouping them by store, channel and hour.
//!
//! Names look like `<tank>_<block>_<STORE>_Ch<n>[-<h>h].sev`: the store is the last 4-character
//! token between underscores, the channel follows `_Ch`, and RS4 splits long recordings into
//! hour files (`-1h`, `-2h`, …) that are concatenated in order.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::header::{SevHeader, HEADER_BYTES};
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct SevFile {
    pub path: PathBuf,
    pub store: String,
    pub channel: u16,
    pub hour: u32,
    pub header: SevHeader,
    /// Payload bytes after the header.
    pub data_bytes: u64,
}

/// Store name, channel and hour from a file name (without extension).
pub fn parse_name(stem: &str) -> (Option<String>, Option<u16>, u32) {
    let lower = stem.to_ascii_lowercase();
    let channel = lower.rfind("_ch").and_then(|i| {
        let digits: String = stem[i + 3..].chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    });
    let hour = stem.rfind('-').and_then(|i| stem[i + 1..].strip_suffix('h')?.parse().ok()).unwrap_or(0);
    // Last `_XXXX_` token: exactly four characters between underscores
    let parts: Vec<&str> = stem.split('_').collect();
    let store = parts
        .iter()
        .enumerate()
        .rev()
        .find(|(i, p)| *i > 0 && *i + 1 < parts.len() && p.chars().count() == 4)
        .map(|(_, p)| p.to_string());
    (store, channel, hour)
}

pub fn read_file(path: &Path) -> Result<SevFile> {
    let mut head = [0u8; HEADER_BYTES];
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
    f.read_exact(&mut head).map_err(|e| Error::io(path, e))?;
    let len = f.metadata().map_err(|e| Error::io(path, e))?.len();
    let header = SevHeader::parse(&head)?;
    let stem = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let (name_store, name_channel, hour) = parse_name(&stem);
    let store = header.event_name.clone().or(name_store).ok_or_else(|| Error::format("tdt-sev", format!("{}: no store name", path.display())))?;
    let channel = header.channel.filter(|&c| c > 0).or(name_channel).unwrap_or(1);
    Ok(SevFile { path: path.to_path_buf(), store, channel, hour, header, data_bytes: len.saturating_sub(HEADER_BYTES as u64) })
}

/// Store → channel → hour files in order.
pub type SevStores = BTreeMap<String, BTreeMap<u16, Vec<SevFile>>>;

pub fn group(paths: &[PathBuf], warnings: &mut Vec<String>) -> SevStores {
    let mut out: SevStores = BTreeMap::new();
    for p in paths {
        match read_file(p) {
            Ok(f) => out.entry(f.store.clone()).or_default().entry(f.channel).or_default().push(f),
            Err(e) => warnings.push(format!("SEV file skipped: {e}")),
        }
    }
    for channels in out.values_mut() {
        for files in channels.values_mut() {
            files.sort_by_key(|f| f.hour);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_names() {
        assert_eq!(parse_name("Subject1-211209-130128_RSn1_Ch3"), (Some("RSn1".into()), Some(3), 0));
        assert_eq!(parse_name("tank_block_Wav1_ch12-2h"), (Some("Wav1".into()), Some(12), 2));
        assert_eq!(parse_name("nonsense"), (None, None, 0));
    }
}
