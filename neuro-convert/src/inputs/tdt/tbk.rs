//! Block notes (`.Tbk`): cp437 text; between the 2nd and 3rd `[USERNOTEDELIMITER]` each store
//! is a run of `NAME=field;TYPE=t;VALUE=v;` lines separated by `[STOREHDRITEM]`.
//!
//! The TSQ is authoritative for what was recorded (the Tbk can list stores that never wrote
//! data); this is used for store settings that the TSQ does not carry.

use std::collections::BTreeMap;

use crate::common::text::decode_single_byte;

/// One store's settings (`StoreName`, `NumChan`, `SampleFreq`, `DataFormat`, …).
pub type TbkStore = BTreeMap<String, String>;

pub fn parse_tbk(bytes: &[u8]) -> Vec<TbkStore> {
    let text = decode_single_byte(bytes);
    let parts: Vec<&str> = text.split("[USERNOTEDELIMITER]").collect();
    let Some(section) = parts.get(2) else { return Vec::new() };

    section
        .split("[STOREHDRITEM]")
        .filter_map(|item| {
            let store: TbkStore = item
                .lines()
                .filter_map(|line| {
                    let mut name = None;
                    let mut value = None;
                    for field in line.split(';') {
                        match field.split_once('=') {
                            Some(("NAME", v)) => name = Some(v.trim().to_string()),
                            Some(("VALUE", v)) => value = Some(v.trim().to_string()),
                            _ => {}
                        }
                    }
                    Some((name?, value.unwrap_or_default()))
                })
                .collect();
            store.contains_key("StoreName").then_some(store)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_items() {
        let text = "hdr[USERNOTEDELIMITER]block[USERNOTEDELIMITER]NAME=StoreName;TYPE=T;VALUE=bpPe;\nNAME=NumChan;TYPE=L;VALUE=1;\n\
NAME=SampleFreq;TYPE=L;VALUE=2034.505249;\n[STOREHDRITEM]NAME=StoreName;TYPE=T;VALUE=Tick;\n[USERNOTEDELIMITER]";
        let s = parse_tbk(text.as_bytes());
        assert_eq!(s.len(), 2);
        assert_eq!(s[0]["StoreName"], "bpPe");
        assert_eq!(s[0]["SampleFreq"], "2034.505249");
        assert_eq!(s[1]["StoreName"], "Tick");
    }
}
