//! Text from legacy Windows acquisition software.

/// Decodes single-byte text (cp437 / Latin-1). ASCII is exact; other bytes map to their
/// Latin-1 code points, which keeps the text readable without a full code page table.
pub fn decode_single_byte(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

/// Reads a text file, accepting UTF-8 or single-byte encodings; `None` when missing.
pub fn read_text(path: &std::path::Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8(bytes.clone()).unwrap_or_else(|_| decode_single_byte(&bytes)))
}
