//! Channel map files ([`ChannelMap`] under `[remap]`), TOML or JSON by extension.

use std::path::Path;

use dspk_core::{ChannelMap, ChannelMapFile, DspError, DspResult};

/// Reads a map file: `.toml` or `.json` (the same schema in either).
pub fn read_channel_map(path: &Path) -> DspResult<ChannelMap> {
    let text = std::fs::read_to_string(path).map_err(|e| DspError::InvalidConfig(format!("{}: {e}", path.display())))?;
    let toml = match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("toml") => true,
        Some("json") => false,
        _ => return Err(DspError::InvalidConfig(format!("{}: a channel map is a .toml or .json file", path.display()))),
    };
    parse_channel_map(&text, toml).map_err(|e| DspError::InvalidConfig(format!("{}: {e}", path.display())))
}

/// Parses a map file's text (TOML when `toml`, else JSON) and checks its rules.
pub fn parse_channel_map(text: &str, toml: bool) -> DspResult<ChannelMap> {
    let file: ChannelMapFile = if toml {
        toml::from_str(text).map_err(|e| DspError::InvalidConfig(format!("channel map: {e}")))?
    } else {
        serde_json::from_str(text).map_err(|e| DspError::InvalidConfig(format!("channel map: {e}")))?
    };
    file.remap.order()?;
    Ok(file.remap)
}

/// The map as a TOML file's text.
pub fn channel_map_toml(map: &ChannelMap) -> String {
    toml::to_string(&ChannelMapFile { remap: map.clone() }).expect("a channel map serialises")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_and_json_read_the_same_map() {
        let t = parse_channel_map("[remap]\nbase = 1\npairs = [{ from = 3, to = 1, name = \"A1\" }, { from = 1, to = 2 }]\n", true).unwrap();
        let j = parse_channel_map(r#"{"remap": {"base": 1, "pairs": [{"from": 3, "to": 1, "name": "A1"}, {"from": 1, "to": 2}]}}"#, false).unwrap();
        assert_eq!(t, j);
        assert_eq!(t.order().unwrap(), vec![2, 0]);
        assert_eq!(parse_channel_map(&channel_map_toml(&t), true).unwrap(), t);
        // Default base 0; unknown keys and broken rules refused
        assert_eq!(parse_channel_map("[remap]\npairs = [{ from = 0, to = 0 }]", true).unwrap().base, 0);
        assert!(parse_channel_map("[remap]\npairs = [{ from = 0, to = 0, nme = \"x\" }]", true).is_err());
        assert!(parse_channel_map("[remap]\npairs = [{ from = 0, to = 1 }]", true).is_err());
    }
}
