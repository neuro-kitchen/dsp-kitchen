//! Probe files that say which recording channel each contact is wired to, read as a
//! [`ChannelMap`] (the data in contact order) plus the geometry in that order, and a probe's
//! geometry following a map.
//!
//! - **probeinterface** JSON (`"specification": "probeinterface"`): contact `i` at
//!   `contact_positions[i]`, wired to recording channel `device_channel_indices[i]` (`-1`: not
//!   connected, left out); `contact_ids` name the outputs; several probes are concatenated.
//! - **Kilosort** probe JSON (Kilosort 4: `chanMap`, `xc`, `yc`, `kcoords`, 0-based): site `i`
//!   at `(xc[i], yc[i])` on shank `kcoords[i]` reads data row `chanMap[i]`; sites with
//!   `connected[i] = false` (when given) are left out.
//!
//! Only `.json` files are read; MATLAB `.mat` probe files are refused (converting them is
//! neuro-converter's job, not dspk's).

use std::collections::BTreeMap;
use std::path::Path;

use dspk_core::{ChannelMap, ChannelPair, DspError, DspResult};
use serde::Deserialize;
use serde_json::Value;

use super::{Position3D, SensorLayout, SensorSite};

/// A probe file read as a map and its geometry: output channel `i` reads recording channel
/// `map.order()[i]` and sits at `layout`'s site `i` (channel ids `0..n`).
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeWiring {
    pub map: ChannelMap,
    pub layout: SensorLayout,
}

fn bad(path: &str, m: impl std::fmt::Display) -> DspError {
    DspError::InvalidConfig(format!("{path}: {m}"))
}

/// Reads a probeinterface or Kilosort probe JSON file (detected from its keys); any other
/// extension (e.g. a MATLAB `.mat`) is refused.
pub fn read_probe_wiring(path: &Path) -> DspResult<ProbeWiring> {
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
        return Err(bad(&path.display().to_string(), "a probe file is a .json file (probeinterface or Kilosort 4); .mat files are not read (neuro-converter converts them)"));
    }
    let text = std::fs::read_to_string(path).map_err(|e| bad(&path.display().to_string(), e))?;
    let name = path.file_stem().map_or_else(|| "probe".into(), |s| s.to_string_lossy().into_owned());
    parse_probe_wiring(&text, &name).map_err(|e| match e {
        DspError::InvalidConfig(m) => bad(&path.display().to_string(), m),
        other => other,
    })
}

/// [`read_probe_wiring`] of a file's text; `name` names the layout.
pub fn parse_probe_wiring(text: &str, name: &str) -> DspResult<ProbeWiring> {
    let v: Value = serde_json::from_str(text).map_err(|e| bad(name, e))?;
    if v.get("specification").and_then(Value::as_str) == Some("probeinterface") {
        probeinterface(v, name)
    } else if v.get("chanMap").is_some() {
        kilosort(v, name)
    } else {
        Err(bad(name, "neither a probeinterface file (\"specification\": \"probeinterface\") nor a Kilosort probe (\"chanMap\")"))
    }
}

/// One wired contact: recording channel, position, shank, name.
struct Contact {
    channel: usize,
    position: Position3D,
    shank: usize,
    name: Option<String>,
}

fn wiring(contacts: Vec<Contact>, name: &str) -> DspResult<ProbeWiring> {
    if contacts.is_empty() {
        return Err(bad(name, "no connected contacts"));
    }
    let pairs = contacts.iter().enumerate().map(|(to, c)| ChannelPair { from: c.channel, to, name: c.name.clone() }).collect();
    let map = ChannelMap { base: 0, pairs };
    map.order().map_err(|e| bad(name, e))?;
    let sites = contacts.into_iter().enumerate().map(|(i, c)| SensorSite::new(i, c.position, c.shank)).collect();
    Ok(ProbeWiring { map, layout: SensorLayout::new(name, sites) })
}

fn probeinterface(v: Value, name: &str) -> DspResult<ProbeWiring> {
    #[derive(Deserialize)]
    struct Probe {
        contact_positions: Vec<Vec<f32>>,
        #[serde(default)]
        device_channel_indices: Option<Vec<i64>>,
        #[serde(default)]
        shank_ids: Option<Vec<String>>,
        #[serde(default)]
        contact_ids: Option<Vec<String>>,
    }
    #[derive(Deserialize)]
    struct File {
        probes: Vec<Probe>,
    }
    let file: File = serde_json::from_value(v).map_err(|e| bad(name, e))?;
    // Shanks numbered across probes in order of appearance
    let mut shanks: BTreeMap<(usize, String), usize> = BTreeMap::new();
    let mut contacts = Vec::new();
    for (p, probe) in file.probes.iter().enumerate() {
        let n = probe.contact_positions.len();
        let Some(dci) = &probe.device_channel_indices else {
            return Err(bad(name, format!("probe {p} has no device_channel_indices (which channel each contact is wired to)")));
        };
        if dci.len() != n {
            return Err(bad(name, format!("probe {p}: {} device_channel_indices for {n} contacts", dci.len())));
        }
        for (i, pos) in probe.contact_positions.iter().enumerate() {
            let Ok(channel) = usize::try_from(dci[i]) else { continue };
            let shank_key = (p, probe.shank_ids.as_ref().and_then(|s| s.get(i)).cloned().unwrap_or_default());
            let next = shanks.len();
            let shank = *shanks.entry(shank_key).or_insert(next);
            let at = |k: usize| pos.get(k).copied().unwrap_or(0.0);
            let name = probe.contact_ids.as_ref().and_then(|c| c.get(i)).filter(|c| !c.is_empty()).cloned();
            contacts.push(Contact { channel, position: Position3D::new(at(0), at(1), at(2)), shank, name });
        }
    }
    wiring(contacts, name)
}

fn kilosort(v: Value, name: &str) -> DspResult<ProbeWiring> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Probe {
        chan_map: Vec<f64>,
        #[serde(rename = "xc")]
        xc: Vec<f32>,
        #[serde(rename = "yc")]
        yc: Vec<f32>,
        #[serde(default)]
        kcoords: Option<Vec<f64>>,
        #[serde(default)]
        connected: Option<Vec<Value>>,
    }
    let probe: Probe = serde_json::from_value(v).map_err(|e| bad(name, e))?;
    let n = probe.chan_map.len();
    if probe.xc.len() != n || probe.yc.len() != n {
        return Err(bad(name, format!("chanMap has {n} sites but xc / yc have {} / {}", probe.xc.len(), probe.yc.len())));
    }
    let connected = |i: usize| match probe.connected.as_ref().and_then(|c| c.get(i)) {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(x)) => x.as_f64() != Some(0.0),
        _ => true,
    };
    let mut contacts = Vec::new();
    for i in (0..n).filter(|&i| connected(i)) {
        let c = probe.chan_map[i];
        if c < 0.0 || c.fract() != 0.0 {
            return Err(bad(name, format!("chanMap[{i}] = {c} is not a channel index (0-based)")));
        }
        let shank = probe.kcoords.as_ref().and_then(|k| k.get(i)).map_or(0, |&k| k.max(0.0) as usize);
        contacts.push(Contact { channel: c as usize, position: Position3D::new(probe.xc[i], probe.yc[i], 0.0), shank, name: None });
    }
    wiring(contacts, name)
}

impl SensorLayout {
    /// The geometry of the channels after `map` (whose inputs are this layout's channel ids):
    /// output channel `i` sits where input `order[i]` did, renamed `i`.
    pub fn remap(&self, map: &ChannelMap) -> DspResult<SensorLayout> {
        let order = map.resolve(self.total_channels())?;
        for &c in &order {
            self.get_site(c)?;
        }
        Ok(self.select_channels(&order))
    }

    /// A map that sorts the channels by shank, then depth (`y`, from the tip up), then `x`:
    /// neighbouring rows of the data become neighbouring sites.
    pub fn depth_order(&self) -> ChannelMap {
        let mut sites: Vec<&SensorSite> = self.contacts.iter().collect();
        sites.sort_by(|a, b| {
            (a.shank_id, a.position.y_um, a.position.x_um)
                .partial_cmp(&(b.shank_id, b.position.y_um, b.position.x_um))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ChannelMap::from_order(&sites.iter().map(|s| s.channel_id).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBEINTERFACE: &str = r#"{"specification": "probeinterface", "version": "0.2.21",
        "probes": [{"ndim": 2, "si_units": "um",
                    "contact_positions": [[0, 0], [0, 20], [16, 10], [16, 30]],
                    "device_channel_indices": [3, 1, -1, 0],
                    "shank_ids": ["", "", "", ""],
                    "contact_ids": ["e0", "e1", "e2", "e3"]}]}"#;

    #[test]
    fn probeinterface_contacts_in_order_unconnected_left_out() {
        let w = parse_probe_wiring(PROBEINTERFACE, "p").unwrap();
        assert_eq!(w.map.order().unwrap(), vec![3, 1, 0]);
        assert_eq!(w.map.names(), vec![Some("e0".into()), Some("e1".into()), Some("e3".into())]);
        let ys: Vec<f32> = w.layout.contacts.iter().map(|s| s.position.y_um).collect();
        assert_eq!(ys, vec![0.0, 20.0, 30.0]);
        assert_eq!(w.layout.get_site(2).unwrap().position.x_um, 16.0);
    }

    #[test]
    fn kilosort_probe_json() {
        let ks = r#"{"chanMap": [2, 0, 1], "xc": [0, 32, 0], "yc": [0, 0, 15], "kcoords": [0, 0, 1], "n_chan": 3}"#;
        let w = parse_probe_wiring(ks, "ks").unwrap();
        assert_eq!(w.map.order().unwrap(), vec![2, 0, 1]);
        assert_eq!(w.layout.get_site(2).unwrap().shank_id, 1);
        let unwired = r#"{"chanMap": [0, 1], "xc": [0, 0], "yc": [0, 20], "connected": [true, false]}"#;
        assert_eq!(parse_probe_wiring(unwired, "ks").unwrap().map.order().unwrap(), vec![0]);
        assert!(parse_probe_wiring(r#"{"chanMap": [0, 0], "xc": [0, 0], "yc": [0, 1]}"#, "ks").is_err(), "a channel twice");
        assert!(parse_probe_wiring(r#"{"x": 1}"#, "?").is_err());
    }

    #[test]
    fn only_json_files_are_read() {
        let dir = std::env::temp_dir().join(format!("dspk-wiring-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mat = dir.join("chanMap.mat");
        std::fs::write(&mat, b"MATLAB 5.0 MAT-file").unwrap();
        let err = read_probe_wiring(&mat).unwrap_err().to_string();
        assert!(err.contains(".json"), "{err}");
        let json = dir.join("probe.json");
        std::fs::write(&json, PROBEINTERFACE).unwrap();
        assert_eq!(read_probe_wiring(&json).unwrap().layout.name, "probe");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn geometry_follows_a_map_and_depth_order_sorts_by_shank_and_depth() {
        let np = crate::io::probe::neuropixels_1_0();
        let map = ChannelMap::from_order(&[10, 3, 200]);
        assert_eq!(np.remap(&map).unwrap(), np.select_channels(&[10, 3, 200]));
        assert!(np.remap(&ChannelMap::from_order(&[384])).is_err());
        // Shuffled sites come back sorted by depth
        let shuffled = np.select_channels(&[7, 2, 5, 0]);
        let order = shuffled.depth_order().order().unwrap();
        let ys: Vec<f32> = order.iter().map(|&c| shuffled.get_site(c).unwrap().position.y_um).collect();
        assert!(ys.windows(2).all(|w| w[0] <= w[1]), "{ys:?}");
    }
}
