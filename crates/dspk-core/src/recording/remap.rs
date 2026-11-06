//! Channel maps: which input channel becomes which output channel, written as pairs (like a
//! hardware channel mapper), and a recording read through one.
//!
//! A map file (TOML or JSON, one schema):
//!
//! ```toml
//! [remap]
//! base = 0            # 1 for 1-based numbers (TDT, MATLAB)
//! pairs = [
//!   { from = 4, to = 0, name = "A1" },
//!   { from = 2, to = 1, name = "A2" },
//!   { from = 7, to = 2 },
//! ]
//! ```
//!
//! Every output channel `0..n` gets exactly one input; inputs left out are dropped; the same input
//! twice is refused; names are optional (an unnamed output keeps its input's name).

use std::ops::Range;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::info::RecordingInfo;
use super::source::{check_read, check_read_stored, RecordingSource};
use crate::error::{DspError, DspResult};

/// Default of [`ChannelMap::base`]: channel numbers start at 0.
pub const DEFAULT_CHANNEL_BASE: usize = 0;

/// One output channel: input channel `from` becomes output channel `to`, optionally named.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ChannelPair {
    pub from: usize,
    pub to: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Input channel → output channel pairs (see the module docs for the rules).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ChannelMap {
    /// The number of the first channel in `pairs` (0, or 1 for 1-based maps).
    #[serde(default = "default_base")]
    pub base: usize,
    pub pairs: Vec<ChannelPair>,
}

fn default_base() -> usize {
    DEFAULT_CHANNEL_BASE
}

/// A map file: the map under `[remap]` (TOML) or `"remap"` (JSON).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ChannelMapFile {
    pub remap: ChannelMap,
}

impl ChannelMap {
    /// Output channel `i` reads input `order[i]` (0-based), with no names.
    pub fn from_order(order: &[usize]) -> Self {
        let pairs = order.iter().enumerate().map(|(to, &from)| ChannelPair { from, to, name: None }).collect();
        Self { base: 0, pairs }
    }

    /// The input channel (0-based) of every output channel, in output order. Checks the rules:
    /// numbers at or above `base`, outputs `0..n` each exactly once, no input twice.
    pub fn order(&self) -> DspResult<Vec<usize>> {
        let bad = |m: String| Err(DspError::InvalidConfig(format!("channel map: {m}")));
        if self.pairs.is_empty() {
            return bad("no pairs".into());
        }
        let n = self.pairs.len();
        let mut order = vec![usize::MAX; n];
        let mut inputs = std::collections::HashSet::with_capacity(n);
        for p in &self.pairs {
            let (Some(from), Some(to)) = (p.from.checked_sub(self.base), p.to.checked_sub(self.base)) else {
                return bad(format!("pair {} → {} is below base {}", p.from, p.to, self.base));
            };
            if to >= n {
                return bad(format!("output {} is past the last output ({} pairs give outputs {}..{})", p.to, n, self.base, self.base + n - 1));
            }
            if order[to] != usize::MAX {
                return bad(format!("output {} is given twice", p.to));
            }
            if !inputs.insert(from) {
                return bad(format!("input {} is used twice", p.from));
            }
            order[to] = from;
        }
        Ok(order)
    }

    /// [`Self::order`], with every input checked against a recording of `channels` channels.
    pub fn resolve(&self, channels: usize) -> DspResult<Vec<usize>> {
        let order = self.order()?;
        if let Some(&c) = order.iter().find(|&&c| c >= channels) {
            return Err(DspError::InvalidConfig(format!("channel map: input {} does not exist (the recording has {channels} channels)", c + self.base)));
        }
        Ok(order)
    }

    /// The name of every output channel given one (by output, 0-based).
    pub fn names(&self) -> Vec<Option<String>> {
        let mut names = vec![None; self.pairs.len()];
        for p in &self.pairs {
            if let Some(slot) = p.to.checked_sub(self.base).and_then(|to| names.get_mut(to)) {
                slot.clone_from(&p.name);
            }
        }
        names
    }

    /// This map, then `next` (whose inputs are this map's outputs): one map from this map's inputs
    /// to `next`'s outputs, 0-based. A name from `next` wins over this map's.
    pub fn then(&self, next: &ChannelMap) -> DspResult<ChannelMap> {
        let (first, second) = (self.order()?, next.order()?);
        let first_names = self.names();
        let second_names = next.names();
        let mut pairs = Vec::with_capacity(second.len());
        for (to, &mid) in second.iter().enumerate() {
            let Some(&from) = first.get(mid) else {
                return Err(DspError::InvalidConfig(format!("channel map: input {} does not exist (the map before gives {} channels)", mid + next.base, first.len())));
            };
            pairs.push(ChannelPair { from, to, name: second_names[to].clone().or_else(|| first_names[mid].clone()) });
        }
        Ok(ChannelMap { base: 0, pairs })
    }
}

/// A recording read through a [`ChannelMap`]: channel `i` is the parent's `order[i]`, with the
/// map's name when it gives one. Reads go straight to the parent in the new order (no copy).
pub struct RemappedSource {
    parent: Arc<dyn RecordingSource>,
    order: Vec<usize>,
    info: RecordingInfo,
}

impl RemappedSource {
    pub fn new(parent: Arc<dyn RecordingSource>, map: &ChannelMap) -> DspResult<Self> {
        let p = parent.info();
        let order = map.resolve(p.channel_count())?;
        let mut info = p.clone();
        info.channels = order
            .iter()
            .zip(map.names())
            .map(|(&c, name)| {
                let mut ch = p.channels[c].clone();
                if let Some(name) = name {
                    ch.name = name;
                }
                ch
            })
            .collect();
        Ok(Self { parent, order, info })
    }

    /// The parent channel of every channel (0-based).
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    fn parent_channels(&self, channels: &[usize]) -> Vec<usize> {
        channels.iter().map(|&c| self.order[c]).collect()
    }
}

impl RecordingSource for RemappedSource {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn chunk_samples(&self) -> Option<u64> {
        self.parent.chunk_samples()
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        check_read(&self.info, channels, &samples, out.len())?;
        self.parent.read(&self.parent_channels(channels), samples, out)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        check_read_stored(&self.info, channels, &samples, out.len())?;
        self.parent.read_stored(&self.parent_channels(channels), samples, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::MemoryRecording;

    fn pair(from: usize, to: usize, name: Option<&str>) -> ChannelPair {
        ChannelPair { from, to, name: name.map(str::to_string) }
    }

    #[test]
    fn order_follows_the_pairs_and_checks_the_rules() {
        let map = ChannelMap { base: 1, pairs: vec![pair(3, 2, None), pair(1, 1, Some("A1"))] };
        assert_eq!(map.order().unwrap(), vec![0, 2]);
        assert_eq!(map.names(), vec![Some("A1".to_string()), None]);
        assert!(map.resolve(2).is_err(), "input 3 of a 2-channel recording");
        // An output twice, an input twice, a gap, below base, empty
        for pairs in [
            vec![pair(0, 0, None), pair(1, 0, None)],
            vec![pair(0, 0, None), pair(0, 1, None)],
            vec![pair(0, 0, None), pair(1, 2, None)],
            vec![],
        ] {
            assert!(ChannelMap { base: 0, pairs }.order().is_err());
        }
        assert!(ChannelMap { base: 1, pairs: vec![pair(0, 1, None)] }.order().is_err());
    }

    #[test]
    fn maps_compose() {
        let a = ChannelMap { base: 0, pairs: vec![pair(5, 0, Some("x")), pair(3, 1, Some("y")), pair(1, 2, None)] };
        let b = ChannelMap { base: 0, pairs: vec![pair(2, 0, Some("z")), pair(0, 1, None)] };
        let ab = a.then(&b).unwrap();
        assert_eq!(ab.order().unwrap(), vec![1, 5]);
        assert_eq!(ab.names(), vec![Some("z".to_string()), Some("x".to_string())]);
    }

    #[test]
    fn a_remapped_source_reads_in_the_new_order() {
        // 3 channels × 4 samples: value = channel · 10 + sample
        let data: Vec<f32> = (0..3).flat_map(|c| (0..4).map(move |s| (c * 10 + s) as f32)).collect();
        let parent: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("m", data, 3, 1000.0).unwrap());
        let map = ChannelMap { base: 0, pairs: vec![pair(2, 0, Some("A1")), pair(0, 1, None)] };
        let r = RemappedSource::new(parent, &map).unwrap();
        assert_eq!(r.info().channel_count(), 2);
        assert_eq!(r.info().channels[0].name, "A1");
        assert_eq!(r.info().channels[1].name, "ch0");
        let mut out = vec![0.0; 4];
        r.read(&[1, 0], 1..3, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 21.0, 22.0]);
    }
}
