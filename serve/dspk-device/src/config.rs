//! `device.toml`: what a device is, where it listens, its sources, where it records.
//!
//! ```toml
//! name = "bench-generator"
//! data = "/data"                     # recordings, the session logs, the device's authority
//! listen = "127.0.0.1:50060"         # QUIC (UDP); clients pair, then connect with a certificate
//! names = ["localhost"]              # names in the device's certificate
//!
//! [ring]
//! seconds = 30
//! max_mb = 256
//!
//! [recording]
//! chunk_seconds = 1.0
//!
//! [[sources]]
//! name = "gen"
//! kind = "generator"
//! channels = 32
//! sample_rate_hz = 30000
//! format = "i16"
//! seed = 1
//!
//! [[sources]]                        # the minimal sensor format
//! name = "ppg"
//! kind = "sensor"
//! transport = { tcp = "127.0.0.1:50070" }
//! channels = ["red", "ir"]
//! sample_rate_hz = 400
//! format = "i32"
//! unit = "Dimensionless"
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use dspk_core::{ChannelInfo, DspError, DspResult, SampleFormat, SampleRate, SignalUnit};
use dspk_io::live::sensor::SensorTransport;
use dspk_io::live::{GeneratorParams, LiveLayout};
use serde::{Deserialize, Serialize};

/// Default QUIC port of a device.
pub const DEFAULT_PORT: u16 = 50060;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConfig {
    pub name: String,
    /// Recordings, session logs, the device's id and authority.
    pub data: PathBuf,
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// Names in the device's certificate (what clients connect to).
    #[serde(default = "default_names")]
    pub names: Vec<String>,
    /// `wgpu`, `cuda`, `cpu`; default: `DSPK_RUNTIME`, else the first compiled in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(default)]
    pub ring: RingConfig,
    #[serde(default)]
    pub recording: RecordingConfig,
    pub sources: Vec<SourceConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RingConfig {
    pub seconds: f64,
    pub max_mb: u64,
}

impl Default for RingConfig {
    fn default() -> Self {
        Self { seconds: 30.0, max_mb: 256 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingConfig {
    /// Length of a Zarr chunk (s): what a crash can lose at most, and the disk's write rhythm.
    pub chunk_seconds: f64,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self { chunk_seconds: 1.0 }
    }
}

/// One source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceConfig {
    /// Noise, spikes and a sine, paced at the sample rate.
    Generator {
        name: String,
        #[serde(flatten)]
        params: GeneratorParams,
    },
    /// Packets of the minimal sensor format.
    Sensor {
        name: String,
        transport: SensorTransport,
        /// Packets over the network are not authenticated: refused off loopback unless set.
        #[serde(default)]
        allow_remote: bool,
        /// Channel names (their number is the packets' channel count).
        channels: Vec<String>,
        sample_rate_hz: f64,
        format: SampleFormat,
        /// Unit per stored step, and the unit.
        #[serde(default = "one")]
        gain: f32,
        #[serde(default)]
        unit: SignalUnit,
    },
    /// A recording replayed at its rate.
    Replay {
        name: String,
        path: PathBuf,
        #[serde(default = "default_true")]
        looped: bool,
        #[serde(default = "default_block_ms")]
        block_ms: f64,
    },
}

fn default_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT))
}
fn default_names() -> Vec<String> {
    vec!["localhost".into()]
}
fn one() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_block_ms() -> f64 {
    1.0
}

impl SourceConfig {
    pub fn name(&self) -> &str {
        match self {
            SourceConfig::Generator { name, .. } | SourceConfig::Sensor { name, .. } | SourceConfig::Replay { name, .. } => name,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            SourceConfig::Generator { .. } => "generator",
            SourceConfig::Sensor { .. } => "sensor",
            SourceConfig::Replay { .. } => "replay",
        }
    }

    /// The layout a sensor's packets must have (others learn theirs when opened).
    pub fn sensor_layout(&self) -> Option<DspResult<LiveLayout>> {
        let SourceConfig::Sensor { name, channels, sample_rate_hz, format, gain, unit, .. } = self else { return None };
        Some((|| {
            if dspk_io::live::sensor::sample_type_code(*format).is_none() {
                return Err(DspError::InvalidConfig(format!("sensor {name}: format {} (int16, int32 or float32)", format.name())));
            }
            let channels = channels.iter().map(|n| ChannelInfo { name: n.clone(), gain: *gain, offset: 0.0, unit: unit.clone() }).collect();
            Ok(LiveLayout { name: name.clone(), channels, sample_rate: SampleRate::new(*sample_rate_hz)?, format: *format })
        })())
    }
}

impl DeviceConfig {
    pub fn from_file(path: &Path) -> DspResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        let mut config: Self = toml::from_str(&text).map_err(|e| DspError::InvalidConfig(format!("{}: {e}", path.display())))?;
        if config.data.is_relative()
            && let Some(dir) = path.parent()
        {
            config.data = dir.join(&config.data);
        }
        config.check()?;
        Ok(config)
    }

    /// A device with one generator: `dspk-device --generator` and tests.
    pub fn generator(data: PathBuf, listen: SocketAddr, params: GeneratorParams) -> Self {
        Self {
            name: "generator".into(),
            data,
            listen,
            names: default_names(),
            runtime: None,
            ring: RingConfig::default(),
            recording: RecordingConfig::default(),
            sources: vec![SourceConfig::Generator { name: "gen".into(), params }],
        }
    }

    pub fn check(&self) -> DspResult<()> {
        match self.sources.len() {
            0 => return Err(DspError::InvalidConfig("a device needs a source".into())),
            1 => {}
            n => return Err(DspError::InvalidConfig(format!("{n} sources: a device runs one source in this version"))),
        }
        if !(self.ring.seconds > 0.0) || !(self.recording.chunk_seconds > 0.0) {
            return Err(DspError::InvalidConfig("ring.seconds and recording.chunk_seconds must be positive".into()));
        }
        Ok(())
    }

    /// Where recordings go.
    pub fn recordings_dir(&self) -> PathBuf {
        self.data.join("recordings")
    }

    /// The device's authority (who may connect).
    pub fn authority_dir(&self) -> PathBuf {
        self.data.join("authority")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_example() {
        let text = r#"
            name = "bench"
            data = "/tmp/x"
            [[sources]]
            name = "ppg"
            kind = "sensor"
            transport = { tcp = "127.0.0.1:50070" }
            channels = ["red", "ir"]
            sample_rate_hz = 400
            format = "i32"
        "#;
        let c: DeviceConfig = toml::from_str(text).unwrap();
        c.check().unwrap();
        let layout = c.sources[0].sensor_layout().unwrap().unwrap();
        assert_eq!(layout.channel_count(), 2);
        assert_eq!(c.listen.port(), DEFAULT_PORT);

        let g: DeviceConfig = toml::from_str("name = \"g\"\ndata = \"d\"\n[[sources]]\nname = \"gen\"\nkind = \"generator\"\nchannels = 8\nsample_rate_hz = 1000\n").unwrap();
        let SourceConfig::Generator { params, .. } = &g.sources[0] else { panic!() };
        assert_eq!((params.channels, params.sample_rate_hz), (8, 1000.0));
    }
}
