//! A `device.toml` source → a [`LiveSource`].

use std::sync::Arc;

use dspk_core::{DspError, DspResult};
use dspk_io::live::sensor::{SensorListener, SensorSource};
use dspk_io::live::{Generator, LiveSource, Paced};

use crate::config::SourceConfig;

/// Opens `config`'s source (a sensor's listener is bound here).
pub fn open_source(config: &SourceConfig) -> DspResult<Box<dyn LiveSource>> {
    Ok(match config {
        SourceConfig::Generator { name, params } => Box::new(Generator::new(name.clone(), params)?),
        SourceConfig::Sensor { transport, allow_remote, .. } => {
            let layout = config.sensor_layout().expect("a sensor has a layout")?;
            let listener = SensorListener::bind(transport, *allow_remote)?;
            tracing::info!("sensor {} listens on {}", layout.name, listener.label());
            Box::new(SensorSource::new(listener, layout))
        }
        SourceConfig::Replay { name, path, looped, block_ms } => {
            let recording = dspk_io::open(path).map_err(|e| DspError::Io(format!("replay {name}: {}: {e}", path.display())))?;
            let mut paced = Paced::new(Arc::from(recording), *block_ms, *looped);
            paced.rename(name);
            Box::new(paced)
        }
    })
}
