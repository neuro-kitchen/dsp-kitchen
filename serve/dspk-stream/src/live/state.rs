//! The device's state machine (Synapse's states):
//! `initializing → stopped ⇄ running`, `initializing → error` when the device fails to start.
//! Configure only when stopped (back to stopped whether it succeeds or not); start needs a chain;
//! a chain failing during a run ends the run (stopped, with the reason), it is not an error state.

use super::wire::DeviceState;
use super::{LiveError, LiveResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lifecycle {
    state: DeviceState,
    configured: bool,
    /// The last failure or result, in words.
    message: String,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self { state: DeviceState::Initializing, configured: false, message: String::new() }
    }
}

impl Lifecycle {
    pub fn state(&self) -> DeviceState {
        self.state
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn configured(&self) -> bool {
        self.configured
    }

    /// The device started (sources open, ready).
    pub fn ready(&mut self) {
        if self.state == DeviceState::Initializing {
            self.state = DeviceState::Stopped;
        }
    }

    /// The device failed to start: only a restart leaves this state.
    pub fn failed(&mut self, why: impl Into<String>) {
        self.state = DeviceState::Error;
        self.message = why.into();
    }

    /// Before checking a chain: only when stopped.
    pub fn begin_configure(&mut self) -> LiveResult<()> {
        self.expect(DeviceState::Stopped, "configure")?;
        self.state = DeviceState::Initializing;
        Ok(())
    }

    /// After checking it: back to stopped either way.
    pub fn end_configure(&mut self, accepted: bool, message: impl Into<String>) {
        self.state = DeviceState::Stopped;
        if accepted {
            self.configured = true;
        }
        self.message = message.into();
    }

    pub fn start(&mut self) -> LiveResult<()> {
        self.expect(DeviceState::Stopped, "start")?;
        if !self.configured {
            return Err(LiveError::wrong_state("no chain is configured: configure one first"));
        }
        self.state = DeviceState::Running;
        self.message.clear();
        Ok(())
    }

    /// The run ended: asked to (`message` says by whom) or because the chain failed.
    pub fn stop(&mut self, message: impl Into<String>) -> LiveResult<()> {
        self.expect(DeviceState::Running, "stop")?;
        self.state = DeviceState::Stopped;
        self.message = message.into();
        Ok(())
    }

    fn expect(&self, state: DeviceState, what: &str) -> LiveResult<()> {
        if self.state == state {
            return Ok(());
        }
        let now = match self.state {
            DeviceState::Running => "running: stop it first",
            DeviceState::Stopped => "stopped",
            DeviceState::Initializing => "busy (starting up or checking a chain)",
            DeviceState::Error => "in error: it needs a restart",
            DeviceState::Unspecified => "in an unknown state",
        };
        Err(LiveError::wrong_state(format!("cannot {what}: the device is {now}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions() {
        let mut l = Lifecycle::default();
        assert!(l.begin_configure().is_err(), "still starting up");
        l.ready();
        assert!(l.start().is_err(), "no chain");
        l.begin_configure().unwrap();
        assert_eq!(l.state(), DeviceState::Initializing);
        l.end_configure(false, "refused");
        assert_eq!(l.state(), DeviceState::Stopped);
        assert!(l.start().is_err(), "the refused chain was not taken");
        l.begin_configure().unwrap();
        l.end_configure(true, "");
        l.start().unwrap();
        assert!(l.begin_configure().is_err(), "configure while running");
        l.stop("by a").unwrap();
        assert!(l.stop("again").is_err());
        l.failed("no source");
        assert!(l.start().is_err());
    }
}
