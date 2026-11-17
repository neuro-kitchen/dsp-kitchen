//! How a tap's frames travel, named by its endpoint's URI scheme (`quic://host:port/tap`; later
//! `zmq+tcp://…`). The frames, the chain and the rules are the same over every transport: a
//! transport only says where a tap is reached and how its endpoint reads. QUIC is the device's
//! own (its taps' `endpoint` may stay empty: the device's address); another transport (gRPC +
//! ZeroMQ, Synapse-style) is a [`TapTransport`] of its own, registered by whoever adds it.

use std::sync::{Arc, RwLock};

/// One transport of taps.
pub trait TapTransport: Send + Sync {
    /// Its URI scheme (`quic`).
    fn scheme(&self) -> &'static str;

    /// The endpoint of `tap` on a device at `address` (`host:port`).
    fn endpoint(&self, address: &str, tap: &str) -> String {
        format!("{}://{address}/{tap}", self.scheme())
    }
}

/// The device's own transport: QUIC + TLS, the live protocol's streams.
pub struct Quic;

impl TapTransport for Quic {
    fn scheme(&self) -> &'static str {
        "quic"
    }
}

/// An endpoint split: scheme, `host:port`, tap. `None` for what is not `scheme://address/tap`.
pub fn parse_endpoint(endpoint: &str) -> Option<(&str, &str, &str)> {
    let (scheme, rest) = endpoint.split_once("://")?;
    let (address, tap) = rest.split_once('/')?;
    (!scheme.is_empty() && !address.is_empty() && !tap.is_empty()).then_some((scheme, address, tap))
}

fn registry() -> &'static RwLock<Vec<Arc<dyn TapTransport>>> {
    static ALL: std::sync::OnceLock<RwLock<Vec<Arc<dyn TapTransport>>>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| RwLock::new(vec![Arc::new(Quic)]))
}

/// Adds a transport (a crate that brings one registers it at start).
pub fn register(transport: Arc<dyn TapTransport>) {
    registry().write().unwrap_or_else(|e| e.into_inner()).push(transport);
}

/// The transport of `endpoint` (an empty endpoint is the device's QUIC); `None`: none registered
/// speaks its scheme.
pub fn transport_for(endpoint: &str) -> Option<Arc<dyn TapTransport>> {
    let scheme = if endpoint.is_empty() { "quic" } else { parse_endpoint(endpoint)?.0 };
    registry().read().unwrap_or_else(|e| e.into_inner()).iter().find(|t| t.scheme() == scheme).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Zmq;
    impl TapTransport for Zmq {
        fn scheme(&self) -> &'static str {
            "zmq+tcp"
        }
    }

    /// Endpoints read by scheme; QUIC is there from the start, another transport once registered.
    #[test]
    fn endpoints_name_their_transport() {
        assert_eq!(Quic.endpoint("10.0.0.2:50060", "spikes"), "quic://10.0.0.2:50060/spikes");
        assert_eq!(parse_endpoint("quic://10.0.0.2:50060/spikes"), Some(("quic", "10.0.0.2:50060", "spikes")));
        assert_eq!(parse_endpoint("spikes"), None);
        assert_eq!(transport_for("").unwrap().scheme(), "quic");
        assert!(transport_for("zmq+tcp://h:5555/raw").is_none());
        register(Arc::new(Zmq));
        assert_eq!(transport_for("zmq+tcp://h:5555/raw").unwrap().scheme(), "zmq+tcp");
    }
}
