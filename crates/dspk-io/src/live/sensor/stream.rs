//! Packets of the minimal sensor format arriving over TCP, UDP or a Unix socket.
//!
//! The format has no authentication, so a listener on anything but loopback (or a Unix socket) is
//! refused unless the configuration allows it explicitly ([`SensorListener::bind`]'s
//! `allow_remote`), and then it is logged as a warning. A serial port (the Arduino) comes with
//! feature `serial`.

use std::io::{ErrorKind, Read};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use dspk_core::{DspError, DspResult};
use serde::{Deserialize, Serialize};

use super::PacketReader;
use crate::live::{LiveEvent, LiveLayout, LiveSource, SourceCounters};

/// Pause between accept attempts while no sensor is connected.
const ACCEPT_POLL: Duration = Duration::from_millis(20);
/// Largest datagram (bytes).
const MAX_DATAGRAM: usize = 65_536;

/// Where a sensor's packets arrive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SensorTransport {
    /// A TCP listener; one sensor connected at a time (a reconnecting sensor replaces the last).
    Tcp(SocketAddr),
    /// UDP datagrams, one or more packets each.
    Udp(SocketAddr),
    /// A Unix socket (a driver program in the same container).
    Unix(PathBuf),
}

enum Listener {
    Tcp(TcpListener),
    Udp(UdpSocket),
    #[cfg(unix)]
    Unix(UnixListener, PathBuf),
}

enum Connection {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

impl Connection {
    fn read_for(&mut self, buf: &mut [u8], wait: Duration) -> std::io::Result<usize> {
        let wait = Some(wait.max(Duration::from_millis(1)));
        match self {
            Connection::Tcp(s) => {
                s.set_read_timeout(wait)?;
                s.read(buf)
            }
            #[cfg(unix)]
            Connection::Unix(s) => {
                s.set_read_timeout(wait)?;
                s.read(buf)
            }
        }
    }
}

/// A bound endpoint, before any sensor speaks.
pub struct SensorListener {
    listener: Listener,
    label: String,
}

impl SensorListener {
    /// Binds `transport`. Network addresses that are not loopback are refused unless
    /// `allow_remote`: the format has no authentication.
    pub fn bind(transport: &SensorTransport, allow_remote: bool) -> DspResult<Self> {
        let io = |what: &str| {
            let what = what.to_string();
            move |e: std::io::Error| DspError::Io(format!("{what}: {e}"))
        };
        let check = |addr: &SocketAddr| {
            if addr.ip().is_loopback() || allow_remote {
                if !addr.ip().is_loopback() {
                    tracing::warn!("sensor packets accepted from the network on {addr}: they are not authenticated");
                }
                Ok(())
            } else {
                Err(DspError::InvalidConfig(format!(
                    "{addr} is not loopback: sensor packets are not authenticated; use a Unix socket, a serial port or 127.0.0.1, or allow it explicitly (allow_remote = true)"
                )))
            }
        };
        let (listener, label) = match transport {
            SensorTransport::Tcp(addr) => {
                check(addr)?;
                let l = TcpListener::bind(addr).map_err(io(&format!("tcp {addr}")))?;
                l.set_nonblocking(true).map_err(io("tcp"))?;
                (Listener::Tcp(l), format!("tcp://{addr}"))
            }
            SensorTransport::Udp(addr) => {
                check(addr)?;
                (Listener::Udp(UdpSocket::bind(addr).map_err(io(&format!("udp {addr}")))?), format!("udp://{addr}"))
            }
            #[cfg(unix)]
            SensorTransport::Unix(path) => {
                // A socket file left by an earlier run
                if path.exists() {
                    std::fs::remove_file(path).map_err(io(&path.display().to_string()))?;
                }
                let l = UnixListener::bind(path).map_err(io(&path.display().to_string()))?;
                l.set_nonblocking(true).map_err(io("unix"))?;
                (Listener::Unix(l, path.clone()), format!("unix://{}", path.display()))
            }
            #[cfg(not(unix))]
            SensorTransport::Unix(path) => return Err(DspError::UnsupportedFormat(format!("{}: Unix sockets on this system", path.display()))),
        };
        Ok(Self { listener, label })
    }

    /// Where it listens (for logs and the device's status).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The local address (TCP / UDP; useful with port 0 in tests).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        match &self.listener {
            Listener::Tcp(l) => l.local_addr().ok(),
            Listener::Udp(s) => s.local_addr().ok(),
            #[cfg(unix)]
            Listener::Unix(..) => None,
        }
    }
}

impl Drop for SensorListener {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Listener::Unix(_, path) = &self.listener {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A live source fed by sensor packets.
pub struct SensorSource {
    listener: SensorListener,
    connection: Option<Connection>,
    reader: PacketReader,
    buf: Vec<u8>,
}

impl SensorSource {
    pub fn new(listener: SensorListener, layout: LiveLayout) -> Self {
        Self { listener, connection: None, reader: PacketReader::new(layout), buf: vec![0; MAX_DATAGRAM] }
    }

    pub fn label(&self) -> &str {
        self.listener.label()
    }

    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accepts a waiting sensor (a new one replaces the current one).
    fn accept(&mut self) -> DspResult<bool> {
        let accepted = match &self.listener.listener {
            Listener::Tcp(l) => match l.accept() {
                Ok((s, peer)) => {
                    s.set_nonblocking(false).map_err(|e| DspError::Io(e.to_string()))?;
                    tracing::info!("sensor connected from {peer} on {}", self.listener.label);
                    Some(Connection::Tcp(s))
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => None,
                Err(e) => return Err(DspError::Io(format!("{}: {e}", self.listener.label))),
            },
            #[cfg(unix)]
            Listener::Unix(l, _) => match l.accept() {
                Ok((s, _)) => {
                    s.set_nonblocking(false).map_err(|e| DspError::Io(e.to_string()))?;
                    tracing::info!("sensor driver connected on {}", self.listener.label);
                    Some(Connection::Unix(s))
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => None,
                Err(e) => return Err(DspError::Io(format!("{}: {e}", self.listener.label))),
            },
            Listener::Udp(_) => None,
        };
        let got = accepted.is_some();
        if let Some(c) = accepted {
            self.connection = Some(c);
        }
        Ok(got)
    }
}

impl LiveSource for SensorSource {
    fn layout(&self) -> &LiveLayout {
        self.reader.layout()
    }

    fn next(&mut self, wait: Duration) -> DspResult<LiveEvent> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(block) = self.reader.next_block() {
                return Ok(LiveEvent::Block(block));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(LiveEvent::Idle);
            }
            if let Listener::Udp(socket) = &self.listener.listener {
                socket.set_read_timeout(Some(left)).map_err(|e| DspError::Io(e.to_string()))?;
                match socket.recv(&mut self.buf) {
                    Ok(n) => self.reader.feed(&self.buf[..n]),
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => return Ok(LiveEvent::Idle),
                    Err(e) => return Err(DspError::Io(format!("{}: {e}", self.listener.label))),
                }
                continue;
            }
            // A newer sensor connection replaces the current one
            self.accept()?;
            let Some(conn) = self.connection.as_mut() else {
                std::thread::sleep(ACCEPT_POLL.min(left));
                continue;
            };
            match conn.read_for(&mut self.buf, left.min(Duration::from_millis(100))) {
                Ok(0) => {
                    tracing::info!("sensor disconnected from {}", self.listener.label);
                    self.connection = None;
                }
                Ok(n) => self.reader.feed(&self.buf[..n]),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(e) => {
                    tracing::warn!("sensor connection on {}: {e}", self.listener.label);
                    self.connection = None;
                }
            }
        }
    }

    fn counters(&self) -> SourceCounters {
        self.reader.counters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::sensor::encode_packet;
    use dspk_core::{ChannelInfo, SampleFormat, SampleRate, SignalUnit};
    use std::io::Write;

    fn layout() -> LiveLayout {
        let channels = (0..2).map(|i| ChannelInfo { name: format!("ch{i}"), gain: 1.0, offset: 0.0, unit: SignalUnit::Dimensionless }).collect();
        LiveLayout { name: "s".into(), channels, sample_rate: SampleRate::new(100.0).unwrap(), format: SampleFormat::I16 }
    }

    fn packet(seq: u32) -> Vec<u8> {
        encode_packet(SampleFormat::I16, 2, seq, 4, None, &[0u8; 2 * 4 * 2]).unwrap()
    }

    #[test]
    fn refuses_the_network_unless_allowed() {
        let open: SocketAddr = "0.0.0.0:0".parse().unwrap();
        assert!(SensorListener::bind(&SensorTransport::Tcp(open), false).is_err());
        assert!(SensorListener::bind(&SensorTransport::Udp(open), false).is_err());
    }

    #[test]
    fn tcp_sensor_reconnects() {
        let listener = SensorListener::bind(&SensorTransport::Tcp("127.0.0.1:0".parse().unwrap()), false).unwrap();
        let addr = listener.local_addr().unwrap();
        let mut source = SensorSource::new(listener, layout());
        assert!(matches!(source.next(Duration::from_millis(30)).unwrap(), LiveEvent::Idle));
        let mut firsts = Vec::new();
        for session in 0..2u32 {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(&packet(session * 2)).unwrap();
            s.write_all(&packet(session * 2 + 1)).unwrap();
            let mut got = 0;
            while got < 2 {
                if let LiveEvent::Block(b) = source.next(Duration::from_millis(500)).unwrap() {
                    firsts.push(b.first_sample);
                    got += 1;
                }
            }
        }
        assert_eq!(firsts, vec![0, 4, 8, 12]);
    }

    #[test]
    fn udp_datagrams() {
        let listener = SensorListener::bind(&SensorTransport::Udp("127.0.0.1:0".parse().unwrap()), false).unwrap();
        let addr = listener.local_addr().unwrap();
        let mut source = SensorSource::new(listener, layout());
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        let mut two = packet(0);
        two.extend(packet(1));
        sender.send_to(&two, addr).unwrap();
        let mut got = Vec::new();
        while got.len() < 2 {
            if let LiveEvent::Block(b) = source.next(Duration::from_millis(500)).unwrap() {
                got.push(b.first_sample);
            }
        }
        assert_eq!(got, vec![0, 4]);
    }
}
