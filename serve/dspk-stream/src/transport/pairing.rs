//! Pairing: a computer gets its certificate with a short code instead of a file handed over
//! (as lucem does).
//!
//! 1. On the server, `dspk serve --pair` (or `dspk clients pair`) makes a code such as
//!    `42-8317-2956`: the first part names it (public), the rest is the secret. It works once,
//!    for [`CODE_MINUTES`] minutes and [`CODE_TRIES`] tries. Codes live in the authority's folder
//!    (`pairing.json`, mode 0600), so a running server sees new ones at once.
//! 2. Both sides run SPAKE2 with the secret: each learns a shared key only if both know the
//!    code, and the code itself never crosses the network. A man in the middle gets one guess
//!    per try, then the code is spent.
//! 3. The computer makes its own key pair and sends only the public key, with a MAC under the
//!    shared key. The server issues a certificate for it, records the client in its registry,
//!    and answers with the certificate and its CA, MAC'd the same way.
//!
//! The exchange is written as steps over plain messages ([`PairHello`], [`PairReply`],
//! [`PairKey`]), with no transport in it: the QUIC server and client, and later a browser
//! endpoint, carry the same messages.

use std::path::{Path, PathBuf};

use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use super::auth::{now_unix, Authority, ClientIdentity, Permission};
use crate::error::{StreamError, StreamResult};

/// How long a code works (minutes).
pub const CODE_MINUTES: u64 = 10;
/// Wrong tries before a code is spent.
pub const CODE_TRIES: u32 = 3;

const CODES_FILE: &str = "pairing.json";
const ID_CLIENT: &[u8] = b"dspk-pair-client";
const ID_SERVER: &[u8] = b"dspk-pair-server";

/// A pairing code: `id-secret` as people type it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairingCode {
    /// Two digits naming the code (sent in the clear).
    pub id: String,
    /// Eight digits, the shared secret (never sent).
    pub secret: String,
    pub permission: Permission,
    pub expires_unix: u64,
    pub tries_left: u32,
}

impl PairingCode {
    /// `42-8317-2956`.
    pub fn display(&self) -> String {
        format!("{}-{}-{}", self.id, &self.secret[..4], &self.secret[4..])
    }
}

/// A live code as listed: everything but its secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveCode {
    pub id: String,
    pub permission: Permission,
    pub expires_unix: u64,
    pub tries_left: u32,
}

/// `42-8317-2956` (spaces and dashes optional) as its id and secret.
pub fn parse_code(code: &str) -> Option<(String, String)> {
    let digits: String = code.chars().filter(|c| !c.is_whitespace() && *c != '-').collect();
    (digits.len() == 10 && digits.chars().all(|c| c.is_ascii_digit())).then(|| (digits[..2].to_string(), digits[2..].to_string()))
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Codes {
    codes: Vec<PairingCode>,
}

fn random_digits(n: usize) -> StreamResult<String> {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).map_err(|e| StreamError::Tls(format!("no randomness: {e}")))?;
    // 250 = 25 × 10: values above it are drawn again, so every digit is equally likely
    let mut out = String::with_capacity(n);
    let mut pool = bytes.into_iter();
    while out.len() < n {
        let b = match pool.next() {
            Some(b) => b,
            None => {
                let mut one = [0u8; 1];
                getrandom::fill(&mut one).map_err(|e| StreamError::Tls(format!("no randomness: {e}")))?;
                one[0]
            }
        };
        if b < 250 {
            out.push(char::from(b'0' + b % 10));
        }
    }
    Ok(out)
}

impl Authority {
    fn codes_path(&self) -> PathBuf {
        self.dir().join(CODES_FILE)
    }

    fn read_codes(&self) -> StreamResult<Codes> {
        match std::fs::read_to_string(self.codes_path()) {
            Ok(t) => serde_json::from_str(&t).map_err(|e| StreamError::Server(format!("{}: {e}", self.codes_path().display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Codes::default()),
            Err(e) => Err(e.into()),
        }
    }

    fn write_codes(&self, codes: &Codes) -> StreamResult<()> {
        super::auth::replace_private(&self.codes_path(), serde_json::to_string_pretty(codes).map_err(|e| StreamError::Server(e.to_string()))?.as_bytes())
    }

    /// A new pairing code for a computer with `permission`, valid [`CODE_MINUTES`] minutes.
    pub fn new_pairing_code(&self, permission: Permission) -> StreamResult<PairingCode> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let id = loop {
            let id = random_digits(2)?;
            if !codes.codes.iter().any(|c| c.id == id) {
                break id;
            }
        };
        let code = PairingCode { id, secret: random_digits(8)?, permission, expires_unix: now + CODE_MINUTES * 60, tries_left: CODE_TRIES };
        codes.codes.push(code.clone());
        self.write_codes(&codes)?;
        Ok(code)
    }

    /// The live code named `id`, one try taken from it (a try counts before the answer is known,
    /// so guessing in parallel gains nothing).
    fn take_try(&self, id: &str) -> StreamResult<PairingCode> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let Some(code) = codes.codes.iter_mut().find(|c| c.id == id) else {
            self.write_codes(&codes)?;
            return Err(StreamError::Server("no such pairing code, or it has expired: make a new one on the server".into()));
        };
        code.tries_left -= 1;
        let taken = code.clone();
        self.write_codes(&codes)?;
        Ok(taken)
    }

    /// The live codes, without their secrets: what an operator may see to decide what to cancel.
    pub fn pairing_codes(&self) -> StreamResult<Vec<LiveCode>> {
        let now = now_unix();
        Ok(self.read_codes()?.codes.into_iter().filter(|c| c.expires_unix > now && c.tries_left > 0).map(|c| LiveCode { id: c.id, permission: c.permission, expires_unix: c.expires_unix, tries_left: c.tries_left }).collect())
    }

    /// Cancels the live code named `id`, or every code (`None`): a code that may have been seen
    /// stops working at once instead of at its expiry. Returns how many were cancelled.
    pub fn cancel_pairing_codes(&self, id: Option<&str>) -> StreamResult<usize> {
        let mut codes = self.read_codes()?;
        let now = now_unix();
        codes.codes.retain(|c| c.expires_unix > now && c.tries_left > 0);
        let before = codes.codes.len();
        match id {
            Some(id) => codes.codes.retain(|c| c.id != id),
            None => codes.codes.clear(),
        }
        let cancelled = before - codes.codes.len();
        self.write_codes(&codes)?;
        Ok(cancelled)
    }

    /// Spends the code named `id` (it paired a computer).
    fn spend(&self, id: &str) -> StreamResult<()> {
        let mut codes = self.read_codes()?;
        codes.codes.retain(|c| c.id != id);
        self.write_codes(&codes)
    }
}

// ---- the exchange (no transport) ---------------------------------------------------------------

/// The computer's first message: which code, its name, its SPAKE2 message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairHello {
    pub id: String,
    pub name: String,
    pub spake: Vec<u8>,
}

/// The server's answers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "pair", rename_all = "snake_case", deny_unknown_fields)]
pub enum PairReply {
    /// The server's SPAKE2 message.
    Spake { spake: Vec<u8> },
    /// The certificate issued, the CA, the server's names (to connect by) and the permission,
    /// with a MAC proving the server knows the code.
    Paired { certificate_pem: String, ca_pem: String, names: Vec<String>, permission: Permission, mac: Vec<u8> },
    Error { reason: String },
}

/// The computer's public key (SubjectPublicKeyInfo, DER) and its MAC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairKey {
    pub public_key: Vec<u8>,
    pub mac: Vec<u8>,
}

fn hmac_of(key: &[u8], parts: &[&[u8]]) -> Hmac<Sha256> {
    let mut m = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        m.update(&(p.len() as u64).to_le_bytes());
        m.update(p);
    }
    m
}

fn mac(key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    hmac_of(key, parts).finalize().into_bytes().to_vec()
}

fn mac_ok(key: &[u8], parts: &[&[u8]], tag: &[u8]) -> bool {
    hmac_of(key, parts).verify_slice(tag).is_ok()
}

fn error(reason: impl Into<String>) -> PairReply {
    PairReply::Error { reason: reason.into() }
}

/// The server's side after its SPAKE2 message: waits for the computer's key.
pub struct ServerPairing {
    key: Vec<u8>,
    name: String,
    code: PairingCode,
}

impl Authority {
    /// The server's first step: one try is taken from the code `hello` names, and the answer is
    /// the server's SPAKE2 message (then [`Self::pairing_finish`] with the returned state), or
    /// an error to send back.
    pub fn pairing_start(&self, hello: &PairHello) -> Result<(ServerPairing, PairReply), PairReply> {
        let code = self.take_try(&hello.id).map_err(|e| error(e.to_string()))?;
        if super::auth::check_name(&hello.name).is_err() {
            return Err(error("a computer's name is letters, digits, '-', '_' or '.'"));
        }
        let (state, outbound) = Spake2::<Ed25519Group>::start_b(&Password::new(code.secret.as_bytes()), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
        let key = state.finish(&hello.spake).map_err(|_| error("pairing failed"))?;
        Ok((ServerPairing { key, name: hello.name.clone(), code }, PairReply::Spake { spake: outbound }))
    }

    /// The server's second step: checks the computer knows the code, issues its certificate,
    /// spends the code. `names`: the names the server's certificate carries.
    pub fn pairing_finish(&self, pairing: ServerPairing, key: &PairKey, names: &[String]) -> PairReply {
        let ServerPairing { key: shared, name, code } = pairing;
        if !mac_ok(&shared, &[b"client", name.as_bytes(), &key.public_key], &key.mac) {
            tracing::warn!("pairing as {name:?} failed: wrong code ({} tries left)", code.tries_left);
            return error("wrong code");
        }
        let (record, certificate_pem) = match self.enroll_public_key(&name, code.permission, &key.public_key) {
            Ok(r) => r,
            Err(e) => return error(e.to_string()),
        };
        if let Err(e) = self.spend(&code.id) {
            return error(e.to_string());
        }
        let ca_pem = self.ca_pem().to_string();
        let tag = mac(&shared, &[b"server", certificate_pem.as_bytes(), ca_pem.as_bytes(), names.join(",").as_bytes()]);
        tracing::info!("paired {} ({})", record.name, record.permission);
        PairReply::Paired { certificate_pem, ca_pem, names: names.to_vec(), permission: code.permission, mac: tag }
    }
}

/// What a computer keeps after pairing.
#[derive(Debug, Clone)]
pub struct Paired {
    /// Certificate, private key and CA in one PEM (as an enrollment file): [`ClientIdentity::from_pem`].
    pub identity_pem: String,
    /// The names the server's certificate carries (connect with one of them).
    pub names: Vec<String>,
    pub permission: Permission,
    /// SHA-256 of the server's CA certificate (hex): how the server is recognised later.
    pub ca_fingerprint: String,
}

/// The computer's side, step by step. Its private key is made here and never sent.
pub struct ClientPairing {
    state: Spake2<Ed25519Group>,
    name: String,
}

/// After the server's SPAKE2 message: waits for the certificate.
pub struct ClientPairingKey {
    shared: Vec<u8>,
    key_pair: rcgen::KeyPair,
}

impl ClientPairing {
    /// Starts pairing as `name` with `code`: the state and the first message to send.
    pub fn start(code: &str, name: &str) -> StreamResult<(Self, PairHello)> {
        let (id, secret) = parse_code(code).ok_or_else(|| StreamError::Server("a pairing code is ten digits, like 42-8317-2956".into()))?;
        let (state, outbound) = Spake2::<Ed25519Group>::start_a(&Password::new(secret.as_bytes()), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
        Ok((Self { state, name: name.into() }, PairHello { id, name: name.into(), spake: outbound }))
    }

    /// With the server's first answer: the computer's key to send.
    pub fn key(self, reply: PairReply) -> StreamResult<(ClientPairingKey, PairKey)> {
        let spake = match reply {
            PairReply::Spake { spake } => spake,
            PairReply::Error { reason } => return Err(StreamError::Server(reason)),
            PairReply::Paired { .. } => return Err(StreamError::Server("the server answered out of turn".into())),
        };
        let shared = self.state.finish(&spake).map_err(|_| StreamError::Server("pairing failed".into()))?;
        let key_pair = rcgen::KeyPair::generate().map_err(|e| StreamError::Tls(e.to_string()))?;
        let public_key = rcgen::PublicKeyData::subject_public_key_info(&key_pair);
        let tag = mac(&shared, &[b"client", self.name.as_bytes(), &public_key]);
        Ok((ClientPairingKey { shared, key_pair }, PairKey { public_key, mac: tag }))
    }
}

impl ClientPairingKey {
    /// With the server's last answer: the identity to keep, once the server has proved it knows
    /// the code.
    pub fn finish(self, reply: PairReply) -> StreamResult<Paired> {
        match reply {
            PairReply::Paired { certificate_pem, ca_pem, names, permission, mac: tag } => {
                if !mac_ok(&self.shared, &[b"server", certificate_pem.as_bytes(), ca_pem.as_bytes(), names.join(",").as_bytes()], &tag) {
                    return Err(StreamError::Server("the server could not prove it knows the code: not paired".into()));
                }
                let identity_pem = format!("{certificate_pem}{}{ca_pem}", self.key_pair.serialize_pem());
                let identity = ClientIdentity::from_pem(identity_pem.as_bytes())?;
                Ok(Paired { ca_fingerprint: identity.ca_fingerprint(), identity_pem, names, permission })
            }
            PairReply::Error { reason } => Err(StreamError::Server(reason)),
            PairReply::Spake { .. } => Err(StreamError::Server("the server answered out of turn".into())),
        }
    }
}

// ---- over QUIC ---------------------------------------------------------------------------------

/// Largest pairing message (a certificate and a CA in PEM are a few kilobytes).
const MAX_PAIR_MESSAGE_BYTES: usize = 64 << 10;
/// How long a pairing connection may take in all.
const PAIR_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Whether `connection` was opened to pair ([`crate::protocol::PAIR_ALPN`]).
pub(crate) fn is_pairing(connection: &quinn::Connection) -> bool {
    connection
        .handshake_data()
        .and_then(|d| d.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
        .and_then(|d| d.protocol)
        .is_some_and(|p| p == crate::protocol::PAIR_ALPN)
}

async fn send_json(stream: &mut quinn::SendStream, message: &impl Serialize) -> StreamResult<()> {
    let body = serde_json::to_vec(message).map_err(|e| StreamError::Protocol(e.to_string()))?;
    let mut buf = Vec::with_capacity(4 + body.len());
    buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
    buf.extend_from_slice(&body);
    stream.write_all(&buf).await?;
    Ok(())
}

async fn receive_json<M: for<'de> Deserialize<'de>>(stream: &mut quinn::RecvStream) -> StreamResult<M> {
    let mut prefix = [0u8; 4];
    stream.read_exact(&mut prefix).await?;
    let len = u32::from_be_bytes(prefix) as usize;
    if len > MAX_PAIR_MESSAGE_BYTES {
        return Err(StreamError::TooLarge { len, max: MAX_PAIR_MESSAGE_BYTES });
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    serde_json::from_slice(&body).map_err(|e| StreamError::Protocol(e.to_string()))
}

/// The server's side of a pairing connection: one stream, [`PairHello`] → [`PairReply`] →
/// [`PairKey`] → [`PairReply`].
pub(crate) async fn serve_pairing(authority: &Authority, names: &[String], connection: &quinn::Connection) -> StreamResult<()> {
    let exchange = async {
        let (mut send, mut recv) = connection.accept_bi().await?;
        let hello: PairHello = receive_json(&mut recv).await?;
        let pairing = match authority.pairing_start(&hello) {
            Ok((pairing, reply)) => {
                send_json(&mut send, &reply).await?;
                pairing
            }
            Err(refusal) => {
                send_json(&mut send, &refusal).await?;
                let _ = send.finish();
                return Ok(());
            }
        };
        let key: PairKey = receive_json(&mut recv).await?;
        send_json(&mut send, &authority.pairing_finish(pairing, &key, names)).await?;
        let _ = send.finish();
        // The computer closes once it has read the answer
        connection.closed().await;
        Ok(())
    };
    match tokio::time::timeout(PAIR_TIMEOUT, exchange).await {
        Ok(result) => result,
        Err(_) => {
            connection.close(quinn::VarInt::from_u32(0), b"pairing took too long");
            Err(StreamError::Protocol("pairing took too long".into()))
        }
    }
}

/// The computer's side: pairs with the server at `addr` using `code`, as `name`. Its private key
/// is made here and never sent.
pub async fn pair(addr: std::net::SocketAddr, code: &str, name: &str) -> StreamResult<Paired> {
    let (client, hello) = ClientPairing::start(code, name)?;
    let mut endpoint = quinn::Endpoint::client((if addr.is_ipv6() { "[::]:0" } else { "0.0.0.0:0" }).parse().expect("an address"))?;
    endpoint.set_default_client_config(super::tls::pairing_client_config()?);
    let connection = endpoint.connect(addr, "dspk-pairing")?.await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    send_json(&mut send, &hello).await?;
    let (client, key) = client.key(receive_json(&mut recv).await?)?;
    send_json(&mut send, &key).await?;
    let reply: PairReply = receive_json(&mut recv).await?;
    connection.close(quinn::VarInt::from_u32(0), b"paired");
    client.finish(reply)
}

/// Writes a pairing code to `path`, readable by its owner only (for a script that pairs: it
/// reads the file and removes it; the code never goes through a pipe or a log).
pub fn save_code(path: &Path, code: &PairingCode) -> StreamResult<()> {
    super::auth::replace_private(path, format!("{}\n", code.display()).as_bytes())
}

/// Writes `pem` readable by its owner only.
pub fn save_identity(path: &Path, pem: &str) -> StreamResult<()> {
    super::auth::replace_private(path, pem.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_read_with_or_without_dashes() {
        assert_eq!(parse_code("42-8317-2956"), Some(("42".into(), "83172956".into())));
        assert_eq!(parse_code("42 8317 2956"), Some(("42".into(), "83172956".into())));
        assert_eq!(parse_code("42-8317-295"), None);
        assert_eq!(parse_code("ab-8317-2956"), None);
    }

    #[test]
    fn a_code_is_spent_after_its_tries() {
        let dir = std::env::temp_dir().join(format!("dspk-pair-codes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Authority::open_or_create(&dir).unwrap();
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        assert_eq!(code.display().len(), 12);
        for _ in 0..CODE_TRIES {
            authority.take_try(&code.id).unwrap();
        }
        assert!(authority.take_try(&code.id).is_err(), "no tries left");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn live_codes_list_without_secrets_and_cancel() {
        let dir = std::env::temp_dir().join(format!("dspk-pair-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Authority::open_or_create(&dir).unwrap();
        let a = authority.new_pairing_code(Permission::Read).unwrap();
        let b = authority.new_pairing_code(Permission::Control).unwrap();
        let listed = authority.pairing_codes().unwrap();
        assert_eq!(listed.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), [a.id.as_str(), b.id.as_str()]);
        assert_eq!(listed[1].permission, Permission::Control);
        assert_eq!(authority.cancel_pairing_codes(Some(&a.id)).unwrap(), 1);
        assert!(authority.take_try(&a.id).is_err(), "a cancelled code no longer works");
        assert_eq!(authority.cancel_pairing_codes(Some("no such id")).unwrap(), 0);
        assert_eq!(authority.cancel_pairing_codes(None).unwrap(), 1);
        assert!(authority.pairing_codes().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn authority(name: &str) -> (Authority, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("dspk-pair-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (Authority::open_or_create(&dir).unwrap(), dir)
    }

    #[test]
    fn a_computer_pairs_with_the_code_and_connects_as_enrolled() {
        let (authority, dir) = authority("exchange");
        let code = authority.new_pairing_code(Permission::Control).unwrap();
        let names = vec!["localhost".to_string()];
        let (client, hello) = ClientPairing::start(&code.display(), "laptop").unwrap();
        let (server, reply) = authority.pairing_start(&hello).unwrap();
        let (client, key) = client.key(reply).unwrap();
        let paired = client.finish(authority.pairing_finish(server, &key, &names)).unwrap();
        assert_eq!((paired.permission, paired.names.clone()), (Permission::Control, names));
        // The certificate it holds is one the server authorizes, under its name
        let identity = ClientIdentity::from_pem(paired.identity_pem.as_bytes()).unwrap();
        assert_eq!(paired.ca_fingerprint, identity.ca_fingerprint());
        let clients = authority.clients().unwrap();
        assert_eq!(clients.iter().map(|c| (c.name.as_str(), c.permission)).collect::<Vec<_>>(), [("laptop", Permission::Control)]);
        assert!(authority.pairing_codes().unwrap().is_empty(), "the code is spent");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_code_pairs_nothing_and_costs_a_try() {
        let (authority, dir) = authority("wrong");
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        let wrong = format!("{}-0000-0000", code.id);
        let (client, hello) = ClientPairing::start(&wrong, "laptop").unwrap();
        let (server, reply) = authority.pairing_start(&hello).unwrap();
        // The two sides hold different keys: the computer's MAC is refused
        let (_, key) = client.key(reply).unwrap();
        assert!(matches!(authority.pairing_finish(server, &key, &[]), PairReply::Error { .. }));
        assert!(authority.clients().unwrap().is_empty());
        assert_eq!(authority.pairing_codes().unwrap()[0].tries_left, CODE_TRIES - 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_server_that_does_not_know_the_code_is_refused() {
        let (authority, dir) = authority("impostor");
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        let (client, _) = ClientPairing::start(&code.display(), "laptop").unwrap();
        // An impostor answers with its own SPAKE2 message for another secret, then a certificate
        let (_, forged) = Spake2::<Ed25519Group>::start_b(&Password::new(b"00000000"), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
        let (client, _) = client.key(PairReply::Spake { spake: forged }).unwrap();
        let reply = PairReply::Paired { certificate_pem: "x".into(), ca_pem: "y".into(), names: vec![], permission: Permission::Control, mac: vec![0; 32] };
        assert!(client.finish(reply).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
