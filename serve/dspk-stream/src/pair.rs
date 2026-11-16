//! Pairing with a short code, the part every side shares: the code as people type it, the
//! messages of the exchange, and a browser's side of it (which ends in a token). It has no
//! transport and no file in it, so a page built for WebAssembly uses it as the server does. The
//! server's side, and a computer's (which ends in a certificate), are in `transport::pairing`.

use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use crate::error::{StreamError, StreamResult};

pub(crate) const ID_CLIENT: &[u8] = b"dspk-pair-client";
pub(crate) const ID_SERVER: &[u8] = b"dspk-pair-server";

/// The path of the WebSocket on the server for browsers.
pub const WS_PATH: &str = "/ws";

/// What a client may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Permission {
    /// Recordings and views.
    Read,
    /// Also configure, start and stop jobs, and write.
    Control,
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Control => "control",
        })
    }
}

/// A page's first frame on the WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "hello", rename_all = "snake_case", deny_unknown_fields)]
pub enum WebHello {
    /// A token from an earlier pairing.
    Token { token: String },
    /// Pairing with a one-time code: the first message of the exchange. The server answers
    /// [`PairReply::Spake`], the page sends a [`PairToken`], the server answers
    /// [`PairReply::Token`].
    Pair { pair: PairHello },
}

/// The server's answer once it knows who the page is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebWelcome {
    pub server_name: String,
    pub server_version: String,
    pub peer_name: String,
    pub permission: Permission,
    pub has_files: bool,
}

/// `42-8317-2956` (spaces and dashes optional) as its id and secret.
pub fn parse_code(code: &str) -> Option<(String, String)> {
    let digits: String = code.chars().filter(|c| !c.is_whitespace() && *c != '-').collect();
    (digits.len() == 10 && digits.chars().all(|c| c.is_ascii_digit())).then(|| (digits[..2].to_string(), digits[2..].to_string()))
}

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
    /// For a browser (see [`PairToken`]): a token instead of a certificate, with a MAC proving
    /// the server knows the code.
    Token { token: String, names: Vec<String>, permission: Permission, mac: Vec<u8> },
    Error { reason: String },
}

/// A browser's second message: it asks for a token (it cannot hold a certificate), with a MAC
/// proving it knows the code.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairToken {
    pub mac: Vec<u8>,
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

pub(crate) fn mac(key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    hmac_of(key, parts).finalize().into_bytes().to_vec()
}

pub(crate) fn mac_ok(key: &[u8], parts: &[&[u8]], tag: &[u8]) -> bool {
    hmac_of(key, parts).verify_slice(tag).is_ok()
}

/// What a browser keeps after pairing.
#[derive(Debug, Clone)]
pub struct PairedToken {
    pub token: String,
    pub names: Vec<String>,
    pub permission: Permission,
}

/// The computer's side, step by step. Its private key is made here and never sent.
pub struct ClientPairing {
    pub(crate) state: Spake2<Ed25519Group>,
    pub(crate) name: String,
}


impl ClientPairing {
    /// Starts pairing as `name` with `code`: the state and the first message to send.
    pub fn start(code: &str, name: &str) -> StreamResult<(Self, PairHello)> {
        let (id, secret) = parse_code(code).ok_or_else(|| StreamError::Server("a pairing code is ten digits, like 42-8317-2956".into()))?;
        let (state, outbound) = Spake2::<Ed25519Group>::start_a(&Password::new(secret.as_bytes()), &Identity::new(ID_CLIENT), &Identity::new(ID_SERVER));
        Ok((Self { state, name: name.into() }, PairHello { id, name: name.into(), spake: outbound }))
    }
}

/// A browser's side after the server's SPAKE2 message: waits for its token.
pub struct ClientPairingToken {
    shared: Vec<u8>,
}

impl ClientPairing {
    /// With the server's first answer, for a browser: the request for a token to send.
    pub fn token(self, reply: PairReply) -> StreamResult<(ClientPairingToken, PairToken)> {
        let PairReply::Spake { spake } = reply else {
            return Err(StreamError::Server(match reply {
                PairReply::Error { reason } => reason,
                _ => "the server answered out of turn".into(),
            }));
        };
        let shared = self.state.finish(&spake).map_err(|_| StreamError::Server("pairing failed".into()))?;
        let tag = mac(&shared, &[b"client-token", self.name.as_bytes()]);
        Ok((ClientPairingToken { shared }, PairToken { mac: tag }))
    }
}

impl ClientPairingToken {
    /// With the server's last answer: the token to keep, once the server has proved it knows
    /// the code.
    pub fn finish(self, reply: PairReply) -> StreamResult<PairedToken> {
        match reply {
            PairReply::Token { token, names, permission, mac: tag } => {
                if !mac_ok(&self.shared, &[b"server-token", token.as_bytes(), names.join(",").as_bytes()], &tag) {
                    return Err(StreamError::Server("the server could not prove it knows the code: not paired".into()));
                }
                Ok(PairedToken { token, names, permission })
            }
            PairReply::Error { reason } => Err(StreamError::Server(reason)),
            _ => Err(StreamError::Server("the server answered out of turn".into())),
        }
    }
}

