//! TLS 1.3 for QUIC: the server's certificate (self-signed for development, or PEM files) and
//! what a client trusts (a given certificate, or, on a trusted network only, any).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

use super::auth::{Authority, ClientIdentity};
use crate::error::{StreamError, StreamResult};
use crate::protocol::ALPN;

/// Requests (view streams) a client may have open at once; one more for its control stream.
pub const MAX_CONCURRENT_REQUESTS: u32 = 8;
/// A connection with nothing sent for this long is closed (clients send keep-alives meanwhile).
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a client shows it is alive when it has nothing else to send.
pub const KEEP_ALIVE: Duration = Duration::from_secs(10);

/// The server's QUIC limits: [`MAX_CONCURRENT_REQUESTS`] bidirectional streams plus the control
/// one, no unidirectional streams from clients (only the server sends one, the signal), and
/// [`IDLE_TIMEOUT`].
fn server_transport() -> quinn::TransportConfig {
    let mut t = quinn::TransportConfig::default();
    t.max_concurrent_bidi_streams(quinn::VarInt::from_u32(MAX_CONCURRENT_REQUESTS + 1))
        .max_concurrent_uni_streams(quinn::VarInt::from_u32(0))
        .max_idle_timeout(Some(quinn::IdleTimeout::try_from(IDLE_TIMEOUT).expect("idle timeout fits QUIC's range")));
    t
}

/// The client's side: keep-alives within the server's idle timeout.
fn client_transport() -> quinn::TransportConfig {
    let mut t = quinn::TransportConfig::default();
    t.keep_alive_interval(Some(KEEP_ALIVE)).max_idle_timeout(Some(quinn::IdleTimeout::try_from(IDLE_TIMEOUT).expect("idle timeout fits QUIC's range")));
    t
}

/// The certificate a server presents.
#[derive(Debug, Clone)]
pub enum ServerIdentity {
    /// A new self-signed certificate for these names (development: clients must be given it).
    SelfSigned { names: Vec<String> },
    /// A certificate chain and private key in PEM files (e.g. from a certificate authority).
    Files { certificate_chain: PathBuf, private_key: PathBuf },
}

/// What a client accepts as the server's certificate.
#[derive(Debug, Clone)]
pub enum ServerTrust {
    /// Certificates (or their issuers) to trust, DER.
    Certificates(Vec<Vec<u8>>),
    /// Any certificate: the connection is encrypted but the server is not authenticated. Only on
    /// a network where every host is trusted.
    DangerAcceptAnyCertificate,
}

impl ServerTrust {
    /// Trusts the certificates in `path`: PEM (one or more) or a single DER certificate.
    pub fn from_file(path: &std::path::Path) -> StreamResult<Self> {
        let bytes = std::fs::read(path)?;
        let pem: Vec<Vec<u8>> = CertificateDer::pem_slice_iter(&bytes).filter_map(Result::ok).map(|c| c.to_vec()).collect();
        Ok(Self::Certificates(if pem.is_empty() { vec![bytes] } else { pem }))
    }
}

/// A server configuration and the certificate it presents (DER, to hand to clients).
pub struct ServerTls {
    pub config: quinn::ServerConfig,
    pub certificate: Vec<u8>,
}

fn tls_error(e: impl std::fmt::Display) -> StreamError {
    StreamError::Tls(e.to_string())
}

impl ServerTrust {
    /// Trusts the authority (CA) that enrolled `identity`: the server it was enrolled by.
    pub fn from_identity(identity: &ClientIdentity) -> Self {
        Self::Certificates(identity.authorities.clone())
    }
}

/// The QUIC server configuration for `identity`, without client certificates (anyone who
/// reaches the server may connect): tests and trusted networks only. A served deployment uses
/// [`server_config_with_clients`].
pub fn server_config(identity: &ServerIdentity) -> StreamResult<ServerTls> {
    let (chain, key) = identity_chain(identity)?;
    let certificate = chain[0].to_vec();
    let crypto = rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(chain, key).map_err(tls_error)?;
    finish_server(crypto, certificate)
}

/// The QUIC server configuration that requires a client certificate issued by `authority`
/// (mutual TLS); with `anonymous`, clients without one are let in too (read only, decided after
/// the handshake). The server presents `identity` when it is files, else
/// ([`ServerIdentity::SelfSigned`]) a certificate the authority issues for its names; clients
/// then trust the authority (the returned `certificate` is its CA).
pub fn server_config_with_clients(authority: &Authority, identity: &ServerIdentity, anonymous: bool) -> StreamResult<ServerTls> {
    let (chain, key, certificate) = match identity {
        ServerIdentity::Files { .. } => {
            let (chain, key) = identity_chain(identity)?;
            let first = chain[0].to_vec();
            (chain, key, first)
        }
        ServerIdentity::SelfSigned { names } => {
            let (chain, key) = authority.server_certificate(names)?;
            (chain, key, authority.ca_certificate().to_vec())
        }
    };
    let mut roots = rustls::RootCertStore::empty();
    roots.add(CertificateDer::from(authority.ca_certificate().to_vec())).map_err(tls_error)?;
    let mut verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots));
    if anonymous {
        verifier = verifier.allow_unauthenticated();
    }
    let verifier = verifier.build().map_err(tls_error)?;
    let crypto = rustls::ServerConfig::builder().with_client_cert_verifier(verifier).with_single_cert(chain, key).map_err(tls_error)?;
    finish_server(crypto, certificate)
}

fn finish_server(mut crypto: rustls::ServerConfig, certificate: Vec<u8>) -> StreamResult<ServerTls> {
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(crypto).map_err(tls_error)?;
    let mut config = quinn::ServerConfig::with_crypto(Arc::new(quic));
    config.transport_config(Arc::new(server_transport()));
    Ok(ServerTls { config, certificate })
}

/// The certificate chain and key `identity` names (a new self-signed one, or PEM files).
fn identity_chain(identity: &ServerIdentity) -> StreamResult<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    Ok(match identity {
        ServerIdentity::SelfSigned { names } => {
            let generated = rcgen::generate_simple_self_signed(names.clone()).map_err(tls_error)?;
            let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(generated.signing_key.serialize_der()));
            (vec![generated.cert.der().clone()], key)
        }
        ServerIdentity::Files { certificate_chain, private_key } => {
            let chain = CertificateDer::pem_file_iter(certificate_chain).map_err(tls_error)?.collect::<Result<Vec<_>, _>>().map_err(tls_error)?;
            if chain.is_empty() {
                return Err(StreamError::Tls(format!("no certificate in {}", certificate_chain.display())));
            }
            (chain, PrivateKeyDer::from_pem_file(private_key).map_err(tls_error)?)
        }
    })
}

/// The QUIC client configuration for `trust`, without a client certificate.
pub fn client_config(trust: &ServerTrust) -> StreamResult<quinn::ClientConfig> {
    client_config_with(trust, None)
}

/// The QUIC client configuration for `trust`, presenting `identity` (from `dspk clients enroll`)
/// when given.
pub fn client_config_with(trust: &ServerTrust, identity: Option<&ClientIdentity>) -> StreamResult<quinn::ClientConfig> {
    let builder = match trust {
        ServerTrust::Certificates(certificates) => {
            let mut roots = rustls::RootCertStore::empty();
            for der in certificates {
                roots.add(CertificateDer::from(der.clone())).map_err(tls_error)?;
            }
            rustls::ClientConfig::builder().with_root_certificates(roots)
        }
        ServerTrust::DangerAcceptAnyCertificate => rustls::ClientConfig::builder().dangerous().with_custom_certificate_verifier(Arc::new(AcceptAnyCertificate)),
    };
    let mut crypto = match identity {
        Some(id) => builder
            .with_client_auth_cert(vec![CertificateDer::from(id.certificate.clone())], PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(id.private_key.clone())))
            .map_err(tls_error)?,
        None => builder.with_no_client_auth(),
    };
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(tls_error)?;
    let mut config = quinn::ClientConfig::new(Arc::new(quic));
    config.transport_config(Arc::new(client_transport()));
    Ok(config)
}

/// Accepts every server certificate ([`ServerTrust::DangerAcceptAnyCertificate`]); signatures
/// are still checked, so the handshake stays encrypted.
#[derive(Debug)]
struct AcceptAnyCertificate;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyCertificate {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &rustls::crypto::ring::default_provider().signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &rustls::crypto::ring::default_provider().signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider().signature_verification_algorithms.supported_schemes()
    }
}
