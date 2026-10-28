//! TLS 1.3 for QUIC: the server's certificate (self-signed for development, or PEM files) and
//! what a client trusts (a given certificate, or, on a trusted network only, any).

use std::path::PathBuf;
use std::sync::Arc;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

use crate::error::{StreamError, StreamResult};
use crate::protocol::ALPN;

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

/// The QUIC server configuration for `identity`.
pub fn server_config(identity: &ServerIdentity) -> StreamResult<ServerTls> {
    let (chain, key): (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) = match identity {
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
    };
    let certificate = chain[0].to_vec();
    let mut crypto = rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(chain, key).map_err(tls_error)?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(crypto).map_err(tls_error)?;
    Ok(ServerTls { config: quinn::ServerConfig::with_crypto(Arc::new(quic)), certificate })
}

/// The QUIC client configuration for `trust`.
pub fn client_config(trust: &ServerTrust) -> StreamResult<quinn::ClientConfig> {
    let mut crypto = match trust {
        ServerTrust::Certificates(certificates) => {
            let mut roots = rustls::RootCertStore::empty();
            for der in certificates {
                roots.add(CertificateDer::from(der.clone())).map_err(tls_error)?;
            }
            rustls::ClientConfig::builder().with_root_certificates(roots).with_no_client_auth()
        }
        ServerTrust::DangerAcceptAnyCertificate => rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCertificate))
            .with_no_client_auth(),
    };
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(tls_error)?;
    Ok(quinn::ClientConfig::new(Arc::new(quic)))
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
