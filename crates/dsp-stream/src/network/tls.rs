use std::sync::Arc;
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Result containing QUIC server and client configurations paired with a self-signed TLS 1.3 certificate.
pub struct QuicTlsBundle {
    pub server_config: quinn::ServerConfig,
    pub client_config: quinn::ClientConfig,
}

/// Generates a self-signed TLS 1.3 certificate in-memory and builds matching Quinn server & client configs.
pub fn generate_self_signed_tls(
    subject_alt_names: Vec<String>,
) -> Result<QuicTlsBundle, Box<dyn std::error::Error + Send + Sync>> {
    let cert_params = generate_simple_self_signed(subject_alt_names)?;
    let cert_der = cert_params.cert.der().to_vec();
    let key_der = cert_params.signing_key.serialize_der();

    let cert_chain = vec![CertificateDer::from(cert_der.clone())];
    let priv_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_der));

    // Server TLS Config
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain.clone(), priv_key)?;
    server_crypto.alpn_protocols = vec![b"dsp-stream-quic".to_vec()];

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
    ));

    // Client TLS Config: Trust the self-signed certificate root
    let mut root_store = rustls::RootCertStore::empty();
    root_store.add(CertificateDer::from(cert_der))?;

    let mut client_crypto = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![b"dsp-stream-quic".to_vec()];

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)?,
    ));

    Ok(QuicTlsBundle {
        server_config,
        client_config,
    })
}

/// Generates a standalone Quinn ServerConfig with a self-signed certificate, returning the config and DER bytes.
pub fn generate_server_config(
    subject_alt_names: Vec<String>,
) -> Result<(quinn::ServerConfig, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    let cert_params = generate_simple_self_signed(subject_alt_names)?;
    let cert_der = cert_params.cert.der().to_vec();
    let key_der = cert_params.signing_key.serialize_der();

    let cert_chain = vec![CertificateDer::from(cert_der.clone())];
    let priv_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_der));

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, priv_key)?;
    server_crypto.alpn_protocols = vec![b"dsp-stream-quic".to_vec()];

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
    ));

    Ok((server_config, cert_der))
}

/// Builds a QUIC client configuration trusting a specific certificate DER.
pub fn make_client_config_with_cert(
    cert_der: &[u8],
) -> Result<quinn::ClientConfig, Box<dyn std::error::Error + Send + Sync>> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.add(CertificateDer::from(cert_der.to_vec()))?;

    let mut client_crypto = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![b"dsp-stream-quic".to_vec()];

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)?,
    ));
    Ok(client_config)
}

#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
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
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Builds a QUIC client configuration that accepts any server certificate (useful for local benchmarks and test streams).
pub fn make_insecure_client_config() -> Result<quinn::ClientConfig, Box<dyn std::error::Error + Send + Sync>> {
    let mut client_crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![b"dsp-stream-quic".to_vec()];

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)?,
    ));
    Ok(client_config)
}


