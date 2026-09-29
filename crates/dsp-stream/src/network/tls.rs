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
