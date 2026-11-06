//! Who may connect to a server: client certificates the server issues itself.
//!
//! A server keeps an [`Authority`] in a folder of its own (mode 0700): a certificate authority
//! (CA) key (mode 0600) and certificate, and a registry of the clients it enrolled
//! (`clients.json`: name, permission, certificate fingerprint, issued, revoked). Enrolling a
//! client issues it a certificate and key signed by the CA, written with the CA certificate into
//! one file for that client ([`Enrollment`], mode 0600). The client presents its certificate in
//! the TLS handshake (mutual TLS) and trusts the server through the same CA, so nothing else has
//! to be handed out.
//!
//! The handshake only accepts certificates signed by the CA; the registry then decides: a
//! certificate whose fingerprint is not enrolled, or is revoked, is refused. The registry is read
//! at every connection, so revoking takes effect for the next connection without a restart.
//! Each client has a [`Permission`]: `read` (recordings, views) or `control` (also configure and
//! run jobs).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rcgen::{date_time_ymd, BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{StreamError, StreamResult};

/// The CA certificate (PEM), what clients trust.
pub const CA_CERT_FILE: &str = "ca.pem";
/// The CA key (PEM, mode 0600): whoever has it can issue clients.
pub const CA_KEY_FILE: &str = "ca.key.pem";
/// The enrolled clients.
pub const CLIENTS_FILE: &str = "clients.json";
/// Years a client certificate is valid; then it is enrolled again.
pub const CLIENT_YEARS: i32 = 2;
/// Years the CA is valid.
pub const CA_YEARS: i32 = 20;
/// Years a server certificate is valid (it is issued again at every start).
pub const SERVER_YEARS: i32 = 1;
/// Longest client name.
pub const MAX_NAME_CHARS: usize = 64;
/// The name of a client without a certificate (only with anonymous access allowed).
pub const ANONYMOUS: &str = "anonymous";
const CA_NAME: &str = "dspk serve authority";
const SERVER_NAME: &str = "dspk serve";
const SECS_PER_YEAR: u64 = 31_556_952;

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

/// The client on the other end of a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peer {
    pub name: String,
    pub permission: Permission,
}

impl Peer {
    /// A client without a certificate: read only.
    pub fn anonymous() -> Self {
        Self { name: ANONYMOUS.into(), permission: Permission::Read }
    }

    pub fn can_control(&self) -> bool {
        self.permission == Permission::Control
    }
}

/// One enrolled client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientRecord {
    pub name: String,
    pub permission: Permission,
    /// SHA-256 of its certificate (DER), hex.
    pub fingerprint: String,
    pub issued_unix: u64,
    /// Its certificate is valid until the start of this year.
    pub expires_year: i32,
    /// When it was revoked; `None` while active.
    pub revoked_unix: Option<u64>,
}

impl ClientRecord {
    pub fn active(&self) -> bool {
        self.revoked_unix.is_none()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Registry {
    clients: Vec<ClientRecord>,
}

/// A client's certificate, key and the CA, as one PEM file to give to that client.
pub struct Enrollment {
    pub record: ClientRecord,
    pub pem: String,
}

impl Enrollment {
    /// Writes the file (mode 0600); refuses to overwrite one that exists.
    pub fn write(&self, path: &Path) -> StreamResult<()> {
        write_private(path, self.pem.as_bytes(), true)
    }
}

/// A client's identity read from its enrollment file: what it presents, and the CA it trusts.
#[derive(Clone)]
pub struct ClientIdentity {
    /// Its certificate (DER).
    pub certificate: Vec<u8>,
    /// Its private key (PKCS #8, DER).
    pub private_key: Vec<u8>,
    /// The certificate authority of the server(s) that enrolled it (DER).
    pub authorities: Vec<Vec<u8>>,
}

impl std::fmt::Debug for ClientIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key
        f.debug_struct("ClientIdentity").field("fingerprint", &fingerprint(&self.certificate)).finish_non_exhaustive()
    }
}

impl ClientIdentity {
    /// Reads an enrollment file: its first certificate is the client's, the others the CA's.
    pub fn from_file(path: &Path) -> StreamResult<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_pem(&bytes)
    }

    pub fn from_pem(bytes: &[u8]) -> StreamResult<Self> {
        let certs: Vec<Vec<u8>> = CertificateDer::pem_slice_iter(bytes).filter_map(Result::ok).map(|c| c.to_vec()).collect();
        let (certificate, authorities) = certs.split_first().ok_or_else(|| StreamError::Tls("no certificate in the identity file".into()))?;
        if authorities.is_empty() {
            return Err(StreamError::Tls("no authority (CA) certificate in the identity file".into()));
        }
        let key = PrivateKeyDer::from_pem_slice(bytes).map_err(|e| StreamError::Tls(format!("no private key in the identity file: {e}")))?;
        let PrivateKeyDer::Pkcs8(key) = key else { return Err(StreamError::Tls("the identity's key is not PKCS #8".into())) };
        Ok(Self { certificate: certificate.clone(), private_key: key.secret_pkcs8_der().to_vec(), authorities: authorities.to_vec() })
    }
}

/// A server's certificate authority and its registry of clients, in `dir`.
pub struct Authority {
    dir: PathBuf,
    ca_pem: String,
    ca_der: CertificateDer<'static>,
    ca_key: KeyPair,
}

impl std::fmt::Debug for Authority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Authority").field("dir", &self.dir).finish_non_exhaustive()
    }
}

impl Authority {
    /// Opens the authority in `dir`, creating it (a new CA) when the folder holds none. A folder
    /// with only one of the CA's two files is an error, never overwritten.
    pub fn open_or_create(dir: &Path) -> StreamResult<Self> {
        std::fs::create_dir_all(dir)?;
        restrict(dir, 0o700)?;
        let (cert_path, key_path) = (dir.join(CA_CERT_FILE), dir.join(CA_KEY_FILE));
        match (cert_path.exists(), key_path.exists()) {
            (true, true) => {
                let ca_pem = std::fs::read_to_string(&cert_path)?;
                let ca_key = KeyPair::from_pem(&std::fs::read_to_string(&key_path)?).map_err(tls)?;
                let ca_der = CertificateDer::from_pem_slice(ca_pem.as_bytes()).map_err(|e| StreamError::Tls(format!("{}: {e}", cert_path.display())))?.into_owned();
                Ok(Self { dir: dir.to_path_buf(), ca_pem, ca_der, ca_key })
            }
            (false, false) => {
                let ca_key = KeyPair::generate().map_err(tls)?;
                let mut params = ca_params();
                let year = this_year();
                (params.not_before, params.not_after) = (date_time_ymd(year - 1, 1, 1), date_time_ymd(year + CA_YEARS, 1, 1));
                let cert = params.self_signed(&ca_key).map_err(tls)?;
                write_private(&key_path, ca_key.serialize_pem().as_bytes(), true)?;
                let ca_pem = cert.pem();
                std::fs::write(&cert_path, &ca_pem)?;
                Ok(Self { dir: dir.to_path_buf(), ca_pem, ca_der: cert.der().clone(), ca_key })
            }
            _ => Err(StreamError::Tls(format!("{} holds only one of {CA_CERT_FILE} and {CA_KEY_FILE}: restore the other or remove both", dir.display()))),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The CA certificate (DER): what clients trust.
    pub fn ca_certificate(&self) -> &[u8] {
        &self.ca_der
    }

    fn issuer(&self) -> Issuer<'static, &KeyPair> {
        Issuer::new(ca_params(), &self.ca_key)
    }

    /// A new server certificate for `names` (host names or addresses clients use), signed by the
    /// CA, and its key; the chain ends with the CA certificate.
    pub fn server_certificate(&self, names: &[String]) -> StreamResult<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
        let key = KeyPair::generate().map_err(tls)?;
        let mut params = CertificateParams::new(names.to_vec()).map_err(tls)?;
        params.distinguished_name = common_name(SERVER_NAME);
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        let year = this_year();
        (params.not_before, params.not_after) = (date_time_ymd(year - 1, 1, 1), date_time_ymd(year + SERVER_YEARS + 1, 1, 1));
        let cert = params.signed_by(&key, &self.issuer()).map_err(tls)?;
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der()));
        Ok((vec![cert.der().clone(), self.ca_der.clone()], key))
    }

    /// Issues `name` a client certificate with `permission` and records it. A name already
    /// enrolled and not revoked is refused (revoke it first).
    pub fn enroll(&self, name: &str, permission: Permission) -> StreamResult<Enrollment> {
        check_name(name)?;
        let mut registry = self.registry()?;
        if registry.clients.iter().any(|c| c.name == name && c.active()) {
            return Err(StreamError::Tls(format!("{name} is already enrolled: revoke it first to issue a new certificate")));
        }
        let key = KeyPair::generate().map_err(tls)?;
        let mut params = CertificateParams::new(Vec::<String>::new()).map_err(tls)?;
        params.distinguished_name = common_name(name);
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        let year = this_year();
        let expires_year = year + CLIENT_YEARS + 1;
        (params.not_before, params.not_after) = (date_time_ymd(year - 1, 1, 1), date_time_ymd(expires_year, 1, 1));
        let cert = params.signed_by(&key, &self.issuer()).map_err(tls)?;
        let record = ClientRecord { name: name.into(), permission, fingerprint: fingerprint(cert.der()), issued_unix: now_unix(), expires_year, revoked_unix: None };
        registry.clients.push(record.clone());
        self.save(&registry)?;
        let pem = format!("{}{}{}", cert.pem(), key.serialize_pem(), self.ca_pem);
        Ok(Enrollment { record, pem })
    }

    /// Revokes `name`'s certificate: refused from its next connection on.
    pub fn revoke(&self, name: &str) -> StreamResult<ClientRecord> {
        let mut registry = self.registry()?;
        let Some(record) = registry.clients.iter_mut().find(|c| c.name == name && c.active()) else {
            return Err(StreamError::Tls(format!("no active client named {name}")));
        };
        record.revoked_unix = Some(now_unix());
        let revoked = record.clone();
        self.save(&registry)?;
        Ok(revoked)
    }

    /// Every client ever enrolled, revoked ones included.
    pub fn clients(&self) -> StreamResult<Vec<ClientRecord>> {
        Ok(self.registry()?.clients)
    }

    /// Who presents `certificate` (DER): an active enrolled client, else `None`.
    pub fn authorize(&self, certificate: &[u8]) -> Option<Peer> {
        let fp = fingerprint(certificate);
        let registry = self.registry().map_err(|e| tracing::warn!("reading the client registry: {e}")).ok()?;
        registry.clients.into_iter().find(|c| c.fingerprint == fp && c.active()).map(|c| Peer { name: c.name, permission: c.permission })
    }

    fn registry(&self) -> StreamResult<Registry> {
        match std::fs::read(self.dir.join(CLIENTS_FILE)) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| StreamError::Tls(format!("{CLIENTS_FILE}: {e}"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Registry::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Writes the registry whole and renames it into place, so a reader never sees half a file.
    fn save(&self, registry: &Registry) -> StreamResult<()> {
        let json = serde_json::to_vec_pretty(registry).map_err(|e| StreamError::Tls(e.to_string()))?;
        let tmp = self.dir.join(format!("{CLIENTS_FILE}.tmp"));
        let _ = std::fs::remove_file(&tmp);
        write_private(&tmp, &json, true)?;
        std::fs::rename(&tmp, self.dir.join(CLIENTS_FILE))?;
        Ok(())
    }
}

/// The CA's parameters. Issuing rebuilds them from here, so they must not depend on anything
/// but constants (the issued certificates name this issuer).
fn ca_params() -> CertificateParams {
    let mut p = CertificateParams::default();
    p.distinguished_name = common_name(CA_NAME);
    p.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    p.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign, KeyUsagePurpose::DigitalSignature];
    p
}

fn common_name(name: &str) -> DistinguishedName {
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, name);
    dn
}

/// Letters, digits, `.`, `_`, `-`; 1 to [`MAX_NAME_CHARS`] characters (names become file names).
fn check_name(name: &str) -> StreamResult<()> {
    let ok = !name.is_empty() && name.len() <= MAX_NAME_CHARS && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) && !name.starts_with('.');
    if ok { Ok(()) } else { Err(StreamError::Tls(format!("client name {name:?}: use 1–{MAX_NAME_CHARS} letters, digits, '.', '_' or '-', not starting with '.'"))) }
}

/// SHA-256 of `der`, lowercase hex.
pub fn fingerprint(der: &[u8]) -> String {
    Sha256::digest(der).iter().map(|b| format!("{b:02x}")).collect()
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The current year, close enough for validity periods (they start a year early and end at a
/// year's start, so a day's error never makes a certificate not yet valid).
fn this_year() -> i32 {
    1970 + (now_unix() / SECS_PER_YEAR) as i32
}

fn tls(e: impl std::fmt::Display) -> StreamError {
    StreamError::Tls(e.to_string())
}

/// Writes `bytes` to a new file readable only by its owner (0600 on Unix); `create_new` refuses
/// an existing file.
fn write_private(path: &Path, bytes: &[u8], create_new: bool) -> StreamResult<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true);
    if create_new {
        options.create_new(true);
    } else {
        options.create(true).truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Sets `path`'s permission bits (Unix; elsewhere nothing).
fn restrict(path: &Path, mode: u32) -> StreamResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    }
    #[cfg(not(unix))]
    let _ = (path, mode);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dspk-stream-auth-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn enroll_authorize_revoke() {
        let dir = scratch("cycle");
        let auth = Authority::open_or_create(&dir).unwrap();
        let alice = auth.enroll("alice", Permission::Control).unwrap();
        let id = ClientIdentity::from_pem(alice.pem.as_bytes()).unwrap();
        assert_eq!(id.authorities, vec![auth.ca_certificate().to_vec()]);
        assert_eq!(auth.authorize(&id.certificate), Some(Peer { name: "alice".into(), permission: Permission::Control }));
        assert!(auth.enroll("alice", Permission::Read).is_err(), "an active name is not issued twice");

        // The same authority after a restart: same CA, same clients
        let again = Authority::open_or_create(&dir).unwrap();
        assert_eq!(again.ca_certificate(), auth.ca_certificate());
        assert!(again.authorize(&id.certificate).is_some());

        auth.revoke("alice").unwrap();
        assert_eq!(again.authorize(&id.certificate), None, "revoked: refused, read fresh");
        assert!(auth.enroll("alice", Permission::Read).is_ok(), "a revoked name can be enrolled again");
        assert_eq!(auth.clients().unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_files_and_permissions() {
        for bad in ["", ".hidden", "a/b", "../x", "a b", &"x".repeat(MAX_NAME_CHARS + 1)] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
        assert!(check_name("lab-pc_2.local").is_ok());
        let dir = scratch("files");
        let auth = Authority::open_or_create(&dir).unwrap();
        let e = auth.enroll("bob", Permission::Read).unwrap();
        let out = dir.join("bob.pem");
        e.write(&out).unwrap();
        assert!(e.write(&out).is_err(), "never overwritten");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&dir), 0o700);
            assert_eq!(mode(&dir.join(CA_KEY_FILE)), 0o600);
            assert_eq!(mode(&out), 0o600);
            assert_eq!(mode(&dir.join(CLIENTS_FILE)), 0o600);
        }
        // Half an authority is refused, not replaced
        std::fs::remove_file(dir.join(CA_KEY_FILE)).unwrap();
        assert!(Authority::open_or_create(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
