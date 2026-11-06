//! `clients`: who may connect to `dspk serve`. Enrolling issues a client a certificate signed by
//! the server's own authority (written with its key and the authority's certificate to one file
//! for that client); revoking refuses it from its next connection on. The server reads the list
//! at every connection, so neither needs a restart.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Args, Subcommand};
use dspk_stream::{Authority, Permission};

/// Environment variable naming the authority folder (shared by `serve` and `clients`).
pub const AUTH_DIR_ENV: &str = "DSPK_SERVE_AUTH_DIR";
/// The authority folder under the dspk data folder when none is given.
pub const AUTH_DIR_NAME: &str = "serve";
/// The file an enrollment is written to, after the client's name.
pub const IDENTITY_SUFFIX: &str = ".dspk-identity.pem";

/// The authority folder: `--auth-dir`, `$DSPK_SERVE_AUTH_DIR`, else `<data dir>/serve`
/// (`~/.local/share/dspk/serve` on Linux; `$DSPK_DATA_DIR/serve` when that is set).
pub fn auth_dir(given: Option<&Path>) -> PathBuf {
    given.map(Path::to_path_buf).unwrap_or_else(|| crate::sorters_program::data_dir().join(AUTH_DIR_NAME))
}

#[derive(Args, Debug)]
pub struct ClientsArgs {
    /// The server's authority folder (default: <data dir>/serve, e.g. ~/.local/share/dspk/serve)
    #[arg(long, env = AUTH_DIR_ENV, global = true)]
    auth_dir: Option<PathBuf>,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    /// Issue a client its certificate (read only unless --control) and write it to a file to give
    /// to that client (`dspk receive --identity <file>`)
    Enroll {
        /// Who the client is: a person or a machine (letters, digits, '.', '_', '-')
        name: String,
        /// Also configure, start and stop jobs, and write (not only read recordings)
        #[arg(long)]
        control: bool,
        /// Where to write the client's file (default: <name>.dspk-identity.pem here; `-`: to
        /// standard output, e.g. through `podman exec … > file`, so no copy stays on the server);
        /// never overwritten. Never put it under the server's data root
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Refuse a client from its next connection on
    Revoke {
        name: String,
    },
    /// The clients enrolled, active and revoked
    List,
}

pub fn run(args: &ClientsArgs) -> anyhow::Result<()> {
    let dir = auth_dir(args.auth_dir.as_deref());
    let authority = Authority::open_or_create(&dir).with_context(|| format!("opening the authority in {}", dir.display()))?;
    match &args.action {
        Action::Enroll { name, control, out } => {
            let permission = if *control { Permission::Control } else { Permission::Read };
            let out = out.clone().unwrap_or_else(|| PathBuf::from(format!("{name}{IDENTITY_SUFFIX}")));
            if out.as_os_str() == "-" {
                // To standard output only; the messages go to stderr
                let enrollment = authority.enroll(name, permission)?;
                print!("{}", enrollment.pem);
                eprintln!("Enrolled {name} ({permission}), certificate {}; valid until {}", &enrollment.record.fingerprint[..16], enrollment.record.expires_year);
                eprintln!("The identity (with its private key) went to standard output: keep it private (chmod 600)");
                return Ok(());
            }
            // Check the file can be written before recording the client
            if out.exists() {
                anyhow::bail!("{} exists: choose another --out (it is never overwritten)", out.display());
            }
            let enrollment = authority.enroll(name, permission)?;
            enrollment.write(&out).with_context(|| format!("writing {}", out.display()))?;
            println!("Enrolled {name} ({permission}), certificate {}", &enrollment.record.fingerprint[..16]);
            println!("Its file: {} (certificate, private key, the server's authority; mode 0600)", out.display());
            println!("Give this file to the client {name:?} over a private channel; it connects with `dspk remote <address> --identity {} info`", out.display());
            println!("Valid until {}; `dspk clients revoke {name}` refuses it sooner", enrollment.record.expires_year);
        }
        Action::Revoke { name } => {
            let record = authority.revoke(name)?;
            println!("Revoked {name} (certificate {}): refused from its next connection on", &record.fingerprint[..16]);
        }
        Action::List => {
            let clients = authority.clients()?;
            if clients.is_empty() {
                println!("No clients enrolled in {} (`dspk clients enroll <name>`)", dir.display());
            }
            for c in clients {
                let state = if c.active() { "active" } else { "revoked" };
                println!("{:<24} {:<8} {:<8} certificate {}  valid until {}", c.name, c.permission, state, &c.fingerprint[..16], c.expires_year);
            }
        }
    }
    Ok(())
}
