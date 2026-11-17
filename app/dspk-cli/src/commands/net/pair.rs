//! `pair`: this computer gets its certificate from a `dspk serve` server with a one-time code
//! (`dspk serve --pair`, `dspk clients pair`), instead of a file handed over. The private key
//! is made here and never leaves this computer.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use dspk_stream::transport::pairing;

#[derive(Args, Debug)]
pub struct PairArgs {
    /// The server: host:port (the UDP port `dspk serve` listens on)
    server: String,
    /// The code the server showed, e.g. 42-8317-2956
    code: String,
    /// What the server calls this computer (letters, digits, '.', '_', '-'; default: its host name)
    #[arg(long)]
    name: Option<String>,
    /// Where to write this computer's identity (default: <name>.dspk-identity.pem here; mode
    /// 0600). Give it to `dspk receive --identity`, `dspk remote --identity`, or the app
    #[arg(long)]
    out: Option<PathBuf>,
}

fn host_name() -> String {
    let raw = std::fs::read_to_string("/etc/hostname").ok().or_else(|| std::env::var("HOSTNAME").ok()).or_else(|| std::env::var("COMPUTERNAME").ok()).unwrap_or_default();
    let name: String = raw.trim().chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')).collect();
    if name.is_empty() { "computer".into() } else { name }
}

pub async fn run(args: &PairArgs) -> anyhow::Result<()> {
    let name = args.name.clone().unwrap_or_else(host_name);
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from(format!("{name}.dspk-identity.pem")));
    anyhow::ensure!(!out.exists(), "{} exists: choose another --out (an identity is never overwritten)", out.display());
    let addr = tokio::net::lookup_host(&args.server).await.with_context(|| format!("resolving {}", args.server))?.next().with_context(|| format!("{}: no address", args.server))?;
    let paired = pairing::pair(addr, &args.code, &name).await?;
    pairing::save_identity(&out, &paired.identity_pem).with_context(|| format!("writing {}", out.display()))?;
    println!("Paired with {} as {name} ({})", args.server, paired.permission);
    println!("The server's authority: {} (SHA-256)", paired.ca_fingerprint);
    println!("The server's names: {} (connect by one of them)", paired.names.join(", "));
    println!("Identity written to {} (certificate, private key, the server's authority; mode 0600)", out.display());
    println!("Next: dspk receive --identity {} {}  (a dspk serve server)", out.display(), args.server);
    println!("  or: dspk device status --identity {} {}  (a device)", out.display(), args.server);
    Ok(())
}
