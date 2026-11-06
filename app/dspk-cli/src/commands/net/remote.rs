//! `remote`: work with a `dspk serve` server: who it is, its files, sorts it runs. The client
//! proves who it is with its identity file (`dspk clients enroll` on the server).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::{Args, Subcommand};
use dspk_stream::protocol::wire;
use dspk_stream::{client_config_with, ClientIdentity, ServerTrust, Session};

/// How often `sort --wait` and `status --follow` ask for the status.
const FOLLOW_EVERY: Duration = Duration::from_secs(2);
/// Log lines `logs` shows by default.
const DEFAULT_LOG_LINES: u32 = 50;

#[derive(Args, Debug)]
pub struct RemoteArgs {
    /// Server address (IP:port)
    addr: SocketAddr,
    /// This client's identity: the file `dspk clients enroll` wrote for it
    #[arg(long, env = "DSPK_IDENTITY")]
    identity: PathBuf,
    /// Name the server's certificate must carry (one of its --name, or localhost)
    #[arg(long, default_value = "localhost")]
    server_name: String,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    /// Who the server is, who this client is to it, and what it is doing
    Info,
    /// The files under the server's data root (directories, recordings, sortings)
    Files {
        /// A directory under the data root (default: the root)
        #[arg(default_value = "")]
        dir: String,
    },
    /// Sort a recording on the server (control permission)
    Sort {
        /// The recording, under the server's data root
        recording: String,
        /// kilosort4, emusort, mountainsort5, spykingcircus2, tridesclous2
        #[arg(long, default_value = "kilosort4")]
        sorter: String,
        /// Where the sorting goes, under the data root (default: next to the recording,
        /// <name>.<sorter>.sorting.zarr)
        #[arg(long, default_value = "")]
        output: String,
        /// A probe preset (default: the recording's own geometry)
        #[arg(long, default_value = "")]
        probe: String,
        /// The source within the recording's file (default: its default one)
        #[arg(long, default_value = "")]
        source: String,
        /// Start of the part to sort (s)
        #[arg(long, default_value_t = 0.0)]
        start_sec: f64,
        /// Length of the part to sort (s; 0 = to the end)
        #[arg(long, default_value_t = 0.0)]
        duration_sec: f64,
        /// A sorter parameter away from its default, as name=value (repeat; names as in
        /// `dspk sort <sorter> --help`, without "--")
        #[arg(long = "param", value_parser = parse_param)]
        params: Vec<(String, String)>,
        /// Follow the sort to its end
        #[arg(long)]
        wait: bool,
    },
    /// The server's state and its job's progress
    Status {
        /// Keep showing it until the job ends
        #[arg(long)]
        follow: bool,
    },
    /// The job's last log lines
    Logs {
        #[arg(short = 'n', long, default_value_t = DEFAULT_LOG_LINES)]
        lines: u32,
    },
    /// Stop the running job (control permission)
    Stop,
}

fn parse_param(s: &str) -> Result<(String, String), String> {
    let (name, value) = s.split_once('=').ok_or_else(|| format!("{s:?}: use name=value"))?;
    Ok((name.trim_start_matches("--").to_string(), value.to_string()))
}

fn state_name(state: i32) -> &'static str {
    match wire::DeviceState::try_from(state).unwrap_or(wire::DeviceState::Unspecified) {
        wire::DeviceState::Stopped => "stopped",
        wire::DeviceState::Running => "running",
        wire::DeviceState::Error => "error",
        wire::DeviceState::Unspecified => "unknown",
    }
}

fn print_status(s: &wire::Status) {
    let progress = if s.total > 0 { format!("  [{}/{}] {} {}/{} {}", s.step, s.steps, s.stage, s.done, s.total, s.unit) } else { String::new() };
    println!("{}{}{}", state_name(s.state), if s.job.is_empty() { String::new() } else { format!(": {}", s.job) }, progress);
    if !s.message.is_empty() {
        println!("  {}", s.message);
    }
    if !s.output.is_empty() {
        println!("  output: {}", s.output);
    }
    if !s.started_by.is_empty() {
        println!("  started by {}", s.started_by);
    }
}

async fn follow(session: &Session) -> anyhow::Result<wire::Status> {
    let mut last = String::new();
    loop {
        let status = session.status().await?;
        let line = format!("{:?}{}{}{}", status.state, status.stage, status.done, status.message);
        if line != last {
            print_status(&status);
            last = line;
        }
        if status.state != wire::DeviceState::Running as i32 {
            return Ok(status);
        }
        tokio::time::sleep(FOLLOW_EVERY).await;
    }
}

pub async fn run(args: &RemoteArgs) -> anyhow::Result<()> {
    let identity = ClientIdentity::from_file(&args.identity).with_context(|| format!("reading {}", args.identity.display()))?;
    let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity))?;
    let session = Session::connect(args.addr, &args.server_name, config).await.with_context(|| format!("connecting to {}", args.addr))?;
    let server = session.server().clone();
    match &args.action {
        Action::Info => {
            let permission = if server.permission == wire::Permission::Control as i32 { "control" } else { "read" };
            println!("{} (dspk {}), protocol {}", server.server_name, server.server_version, server.protocol_version);
            println!("You: {} ({permission})", server.peer_name);
            println!("State: {}", state_name(server.state));
            println!("Files: {}", if server.has_files { "yes (`dspk remote … files`)" } else { "none" });
        }
        Action::Files { dir } => {
            let list = session.list_files(dir).await?;
            println!("/{}", list.dir);
            for e in &list.entries {
                let kind = match wire::FileKind::try_from(e.kind).unwrap_or(wire::FileKind::Other) {
                    wire::FileKind::Directory => "dir",
                    wire::FileKind::Recording => "recording",
                    wire::FileKind::Sorting => "sorting",
                    _ => "",
                };
                let size = if e.size > 0 { format!("{:.1} MB", e.size as f64 / 1e6) } else { String::new() };
                println!("  {:<10} {:>12}  {}", kind, size, e.name);
            }
        }
        Action::Sort { recording, sorter, output, probe, source, start_sec, duration_sec, params, wait } => {
            let job = wire::Configure {
                sorter: sorter.clone(),
                recording: recording.clone(),
                output: output.clone(),
                params: params.iter().cloned().collect(),
                source: source.clone(),
                probe: probe.clone(),
                start_sec: *start_sec,
                duration_sec: *duration_sec,
            };
            let configured = session.configure(job).await?;
            println!("Configured {} → {}", configured.job, configured.output);
            print_status(&session.start().await?);
            if *wait {
                let end = follow(&session).await?;
                if end.state == wire::DeviceState::Error as i32 {
                    anyhow::bail!("the sort failed: {}", end.message);
                }
            }
        }
        Action::Status { follow: keep } => {
            if *keep {
                follow(&session).await?;
            } else {
                print_status(&session.status().await?);
            }
        }
        Action::Logs { lines } => {
            for line in session.logs(*lines).await? {
                println!("{line}");
            }
        }
        Action::Stop => print_status(&session.stop().await?),
    }
    session.close().await;
    Ok(())
}
