//! Servers in a browser tab: one, the server that served the page, reached through the page's
//! WebSocket (`dspk_stream::web_client`); a profile only names it. The page keeps its token in
//! the browser's storage and asks for a pairing code when it has none. Downloads and jobs are
//! not offered to a page yet.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use dspk_stream::pair::{WebWelcome, WS_PATH};
use dspk_stream::web_client::{self, Hello, WebLink};
use dspk_stream::{Link, RemoteSignal};

use crate::data::jobs::{Cancel, JobMessage};
use crate::data::remote::{parse_uri, remember, wire, ServerProfile, DEFAULT_SERVER_NAME};

/// The name of the page's server among the app's servers (`dspk://server/<path>`).
pub const PAGE_SERVER: &str = "server";
/// Where the page keeps its token.
const TOKEN_KEY: &str = "dspk-token";
const NOT_YET: &str = "not offered to a page yet (the desktop app does it)";

static LINK: OnceLock<Arc<WebLink>> = OnceLock::new();

fn link() -> anyhow::Result<Arc<WebLink>> {
    LINK.get().cloned().ok_or_else(|| anyhow::anyhow!("the page is not connected to its server"))
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

/// `ws://` or `wss://` to the page's own host; `None` when the page was not served over HTTP(S).
fn socket_url() -> Option<String> {
    let location = web_sys::window()?.location();
    let scheme = match location.protocol().ok()?.as_str() {
        "https:" => "wss",
        "http:" => "ws",
        _ => return None,
    };
    Some(format!("{scheme}://{}{WS_PATH}", location.host().ok().unwrap_or_default()))
}

/// Connects the page to the server that served it (once, on the main thread, after the window
/// is open): with the token kept from an earlier visit, else with a pairing code the user types.
/// A page no `dspk serve` served (a plain file server, GitHub Pages) has no server: `Ok(None)`.
pub async fn connect_page() -> anyhow::Result<Option<WebWelcome>> {
    if let Some(l) = LINK.get() {
        return Ok(Some(l.welcome().clone()));
    }
    let Some(url) = socket_url() else { return Ok(None) };
    let kept = storage().and_then(|s| s.get_item(TOKEN_KEY).ok().flatten()).filter(|t| !t.is_empty());
    if let Some(token) = kept {
        match web_client::connect(&url, Hello::Token(token)).await {
            Ok((link, _)) => return Ok(Some(keep(link))),
            Err(_) => {
                if let Some(s) = storage() {
                    let _ = s.remove_item(TOKEN_KEY);
                }
            }
        }
    }
    // Nothing answers the socket: a plain file server, not an error
    let probe = web_client::connect(&url, Hello::Token(String::new())).await;
    if matches!(&probe, Err(e) if e.to_string().starts_with(web_client::UNREACHED)) {
        return Ok(None);
    }
    let prompt = "Pairing code (shown by `dspk serve --pair`, or `dspk clients pair`)";
    let code = web_sys::window().and_then(|w| w.prompt_with_message(prompt).ok().flatten()).unwrap_or_default();
    if code.trim().is_empty() {
        anyhow::bail!("no pairing code was given: reload the page to try again");
    }
    // A name of its own: the server knows each paired browser by name
    let name = format!("browser-{:06x}", (js_sys::Math::random() * 16_777_216.0) as u32);
    let (link, token) = web_client::connect(&url, Hello::Code { code, name }).await?;
    if let (Some(token), Some(s)) = (token, storage()) {
        let _ = s.set_item(TOKEN_KEY, &token);
    }
    Ok(Some(keep(link)))
}

fn keep(link: Arc<WebLink>) -> WebWelcome {
    let welcome = link.welcome().clone();
    let _ = LINK.set(link);
    welcome
}

/// The page's server as a profile, so the screens that list servers show it.
pub fn page_profile() -> ServerProfile {
    ServerProfile { name: PAGE_SERVER.into(), address: "this page's server".into(), server_name: DEFAULT_SERVER_NAME.into(), identity: PathBuf::new() }
}

/// Where the functions below run in a browser: on the page's own executor, when awaited.
pub struct PageRuntime;
/// A task of [`PageRuntime::spawn`]: awaiting it runs it (it never fails to join; the error type
/// matches a tokio task's, so callers are the same on both targets).
pub struct Task<T>(std::pin::Pin<Box<dyn std::future::Future<Output = T>>>);

impl PageRuntime {
    pub fn spawn<F: std::future::Future + 'static>(&self, future: F) -> Task<F::Output> {
        Task(Box::pin(future))
    }
}

impl<T> std::future::Future for Task<T> {
    type Output = Result<T, String>;
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        self.0.as_mut().poll(cx).map(Ok)
    }
}

pub fn runtime() -> &'static PageRuntime {
    &PageRuntime
}

/// Opens the `dspk://` recording `path` on the page's server (from a worker: it waits). The page
/// has one connection, so one recording of the server is open at a time.
pub fn open_uri(path: &Path) -> anyhow::Result<Arc<RemoteSignal>> {
    let (_, remote) = parse_uri(path).ok_or_else(|| anyhow::anyhow!("{} is not a dspk:// path", path.display()))?;
    let link: Arc<dyn Link> = link()?;
    let signal = Arc::new(RemoteSignal::open(link, &remote)?);
    remember(&signal);
    Ok(signal)
}

pub async fn test(_profile: ServerProfile) -> anyhow::Result<wire::ServerInfo> {
    let w = link()?.welcome().clone();
    Ok(wire::ServerInfo { server_name: w.server_name, server_version: w.server_version, peer_name: w.peer_name, has_files: w.has_files, ..Default::default() })
}

pub async fn list(_profile: ServerProfile, dir: String) -> anyhow::Result<wire::FileList> {
    match link()?.request_async(wire::request::Kind::ListFiles(wire::ListFiles { dir })).await? {
        wire::response::Kind::Files(list) => Ok(list),
        _ => anyhow::bail!("the server answered something else than a file list"),
    }
}

pub async fn fetch(_profile: ServerProfile, _path: String, _results: PathBuf) -> anyhow::Result<PathBuf> {
    anyhow::bail!(NOT_YET)
}

pub fn remote_sort(_profile: ServerProfile, _job: wire::Configure, _results: PathBuf, send: impl Fn(JobMessage) + Send + Sync + 'static) -> Cancel {
    send(JobMessage::Failed(NOT_YET.into()));
    Cancel::default()
}
