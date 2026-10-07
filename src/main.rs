mod cli;
mod hlo;
mod server;
#[cfg(test)]
mod tests;
mod tools;
mod trace;
mod xplane;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Context;
use axum::body::{Body, Bytes};
use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::Response;
use axum::{Router, routing::any};
use cli::json::py_repr;
use futures_util::future::{BoxFuture, FutureExt, Shared as Joined, try_join_all};
use include_dir::{Dir, include_dir};
use rayon::prelude::*;
use server::remote::Remote;
use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::panic::AssertUnwindSafe;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::Semaphore;
use tools::opstats::OpStats;
use tower_http::{CompressionLevel, catch_panic::CatchPanicLayer, compression::CompressionLayer};
use trace::json::{View, quoted, render};
use trace::{MAX_SERIAL, Options, Trace};
use xplane::Plane;

/// The physical memory, which sizes the caches and the number of loads at once. A load needs about seven times the size of its file.
static MEMORY: LazyLock<u64> = LazyLock::new(|| {
    let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    (pages.max(0) as u64).saturating_mul(size.max(0) as u64)
});
static BUDGET_BYTES: LazyLock<u64> = LazyLock::new(|| (*MEMORY / 8).clamp(256 << 20, 4 << 30));
const IDLE: Duration = Duration::from_secs(3600);
const POLL: Duration = Duration::from_secs(30);
const FRESH: Duration = Duration::from_secs(2);
const WATCHED_SESSIONS: usize = 8;
const CONCURRENT_PREFETCHES: usize = 1;
/// The prefetch starts a load only after this time without requests. A load at the same time as a request makes the request slow.
const QUIET: Duration = Duration::from_secs(1);
const GZIP_CHUNK: usize = 1 << 20;
const GZIP_HEADER: [u8; 10] = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
const READ_CHUNK: usize = 16 << 20;
const MAX_THREADS: usize = 32;
/// `mi_option_purge_delay` of mimalloc, which `libmimalloc-sys` does not name.
const PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
const DEFAULT_PORT: u16 = 8791;
const DEFAULT_GRPC_PORT: u16 = 50051;
const DEFAULT_RESOLUTION: f64 = 8000.0;
const PREFIX: &str = "/data/plugin/profile";
const VERSION: &str = "2.23.2";
const CACHE_TOOLS: [&str; 2] = ["overview_page", "trace_viewer@"];
pub(crate) const XPLANE_TOOLS: [&str; 19] = [
    "trace_viewer",
    "trace_viewer@",
    "overview_page",
    "input_pipeline_analyzer",
    "framework_op_stats",
    "kernel_stats",
    "memory_profile",
    "pod_viewer",
    "op_profile",
    "hlo_stats",
    "roofline_model",
    "inference_profile",
    "memory_viewer",
    "graph_viewer",
    "megascale_stats",
    "perf_counters",
    "utilization_viewer",
    "kernel_utilization",
    "smart_suggestion",
];
const SWITCHES: [&str; 2] = ["--hide_capture_profile_button", "--enable_tab_name_label"];
const VALUE_FLAGS: [&str; 7] = ["--logdir", "--port", "--host", "--src_prefix", "--grpc_port", "--worker_service_address", "--max_concurrent_worker_requests"];
const NO_DATA: &str = "No Data";
const OUTSIDE: &str = "Path outside logdir";
const USAGE: &str = "usage: xprof-rs [--logdir DIR|URL] [--port PORT] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]\n       [--grpc_port PORT] [--worker_service_address ADDRESS] [--max_concurrent_worker_requests N]  (The server accepts the last three flags and does not use them.)";
const SECURITY_POLICY: &str = "default-src 'self';script-src 'self' 'unsafe-eval' 'unsafe-inline' https://www.gstatic.com;object-src 'none';style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://www.gstatic.com;font-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com data:;connect-src 'self' data: www.gstatic.com;img-src 'self' blob: data:;frame-src 'self' https://ui.perfetto.dev;script-src-elem 'self' 'unsafe-inline' https://cdn.jsdelivr.net/npm/ https://www.gstatic.com";
const ALL_HOSTS_ONLY: [&str; 3] = ["overview_page", "pod_viewer", "smart_suggestion"];
const ALL_HOSTS_ALSO: [&str; 6] = ["input_pipeline_analyzer", "framework_op_stats", "kernel_stats", "overview_page", "pod_viewer", "megascale_stats"];
const CONTENT_TYPES: [(&str, &str); 4] = [("html", "text/html"), ("js", "application/javascript"), ("css", "text/css"), ("wasm", "application/wasm")];
const RENDERED_FROM: [&str; 2] = [".xplane.pb", ".hlo_proto.pb"];
const STATIC_DIR: &str = "XPROF_STATIC_DIR";
static ASSETS: Dir = include_dir!("$CARGO_MANIFEST_DIR/static");

struct Host {
    bytes: u64,
    map: Vec<u8>,
    planes: Vec<Plane>,
    trace: Trace,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Stamp {
    len: u64,
    modified: SystemTime,
    inode: u64,
    device: u64,
}

impl Stamp {
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self { len: meta.len(), modified: meta.modified().ok()?, inode: meta.ino(), device: meta.dev() })
    }

    fn older_than(&self, age: Duration) -> bool {
        self.modified.elapsed().map_or(true, |elapsed| elapsed >= age)
    }
}

type Outcome<T> = Result<T, Arc<str>>;
type Flight<T> = Joined<BoxFuture<'static, Outcome<T>>>;

struct Flights<K, T>(Arc<Mutex<HashMap<K, Flight<T>>>>);

impl<K: Hash + Eq + Clone + Send + 'static, T: Clone + Send + Sync + 'static> Flights<K, T> {
    fn new() -> Self {
        Self(Arc::default())
    }

    fn join<F: Future<Output = T> + Send + 'static>(&self, key: K, work: impl FnOnce() -> F) -> Flight<T> {
        let mut flights = self.0.lock().unwrap();
        let flight = flights.entry(key.clone()).or_insert_with(|| {
            let (flights, work) = (self.0.clone(), AssertUnwindSafe(work()).catch_unwind());
            let task = tokio::spawn(async move {
                let outcome = work.await.map_err(|payload| panic_message(&*payload));
                flights.lock().unwrap().remove(&key);
                outcome
            });
            task.map(|joined| joined.unwrap_or_else(|error| Err(error.to_string().into()))).boxed().shared()
        });
        flight.clone()
    }
}

struct Memo<T> {
    cache: moka::future::Cache<PathBuf, (Stamp, Outcome<T>)>,
    flights: Flights<(PathBuf, Stamp), Outcome<T>>,
}

impl<T: Clone + Send + Sync + 'static> Memo<T> {
    fn new(weigh: impl Fn(&Stamp, &T) -> u64 + Send + Sync + 'static) -> Arc<Self> {
        let cache = cache(*BUDGET_BYTES, move |_, (stamp, outcome): &(Stamp, Outcome<T>)| outcome.as_ref().map_or(0, |value| weigh(stamp, value)));
        Arc::new(Self { cache, flights: Flights::new() })
    }

    async fn get(self: &Arc<Self>, path: PathBuf, permits: &Arc<Semaphore>, build: fn(&Path) -> anyhow::Result<T>) -> Outcome<T> {
        let stamp = Stamp::of(&path).ok_or_else(|| Arc::from(format!("Cannot read {}", path.display())))?;
        if let Some((cached, outcome)) = self.cache.get(&path).await
            && cached == stamp
        {
            return outcome;
        }
        let (memo, permits) = (self.clone(), permits.clone());
        let flight = self.flights.join((path.clone(), stamp), move || async move {
            let _permit = permits.acquire_owned().await;
            let file = path.clone();
            let outcome =
                AssertUnwindSafe(blocking(move || build(&file))).catch_unwind().await.map_err(|payload| panic_message(&*payload)).and_then(|outcome| outcome.map_err(|error| error.to_string().into()));
            if Stamp::of(&path) == Some(stamp) && stamp.older_than(FRESH) {
                memo.cache.insert(path, (stamp, outcome.clone())).await;
            }
            outcome
        });
        flight.await.and_then(|outcome| outcome)
    }
}

#[derive(Clone)]
struct Rendered {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    gzipped: bool,
}

#[derive(Debug, Default, PartialEq)]
struct Settings {
    logdir: PathBuf,
    port: u16,
    host: Option<String>,
    src_prefix: Option<String>,
    hide_capture_profile_button: bool,
    enable_tab_name_label: bool,
    remote: Option<Arc<Remote>>,
}

struct State_ {
    logdir: PathBuf,
    remote: Option<Arc<Remote>>,
    config: String,
    hosts: Arc<Memo<Arc<Host>>>,
    stats: Arc<Memo<Option<Arc<OpStats>>>>,
    rendered: moka::future::Cache<String, Rendered>,
    renders: Flights<String, Rendered>,
    extracts: Flights<PathBuf, Vec<String>>,
    loads: Arc<Semaphore>,
}

type Shared = Arc<State_>;
type Params = HashMap<String, String>;
type Render = Box<dyn FnOnce(&OpStats) -> String + Send>;
type Failure = (StatusCode, String);
type Sessions = Vec<(String, PathBuf, String)>;

fn cache<K: Hash + Eq + Send + Sync + 'static, V: Clone + Send + Sync + 'static>(budget: u64, weigh: impl Fn(&K, &V) -> u64 + Send + Sync + 'static) -> moka::future::Cache<K, V> {
    moka::future::Cache::builder()
        .weigher(move |key, value| (weigh(key, value) >> 10).clamp(1, budget >> 10) as u32)
        .max_capacity(budget >> 10)
        .eviction_policy(moka::policy::EvictionPolicy::lru())
        .time_to_idle(IDLE)
        .build()
}

fn panic_message(payload: &(dyn Any + Send)) -> Arc<str> {
    match (payload.downcast_ref::<&str>(), payload.downcast_ref::<String>()) {
        (Some(message), _) => Arc::from(*message),
        (_, Some(message)) => Arc::from(message.as_str()),
        _ => Arc::from("internal error"),
    }
}

async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(work).await.unwrap_or_else(|error| std::panic::resume_unwind(error.into_panic()))
}

/// Creates a directory that only this user can read. It refuses a directory that another user owns.
fn private_dir(dir: &Path) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    let meta = std::fs::symlink_metadata(dir)?;
    if !meta.is_dir() || meta.uid() != unsafe { libc::getuid() } {
        return Err(std::io::Error::other("another user owns this directory"));
    }
    if meta.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    dir.canonicalize()
}

fn read_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut bytes = vec![0; file.metadata()?.len() as usize];
    bytes.par_chunks_mut(READ_CHUNK).enumerate().try_for_each(|(index, chunk)| file.read_exact_at(chunk, (index * READ_CHUNK) as u64))?;
    Ok(bytes)
}

fn prepare(path: &Path, trace: bool) -> anyhow::Result<(Vec<u8>, Vec<Plane>)> {
    let map = read_file(path)?;
    let mut planes = xplane::parse(&map)?;
    finish(&mut planes, &map, trace);
    Ok((map, planes))
}

/// A file that is not a valid `XSpace` gives `None`. The parse checks the events, and the check of the other fields runs at the same time.
fn parse_checked(map: &[u8], bare: bool) -> anyhow::Result<Option<Vec<Plane>>> {
    let stats = AtomicBool::new(true);
    let (valid, planes) = rayon::join(|| tools::counters::valid_besides_events(map), || xplane::parse_checking(map, Some(&stats), bare));
    match planes {
        _ if !valid => Ok(None),
        Ok(planes) => Ok(stats.into_inner().then_some(planes)),
        // The parse can stop before it checks all of the events.
        Err(error) => {
            if tools::counters::valid_space(map) {
                Err(error)
            } else {
                Ok(None)
            }
        }
    }
}

/// Adds the regions, the groups, and the derived lines.
fn finish(planes: &mut [Plane], map: &[u8], trace: bool) {
    if group(planes, map, trace) {
        derive(planes, map);
    }
}

/// Adds the derived lines of the TPUs.
fn derive(planes: &mut [Plane], map: &[u8]) {
    planes.par_iter_mut().filter(|plane| xplane::derive::is_tensor_core(&plane.name)).for_each(|plane| xplane::derive::derive(plane, map));
}

/// Adds the regions, the groups, and the derived lines of the GPUs, but not the derived lines of the TPUs. Gives `false` if the planes have their groups already.
fn group(planes: &mut [Plane], map: &[u8], trace: bool) -> bool {
    planes.par_iter_mut().for_each(|plane| plane.add_threadpool_regions(map));
    let ungrouped = !xplane::derive::is_grouped(planes);
    if ungrouped {
        let groups = xplane::group::group(planes, map);
        xplane::derive::derive_gpu(planes, map, groups.as_ref().map(|groups| &groups.names), trace);
    }
    ungrouped
}

fn release(garbage: impl Send + 'static) {
    std::thread::spawn(move || {
        drop(garbage);
        unsafe { libmimalloc_sys::mi_collect(true) };
    });
}

fn load_host(path: &Path) -> anyhow::Result<Host> {
    let begin = Instant::now();
    let (map, mut planes) = prepare(path, true)?;
    let name = host_name(path);
    let start = Instant::now();
    let trace = Trace::build(&planes, &name, &map);
    release(planes.iter_mut().map(|plane| std::mem::take(&mut plane.lines)).collect::<Vec<_>>());
    eprintln!("Loaded {name}: {} events. The trace took {:?}. The load took {:?}.", trace.events.len(), start.elapsed(), begin.elapsed());
    Ok(Host { bytes: (map.len() + trace.events.len() * std::mem::size_of::<trace::Event>()) as u64, map, planes, trace })
}

fn shared_host(path: &Path) -> anyhow::Result<Arc<Host>> {
    load_host(path).map(Arc::new)
}

fn host_name(path: &Path) -> String {
    path.file_name().unwrap_or_default().to_string_lossy().trim_end_matches(".xplane.pb").to_string()
}

fn list(dir: &Path, keep: impl Fn(&std::fs::DirEntry) -> bool) -> Vec<PathBuf> {
    std::fs::read_dir(dir).map(|entries| entries.filter_map(Result::ok).filter(|entry| keep(entry)).map(|entry| entry.path()).collect()).unwrap_or_default()
}

fn xplanes(dir: &Path) -> Vec<PathBuf> {
    list(dir, |entry| entry.file_name().to_string_lossy().ends_with(".xplane.pb"))
}

fn sessions(logdir: &Path) -> Vec<(String, PathBuf)> {
    if logdir.as_os_str().is_empty() {
        return Vec::new();
    }
    let is_dir = |entry: &std::fs::DirEntry| entry.file_type().is_ok_and(|kind| kind.is_dir());
    let (mut found, mut pending) = (Vec::new(), vec![(logdir.to_path_buf(), String::new())]);
    while let Some((dir, prefix)) = pending.pop() {
        found.extend(list(&dir.join("plugins/profile"), is_dir).into_iter().map(|session| (format!("{prefix}{}", session.file_name().unwrap().to_string_lossy()), session)));
        pending.extend(
            list(&dir, is_dir)
                .into_iter()
                .filter(|child| child.file_name().is_some_and(|name| name != "plugins"))
                .map(|child| (child.clone(), format!("{prefix}{}/", child.file_name().unwrap().to_string_lossy()))),
        );
    }
    found
}

fn not_found() -> Response {
    failure((StatusCode::NOT_FOUND, NO_DATA.into()))
}

fn outside() -> Failure {
    (StatusCode::BAD_REQUEST, OUTSIDE.into())
}

macro_rules! or_fail {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return failure(error),
        }
    };
}

fn failure((status, message): Failure) -> Response {
    response(status, "text/plain", message)
}

fn internal(message: &str) -> Response {
    response(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", message.to_string())
}

fn response(status: StatusCode, content_type: &str, body: impl Into<Body>) -> Response {
    Response::builder().status(status).header(header::CONTENT_TYPE, content_type).header("x-served-by", "xprof-rs").body(body.into()).unwrap()
}

async fn assets(request: Request) -> Response {
    let path = request.uri().path();
    let name = match path.strip_prefix(PREFIX).unwrap_or(path).trim_matches('/') {
        "" | "local" => "index.html",
        name => name,
    };
    let Some(file) = ASSETS.get_file(format!("{name}.gz")) else { return response(StatusCode::NOT_FOUND, "text/plain", "Not Found") };
    let extension = name.rsplit('.').next().unwrap_or("");
    let content_type = CONTENT_TYPES.iter().find(|(known, _)| *known == extension).map_or("application/octet-stream", |(_, kind)| kind);
    if let Some(base) = std::env::var_os(STATIC_DIR).and_then(|dir| PathBuf::from(dir).canonicalize().ok()).filter(|dir| dir.is_dir()) {
        return match base.join(name).canonicalize().ok().filter(|path| path.starts_with(&base)).and_then(|path| std::fs::read(path).ok()) {
            Some(contents) => response(StatusCode::OK, content_type, contents),
            None => response(StatusCode::NOT_FOUND, "text/plain", "Fail to read the files."),
        };
    }
    negotiate(response(StatusCode::OK, content_type, Body::empty()), Bytes::from_static(file.contents()), accepts_gzip(request.headers()))
}

/// Sets a gzip body without changes when the client accepts gzip, and decoded when it does not.
fn negotiate(mut reply: Response, body: Bytes, accepts_gzip: bool) -> Response {
    let headers = reply.headers_mut();
    headers.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    if accepts_gzip {
        headers.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        *reply.body_mut() = Body::from(body);
    } else {
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(&body[..]).read_to_end(&mut raw).unwrap();
        *reply.body_mut() = Body::from(raw);
    }
    reply
}

/// Resolves the deepest part of the path that exists. The parts after it must be plain names, and the first of them must not be a symlink to a path that does not exist.
fn confine(state: &State_, path: &Path) -> Option<PathBuf> {
    let parts: Vec<Component> = path.components().collect();
    let (found, real) = (0..=parts.len()).rev().find_map(|found| {
        let prefix: PathBuf = parts[..found].iter().collect();
        Some((found, if prefix.as_os_str().is_empty() { Path::new(".") } else { &prefix }.canonicalize().ok()?))
    })?;
    let missing = &parts[found..];
    let plain = missing.iter().all(|part| matches!(part, Component::Normal(_))) && missing.first().is_none_or(|first| real.join(first).symlink_metadata().is_err());
    (plain && real.starts_with(&state.logdir)).then(|| missing.iter().fold(real, |dir, part| dir.join(part)))
}

fn run_dir(state: &State_, run: &str) -> Option<PathBuf> {
    let run = run.trim_end_matches('/');
    if state.logdir.as_os_str().is_empty() || Path::new(run).components().any(|part| !matches!(part, Component::Normal(_) | Component::CurDir)) {
        return None;
    }
    let (prefix, session) = run.rsplit_once('/').unwrap_or((".", run));
    confine(state, &state.logdir.join(prefix).join("plugins/profile").join(session))
}

fn session_map(state: &State_, params: &Params) -> Result<Option<Sessions>, Failure> {
    let is_dir = |entry: &std::fs::DirEntry| entry.file_type().is_ok_and(|kind| kind.is_dir());
    let name = |dir: &Path| dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
    if let Some(path) = params.get("session_path").filter(|path| !path.is_empty()) {
        let dir = confine(state, &state.logdir.join(path)).ok_or_else(outside)?;
        return Ok(Some(if xplanes(&dir).is_empty() { Vec::new() } else { vec![(name(Path::new(path)), dir, path.clone())] }));
    }
    let Some(path) = params.get("run_path").filter(|path| !path.is_empty()) else { return Ok(None) };
    let dir = confine(state, &state.logdir.join(path)).ok_or_else(outside)?;
    Ok(Some(
        list(&dir, is_dir)
            .into_iter()
            .filter(|session| !xplanes(session).is_empty())
            .map(|session| (name(&session), session.clone(), Path::new(path).join(name(&session)).display().to_string()))
            .collect(),
    ))
}

fn session(state: &State_, params: &Params) -> Result<PathBuf, Failure> {
    let run = params.get("run");
    if let Some(map) = session_map(state, params)? {
        return map.iter().find(|(name, _, _)| Some(name) == run).map(|(_, dir, _)| dir.clone()).ok_or_else(|| {
            let entries: Vec<String> = map.iter().map(|(name, _, shown)| format!("{}: {}", py_repr(name), py_repr(shown))).collect();
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Run {} not found in run map: {{{}}}", run.map_or("None", String::as_str), entries.join(", ")))
        });
    }
    match run {
        Some(run) => run_dir(state, run).ok_or_else(outside),
        None => Err((StatusCode::INTERNAL_SERVER_ERROR, "'NoneType' object has no attribute 'rstrip'".into())),
    }
}

fn accepts_gzip(headers: &HeaderMap) -> bool {
    let Some(value) = headers.get(header::ACCEPT_ENCODING).and_then(|value| value.to_str().ok()) else { return false };
    let weight = |name: &str| {
        value.split(',').find_map(|coding| {
            let mut parts = coding.split(';').map(str::trim);
            parts.next()?.eq_ignore_ascii_case(name).then(|| parts.find_map(|part| part.strip_prefix("q=")).map_or(1.0, |weight| weight.parse::<f32>().unwrap_or(0.0)))
        })
    };
    weight("gzip").or_else(|| weight("*")).is_some_and(|weight| weight > 0.0)
}

async fn cached(state: &Shared, key: String, dir: &Path, accepts_gzip: bool, render: impl Future<Output = Response> + Send + 'static) -> Response {
    let mut sources = list(dir, |entry| RENDERED_FROM.iter().any(|suffix| entry.file_name().to_string_lossy().ends_with(suffix)));
    sources.sort();
    let stamps: Vec<(PathBuf, Option<Stamp>)> = sources.into_iter().map(|file| (file.clone(), Stamp::of(&file))).collect();
    let settled = stamps.iter().all(|(_, stamp)| stamp.is_some_and(|stamp| stamp.older_than(FRESH)));
    let mut hasher = std::hash::DefaultHasher::new();
    stamps.hash(&mut hasher);
    let key = format!("{key}#{:x}", hasher.finish());
    let rendered = if let Some(hit) = state.rendered.get(&key).await {
        Ok(hit)
    } else {
        let (cache, stored) = (state.rendered.clone(), key.clone());
        let flight = state.renders.join(key, move || async move {
            let (parts, body) = render.await.into_parts();
            let body = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
            if parts.status != StatusCode::OK {
                return Rendered { status: parts.status, headers: parts.headers, body, gzipped: false };
            }
            let body = blocking(move || {
                let chunks = body.len().div_ceil(GZIP_CHUNK).max(1);
                let parts: Vec<(Vec<u8>, crc32fast::Hasher)> = (0..chunks)
                    .into_par_iter()
                    .map(|index| {
                        let chunk = &body[(index * GZIP_CHUNK).min(body.len())..((index + 1) * GZIP_CHUNK).min(body.len())];
                        let flush = if index + 1 == chunks { flate2::FlushCompress::Finish } else { flate2::FlushCompress::Sync };
                        let (mut deflate, mut out) = (flate2::Compress::new(flate2::Compression::fast(), false), Vec::with_capacity(chunk.len() / 2 + 128));
                        // The writer of flate2 can stop a sync flush when its buffer is full. This loop gives more space until the flush is complete.
                        loop {
                            let consumed = deflate.total_in() as usize;
                            let status = deflate.compress_vec(&chunk[consumed..], &mut out, flush).unwrap();
                            if status == flate2::Status::StreamEnd || (deflate.total_in() as usize == chunk.len() && out.len() < out.capacity()) {
                                break;
                            }
                            out.reserve(out.capacity() / 2 + 1024);
                        }
                        let mut crc = crc32fast::Hasher::new();
                        crc.update(chunk);
                        (out, crc)
                    })
                    .collect();
                let mut packed = GZIP_HEADER.to_vec();
                let mut crc = crc32fast::Hasher::new();
                for (deflated, part) in &parts {
                    packed.extend_from_slice(deflated);
                    crc.combine(part);
                }
                packed.extend_from_slice(&crc.finalize().to_le_bytes());
                packed.extend_from_slice(&(body.len() as u32).to_le_bytes());
                Bytes::from(packed)
            })
            .await;
            let rendered = Rendered { status: parts.status, headers: parts.headers, body, gzipped: true };
            if settled {
                cache.insert(stored, rendered.clone()).await;
            }
            rendered
        });
        flight.await
    };
    let Rendered { status, headers, body, gzipped } = match rendered {
        Ok(rendered) => rendered,
        Err(message) => return internal(&message),
    };
    let mut reply = Response::new(Body::from(body.clone()));
    *reply.status_mut() = status;
    *reply.headers_mut() = headers;
    if gzipped { negotiate(reply, body, accepts_gzip) } else { reply }
}

async fn run_tools(State(state): State<Shared>, Query(params): Query<Params>) -> Response {
    let dir = or_fail!(session(&state, &params));
    response(StatusCode::OK, "application/json", blocking(move || server::run_tools::json(&dir)).await)
}

async fn hosts(State(state): State<Shared>, Query(params): Query<Params>) -> Response {
    let dir = or_fail!(session(&state, &params));
    let mut names: Vec<String> = xplanes(&dir).iter().map(|path| host_name(path)).collect();
    let tool = params.get("tag").map_or("", String::as_str);
    if names.len() > 1 && ALL_HOSTS_ONLY.contains(&tool) {
        names = vec!["ALL_HOSTS".into()];
    } else if names.len() > 1 && ALL_HOSTS_ALSO.contains(&tool) {
        names.push("ALL_HOSTS".into());
    }
    names.sort();
    names.dedup();
    response(StatusCode::OK, "application/json", format!("[{}]", names.iter().map(|name| format!("{{\"hostname\": {}}}", quoted(name))).collect::<Vec<_>>().join(", ")))
}

async fn data_csv(State(state): State<Shared>, Query(params): Query<Params>, uri: Uri, headers: HeaderMap) -> Response {
    let dir = or_fail!(session(&state, &params));
    let (shared, target) = (state.clone(), dir.clone());
    let render = async move {
        let reply = serve(shared, target, params.clone()).await;
        if reply.status() != StatusCode::OK {
            let status = reply.status();
            let body = axum::body::to_bytes(reply.into_body(), usize::MAX).await.unwrap_or_default();
            return match (status, body == NO_DATA.as_bytes()) {
                (StatusCode::NOT_FOUND, true) => response(status, "text/plain", "No Data Found"),
                (StatusCode::NOT_FOUND, false) => response(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", body),
                _ => response(status, "text/plain", body),
            };
        }
        if reply.headers().get(header::CONTENT_TYPE).is_none_or(|kind| kind != "application/json") {
            return response(StatusCode::BAD_REQUEST, "text/plain", "CSV format not supported for this tool type");
        }
        let body = axum::body::to_bytes(reply.into_body(), usize::MAX).await.unwrap_or_default();
        let parsed = serde_json::from_slice::<serde_json::Value>(&body).ok();
        let empty = parsed.as_ref().and_then(|value| value.as_array()).is_some_and(Vec::is_empty);
        let parsed = parsed.map(|value| if let Some(first) = value.get(0) { first.clone() } else { value });
        let Some(table) = parsed.filter(|table| table.get("cols").is_some() || empty) else {
            return response(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "Data format not suitable for CSV (missing 'cols')");
        };
        let quote = |text: &str| format!("\"{}\"", text.replace('"', "\"\""));
        let cell = |value: &serde_json::Value| match value {
            serde_json::Value::Number(number) if number.is_f64() => tools::table::repr(number.as_f64().unwrap()),
            serde_json::Value::Number(number) => number.to_string(),
            serde_json::Value::String(text) => text.clone(),
            serde_json::Value::Bool(flag) => if *flag { "True" } else { "False" }.into(),
            _ => String::new(),
        };
        let mut csv =
            table["cols"].as_array().map_or(String::new(), |cols| cols.iter().map(|col| quote(col["label"].as_str().or(col["id"].as_str()).unwrap_or(""))).collect::<Vec<_>>().join(",") + "\n");
        for row in table["rows"].as_array().into_iter().flatten() {
            let cells = row["c"].as_array().into_iter().flatten().map(|entry| quote(&cell(&entry["v"]))).collect::<Vec<_>>();
            if !cells.is_empty() {
                csv += &(cells.join(",") + "\n");
            }
        }
        let sanitize = |text: &str| text.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect::<String>();
        let host = params.get("host").and_then(|host| host.rsplit('-').next()).unwrap_or("");
        let name = [sanitize(params.get("tag").map_or("data", String::as_str)), sanitize(params.get("run").map_or("", String::as_str)), sanitize(host)]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        let mut reply = response(StatusCode::OK, "text/csv", csv);
        reply.headers_mut().insert(header::CONTENT_DISPOSITION, HeaderValue::from_str(&format!("attachment; filename=\"{name}.csv\"")).unwrap());
        reply
    };
    cached(&state, format!("csv?{}", uri.query().unwrap_or("")), &dir, accepts_gzip(&headers), render).await
}

async fn generate_cache(State(state): State<Shared>, method: Method, Query(params): Query<Params>) -> Response {
    if method != Method::POST {
        return response(StatusCode::METHOD_NOT_ALLOWED, "text/plain", "Method Not Allowed");
    }
    let Some(path) = params.get("session_path").filter(|path| !path.is_empty()) else { return response(StatusCode::BAD_REQUEST, "text/plain", "Missing \"session_path\" parameter") };
    let Some(dir) = confine(&state, &state.logdir.join(path)) else { return failure(outside()) };
    if xplanes(&dir).is_empty() {
        return response(StatusCode::NOT_FOUND, "text/plain", "No XPlane files found in session_path");
    }
    let run = Path::new(path).file_name().unwrap_or_default().to_string_lossy().into_owned();
    let tools_dir = dir.clone();
    let available: Vec<String> = serde_json::from_str(&blocking(move || server::run_tools::json(&tools_dir)).await).unwrap_or_default();
    let requested: Vec<String> = params
        .get("tools")
        .filter(|tools| !tools.is_empty())
        .map_or_else(|| CACHE_TOOLS.map(String::from).to_vec(), |tools| tools.split(',').map(|tool| tool.trim().to_string()).filter(|tool| !tool.is_empty()).collect());
    let mut tools: Vec<String> = requested.into_iter().filter(|tool| available.contains(tool) && XPLANE_TOOLS.contains(&tool.as_str())).collect();
    tools.sort();
    tools.dedup();
    if tools.is_empty() {
        return response(StatusCode::BAD_REQUEST, "text/plain", format!("No valid XPlane tools found or specified for caching in run {run}."));
    }
    tokio::spawn(async move {
        for tool in tools {
            let mut params = params.clone();
            params.insert("tag".into(), tool);
            params.insert("host".into(), "ALL_HOSTS".into());
            params.insert("run".into(), run.clone());
            serve(state.clone(), dir.clone(), params).await;
        }
    });
    response(StatusCode::ACCEPTED, "application/json", "{\"message\": \"Cache generation started\", \"status\": \"ACCEPTED\"}")
}

async fn runs(State(state): State<Shared>, Query(params): Query<Params>) -> Response {
    let logdir = state.logdir.clone();
    let mapped = or_fail!(session_map(&state, &params));
    let names: BTreeSet<String> =
        blocking(move || mapped.map_or_else(|| sessions(&logdir).into_iter().map(|(name, _)| name).collect(), |map| map.into_iter().map(|(name, _, _)| name).collect())).await;
    response(StatusCode::OK, "application/json", format!("[{}]", names.iter().rev().map(|name| quoted(name)).collect::<Vec<_>>().join(", ")))
}

fn select(dir: &Path, tool: &str, params: &Params) -> Result<Vec<PathBuf>, Failure> {
    let run = params.get("run").map_or("", String::as_str);
    let missing = |host: &str| Err((StatusCode::NOT_FOUND, format!("No xplane file found for host: {host} in run: {run}")));
    let (host, hosts) = (params.get("host").map_or("", String::as_str), params.get("hosts").map_or("", String::as_str));
    let mut files = xplanes(dir);
    files.sort();
    if files.is_empty() {
        return Err((StatusCode::NOT_FOUND, format!("No xplane file found for run: {run}, tool: {tool}")));
    }
    if !hosts.is_empty() && tool == "trace_viewer@" {
        let wanted: Vec<&str> = hosts.split(',').collect();
        if let Some(unknown) = wanted.iter().find(|name| !files.iter().any(|file| host_name(file) == **name)) {
            return missing(unknown);
        }
        files.retain(|file| wanted.contains(&host_name(file).as_str()));
        return Ok(files);
    }
    match host {
        "ALL_HOSTS" => Ok(files),
        "" if hosts.is_empty() => Ok(files),
        "" if ALL_HOSTS_ONLY.contains(&tool) => Err((StatusCode::NOT_FOUND, NO_DATA.into())),
        "" => Err((StatusCode::NOT_FOUND, format!("Host must be specified for tool {tool} in run {run}"))),
        host => {
            files.retain(|file| host_name(file) == host);
            if files.is_empty() { missing(host) } else { Ok(files) }
        }
    }
}

fn listed_hosts(dir: &Path, tool: &str, params: &Params) -> Result<Vec<PathBuf>, Failure> {
    let (mut paths, listed) = (select(dir, tool, params)?, xplanes(dir));
    paths.sort_by_key(|path| listed.iter().position(|item| item == path));
    Ok(paths)
}

async fn tool(state: &Shared, paths: Vec<PathBuf>, render: Render) -> Response {
    let all = match try_join_all(paths.iter().map(|path| state.stats.get(path.clone(), &state.loads, tools::opstats::load))).await {
        Ok(all) => all,
        Err(message) => return if blocking(move || tools::counters::corrupt(&paths)).await { not_found() } else { internal(&message) },
    };
    let Some(combined) = OpStats::combine(&all) else { return not_found() };
    let body = blocking(move || render(&combined)).await;
    if body.is_empty() { not_found() } else { response(StatusCode::OK, "application/json", body) }
}

async fn counter_tool(dir: &Path, tag: &str, params: &Params) -> Option<Response> {
    if params.contains_key("hosts") {
        return None;
    }
    let (paths, tag) = (listed_hosts(dir, tag, params).ok()?, tag.to_string());
    let body = tokio::task::spawn_blocking(move || tools::counters::serve(&tag, &paths)).await.ok()??;
    Some(response(StatusCode::OK, "application/json", body))
}

async fn module_list(State(state): State<Shared>, Query(params): Query<Params>) -> Response {
    let Some(run) = params.get("run").filter(|_| !params.contains_key("session_path") && !params.contains_key("run_path")) else { return not_found() };
    let Some(dir) = run_dir(&state, run) else { return failure(outside()) };
    let mut names = hlo::modules(&dir);
    let files = xplanes(&dir);
    if names.is_empty() && !files.is_empty() {
        let target = dir.clone();
        let extract = move || blocking(move || hlo::extracted(&target, &files).unwrap_or_default());
        names = state.extracts.join(dir, extract).await.unwrap_or_default();
    }
    names.retain(|name| !name.is_empty());
    names.sort();
    response(StatusCode::OK, "text/plain", names.join(","))
}

async fn data(State(state): State<Shared>, Query(params): Query<Params>, uri: Uri, headers: HeaderMap) -> Response {
    if params.get("tag").is_some_and(|tag| tag == "perf_counters") && params.get("names_only").is_some_and(|value| value == "1") {
        return match params.get("device_type").filter(|device_type| !device_type.is_empty()) {
            None => response(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "device_type is required for perf_counters with names_only"),
            Some(device_type) => tools::counter_ids::names(device_type).map_or_else(
                || response(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", format!("Unsupported device_type: {device_type}")),
                |names| response(StatusCode::OK, "application/json", names),
            ),
        };
    }
    let dir = or_fail!(session(&state, &params));
    cached(&state, format!("data?{}", uri.query().unwrap_or("")), &dir, accepts_gzip(&headers), serve(state.clone(), dir.clone(), params)).await
}

async fn serve(state: Shared, dir: PathBuf, params: Params) -> Response {
    let tag = params.get("tag").map_or("", String::as_str);
    let single = || select(&dir, tag, &params).is_ok_and(|files| files.len() == 1);
    let group_by = params.get("group_by").cloned();
    let renderer: Option<Render> = match tag {
        "hlo_stats" => Some(Box::new(tools::hlo_stats::json)),
        "kernel_stats" => Some(Box::new(xplane::gpu::kernel_stats_json)),
        "framework_op_stats" => Some(Box::new(tools::framework_op_stats::json)),
        "overview_page" => {
            let paths = listed_hosts(&dir, tag, &params).unwrap_or_default();
            Some(Box::new(move |stats: &OpStats| tools::overview_page::json(stats, &paths)))
        }
        "input_pipeline_analyzer" => Some(Box::new(tools::input_pipeline_analyzer::json)),
        "op_profile" => Some(Box::new(move |stats: &OpStats| tools::op_profile::json(stats, group_by.as_deref()))),
        "roofline_model" => Some(Box::new(tools::roofline::json)),
        "pod_viewer" => Some(Box::new(tools::pod_viewer::json)),
        "perf_counters" | "utilization_viewer" | "kernel_utilization" => return counter_tool(&dir, tag, &params).await.unwrap_or_else(not_found),
        "memory_viewer" | "graph_viewer" => {
            if params.get("module_name").is_some_and(|name| name.contains(['/', '\0'])) {
                return failure(outside());
            }
            or_fail!(select(&dir, tag, &params));
            let (memory, params) = (tag == "memory_viewer", params.clone());
            if memory {
                return blocking(move || hlo::memory::serve(&dir, &params)).await.map_or_else(not_found, |(body, content_type)| response(StatusCode::OK, content_type, body));
            }
            return blocking(move || hlo::graph::serve(&dir, &params)).await.map_or_else(|message| internal(&message), |(body, content_type)| response(StatusCode::OK, content_type, body));
        }
        "megascale_stats" => {
            let paths = or_fail!(listed_hosts(&dir, tag, &params));
            let Some(host) = params.get("host").filter(|host| !host.is_empty()).cloned() else { return not_found() };
            let perfetto = params.get("perfetto").is_some_and(|value| value.eq_ignore_ascii_case("true"));
            let own = paths.iter().find(|path| path.file_name().is_some_and(|name| name.to_string_lossy() == format!("{host}.xplane.pb"))).cloned().filter(|_| perfetto);
            let body = blocking(move || match own {
                Some(path) => {
                    let map = read_file(&path).ok()?;
                    if tools::counters::valid_space(&map) { tools::megascale_perfetto::render(&map) } else { None }
                }
                None => tools::megascale::json(&paths, &host).map(String::into_bytes),
            });
            return body.await.map_or_else(not_found, |body| response(StatusCode::OK, if perfetto { "application/octet-stream" } else { "application/json" }, body));
        }
        "memory_profile" if single() => {
            let paths = or_fail!(listed_hosts(&dir, tag, &params));
            let path = paths[0].clone();
            let _permit = state.loads.clone().acquire_owned().await;
            let outcome = match Stamp::of(&path) {
                None => Err(format!("Cannot read {}", path.display())),
                Some(_) => blocking(move || tools::memory_profile::load(&path)).await.map_err(|error| error.to_string()),
            };
            return match outcome {
                Ok(Some(body)) => response(StatusCode::OK, "application/json", body),
                Ok(None) => not_found(),
                Err(message) => {
                    if blocking(move || tools::counters::corrupt(&paths)).await {
                        not_found()
                    } else {
                        internal(&message)
                    }
                }
            };
        }
        "inference_profile" | "smart_suggestion" => {
            let paths = or_fail!(listed_hosts(&dir, tag, &params));
            let smart = tag == "smart_suggestion";
            let body = blocking(move || if smart { tools::smart_suggestion::json(&paths) } else { tools::inference_profile::json(&paths) }).await;
            return body.map_or_else(not_found, |body| response(StatusCode::OK, "application/json", body));
        }
        _ => None,
    };
    if let Some(renderer) = renderer {
        return tool(&state, or_fail!(listed_hosts(&dir, tag, &params)), renderer).await;
    }
    if tag == "trace_viewer" {
        return match select(&dir, tag, &params).map(<[PathBuf; 1]>::try_from) {
            Ok(Ok([file])) if tools::counters::corrupt(std::slice::from_ref(&file)) => not_found(),
            Ok(Ok([file])) if params.get("format").is_some_and(|format| format == "pb") && !params.contains_key("event_name") => match state.hosts.get(file, &state.loads, shared_host).await {
                Ok(host) => {
                    let body = blocking(move || {
                        let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma: true };
                        trace::delta::render(&[View { trace: &host.trace, map: &host.map, planes: &host.planes, events: host.trace.load(&options) }], None)
                    });
                    response(StatusCode::OK, "application/octet-stream", body.await)
                }
                Err(message) => internal(&message),
            },
            Ok(Ok([file])) => blocking(move || {
                let (map, planes) = prepare(&file, true)?;
                let json = trace::legacy::render(&planes, &map);
                release((map, planes));
                anyhow::Ok(json)
            })
            .await
            .map_or_else(|error| internal(&error.to_string()), |json| response(StatusCode::OK, "application/json", json)),
            Ok(Err(_)) => not_found(),
            Err(error) => failure(error),
        };
    }
    if tag != "trace_viewer@" {
        return not_found();
    }
    let files = or_fail!(select(&dir, tag, &params));
    let hosts = match try_join_all(files.iter().map(|file| state.hosts.get(file.clone(), &state.loads, shared_host))).await {
        Ok(hosts) => hosts,
        Err(message) => return if blocking(move || tools::counters::corrupt(&files)).await { not_found() } else { internal(&message) },
    };
    let option = |key: &str, default: f64| params.get(key).map_or(Some(default), |value| value.trim().parse::<f64>().ok());
    let (Some(start_ms), Some(end_ms), Some(resolution)) = (option("start_time_ms", 0.0), option("end_time_ms", 0.0), option("resolution", DEFAULT_RESOLUTION)) else {
        return not_found();
    };
    let options = Options { start_ms, end_ms, resolution: resolution.trunc(), full_dma: params.get("full_dma").is_some_and(|value| value.eq_ignore_ascii_case("true")) };
    let query = params.get("search_prefix").filter(|prefix| !prefix.is_empty()).cloned();
    let (Some(duration_ms), Some(unique_id)) = (option("duration_ms", 0.0), option("unique_id", 0.0)) else { return not_found() };
    let detail = params.get("event_name").map(|name| (name.clone(), (start_ms * 1e9).round() as u64, (duration_ms * 1e9).round() as u64, unique_id as usize));
    let protobuf = detail.is_none() && params.get("format").is_some_and(|format| format == "pb");
    let with_args = detail.is_some() || (query.is_some() && params.get("search_metadata").is_some_and(|value| value.eq_ignore_ascii_case("true")));
    let body = blocking(move || {
        let views: Vec<View> = hosts
            .iter()
            .map(|host| {
                let trace = &host.trace;
                let events = match (&detail, &query) {
                    (Some((name, ts, dur, serial)), _) => {
                        let index = trace.events.partition_point(|event| event.ts < *ts).saturating_add(*serial);
                        trace
                            .events
                            .get(index)
                            .filter(|event| event.serial < MAX_SERIAL && event.ts == *ts && event.dur == *dur && *trace.names[event.name as usize] == **name)
                            .map(|_| index as u32)
                            .into_iter()
                            .collect()
                    }
                    (None, Some(prefix)) => trace.search(prefix, options.full_dma),
                    (None, None) => trace.load(&options),
                };
                View { trace, map: &host.map, planes: &host.planes, events }
            })
            .collect();
        (detail.is_none() || views.iter().all(|view| !view.events.is_empty()))
            .then(|| if protobuf { trace::delta::render(&views, Some(options.full_dma)) } else { render(&views, options.full_dma, with_args) })
    })
    .await;
    body.map_or_else(not_found, |body| response(StatusCode::OK, if protobuf { "application/octet-stream" } else { "application/json" }, body))
}

static STARTED: LazyLock<Instant> = LazyLock::new(Instant::now);
/// The number of requests in progress, and the end of the last request in milliseconds after `STARTED`.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static LAST: AtomicU64 = AtomicU64::new(0);

struct Active;

impl Drop for Active {
    fn drop(&mut self) {
        LAST.store(STARTED.elapsed().as_millis() as u64, Ordering::Relaxed);
        ACTIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn track(request: axum::extract::Request, next: axum::middleware::Next) -> Response {
    ACTIVE.fetch_add(1, Ordering::Relaxed);
    let _active = Active;
    next.run(request).await
}

async fn quiet() {
    while ACTIVE.load(Ordering::Relaxed) > 0 || (STARTED.elapsed().as_millis() as u64).saturating_sub(LAST.load(Ordering::Relaxed)) < QUIET.as_millis() as u64 {
        tokio::time::sleep(QUIET / 10).await;
    }
}

async fn prefetch(state: Shared) {
    let (permits, mut attempted) = (Arc::new(Semaphore::new(CONCURRENT_PREFETCHES)), Vec::new());
    loop {
        if let Some(remote) = &state.remote {
            let mut used: Vec<(PathBuf, Instant)> = remote.used.lock().unwrap().clone().into_iter().filter(|(dir, _)| !xplanes(dir).is_empty()).collect();
            used.sort_by_key(|(_, at)| std::cmp::Reverse(*at));
            for dir in std::iter::once(remote.mirror.clone()).chain(used.into_iter().take(WATCHED_SESSIONS).map(|(dir, _)| dir)) {
                _ = remote.sync(&dir, false).await.inspect_err(|message| eprintln!("{message}"));
            }
        }
        let logdir = state.logdir.clone();
        let mut found = blocking(move || sessions(&logdir)).await;
        found.sort_by_key(|(_, dir)| std::cmp::Reverse(std::fs::metadata(dir).and_then(|meta| meta.modified()).unwrap_or(SystemTime::UNIX_EPOCH)));
        let watched: Vec<(PathBuf, Stamp)> = found
            .into_iter()
            .take(WATCHED_SESSIONS)
            .flat_map(|(_, dir)| xplanes(&dir))
            .filter_map(|file| Stamp::of(&file).map(|stamp| (file, stamp)))
            .filter(|(_, stamp)| stamp.len > 0 && stamp.len < (*MEMORY / 32).min(*BUDGET_BYTES) && stamp.older_than(POLL))
            .collect();
        attempted.retain(|key| watched.contains(key));
        for key in watched {
            if !attempted.contains(&key) {
                attempted.push(key.clone());
                quiet().await;
                _ = state.stats.get(key.0.clone(), &permits, tools::opstats::load).await;
                quiet().await;
                _ = state.hosts.get(key.0, &permits, shared_host).await;
            }
        }
        tokio::time::sleep(POLL).await;
    }
}

fn arguments(list: impl IntoIterator<Item = String>) -> Result<Settings, String> {
    let (mut list, mut values, mut switches) = (list.into_iter().peekable(), HashMap::new(), Vec::new());
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    let negative = |text: &str| text.strip_prefix('-').map(|rest| rest.split_once('.').unwrap_or(("", rest))).is_some_and(|(whole, part)| digits(whole) && !part.is_empty() && digits(part));
    list.next_if(|first| first == "server");
    while let Some(argument) = list.next() {
        let (flag, inline) = match argument.split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (argument.clone(), None),
        };
        if SWITCHES.contains(&flag.as_str()) && inline.is_none() {
            switches.push(flag);
            continue;
        }
        if !VALUE_FLAGS.contains(&flag.as_str()) {
            return Err(format!("unrecognized argument: {argument}"));
        }
        let value =
            inline.or_else(|| list.next_if(|next| !next.starts_with('-') || next == "-" || next.contains(' ') || negative(next))).ok_or_else(|| format!("argument {flag}: expected one argument"))?;
        values.insert(flag, value);
    }
    let remote = values.get("--logdir").filter(|logdir| logdir.contains("://")).map(|url| Remote::open(url).map(Arc::new)).transpose()?;
    let logdir = match values.remove("--logdir").filter(|logdir| !logdir.is_empty()) {
        Some(_) if let Some(remote) = &remote => remote.mirror.clone(),
        Some(logdir) => {
            let expanded = cli::expand(&logdir);
            let absolute = std::path::absolute(&expanded).unwrap_or(expanded);
            absolute.canonicalize().ok().filter(|dir| dir.is_dir()).ok_or_else(|| format!("Log directory '{}' does not exist or is not a directory.", absolute.display()))?
        }
        None => PathBuf::new(),
    };
    let mut number = |flag: &str, default: u16| values.remove(flag).map_or(Ok(default), |value| value.parse().map_err(|_| format!("argument {flag}: invalid port value: '{value}'")));
    let port = number("--port", DEFAULT_PORT)?;
    if number("--grpc_port", DEFAULT_GRPC_PORT)? == port {
        return Err("The main server port (--port) and the gRPC port (--grpc_port) must be different.".into());
    }
    if let Some(value) = values.get("--max_concurrent_worker_requests").filter(|value| value.parse::<i64>().is_err()) {
        return Err(format!("argument --max_concurrent_worker_requests: invalid int value: '{value}'"));
    }
    Ok(Settings {
        logdir,
        port,
        host: values.remove("--host"),
        src_prefix: values.remove("--src_prefix"),
        hide_capture_profile_button: switches.contains(&"--hide_capture_profile_button".to_string()),
        enable_tab_name_label: switches.contains(&"--enable_tab_name_label".to_string()),
        remote,
    })
}

fn state(settings: &Settings) -> Shared {
    let config = serde_json::json!({"enableTabNameLabel": settings.enable_tab_name_label, "hideCaptureProfileButton": settings.hide_capture_profile_button, "srcPathPrefix": settings.src_prefix});
    Arc::new(State_ {
        logdir: settings.logdir.canonicalize().unwrap_or_else(|_| settings.logdir.clone()),
        remote: settings.remote.clone(),
        config: server::run_tools::python_value(&config),
        hosts: Memo::new(|_, host: &Arc<Host>| host.bytes),
        stats: Memo::new(|stamp, _| stamp.len),
        rendered: cache(*BUDGET_BYTES / 4, |key: &String, value: &Rendered| (key.len() + value.body.len()) as u64),
        renders: Flights::new(),
        extracts: Flights::new(),
        loads: Arc::new(Semaphore::new(((*MEMORY >> 34) as usize).clamp(1, 2))),
    })
}

fn plugin() -> Router<Shared> {
    Router::new()
        .route("/data", any(data))
        .route("/runs", any(runs))
        .route("/run_tools", any(run_tools))
        .route("/hosts", any(hosts))
        .route("/data_csv", any(data_csv))
        .route("/version", any(|| async { response(StatusCode::OK, "text/plain", VERSION) }))
        .route("/config", any(|State(state): State<Shared>| async move { response(StatusCode::OK, "application/json", state.config.clone()) }))
        .route("/module_list", any(module_list))
        .route("/generate_cache", any(generate_cache))
        .route("/capture_profile", any(|State(state): State<Shared>, Query(params): Query<Params>| async move { server::capture::handle(&state.logdir, state.remote.as_deref(), &params).await }))
}

async fn first_values(mut request: Request) -> Request {
    if let Some(query) = request.uri().query() {
        let mut seen = BTreeSet::new();
        let kept: Vec<&str> = query.split('&').filter(|pair| seen.insert(form_urlencoded::parse(pair.as_bytes()).next().map(|(key, _)| key.into_owned()))).collect();
        let target = format!("{}?{}", request.uri().path(), kept.join("&"));
        if let Ok(uri) = target.parse() {
            *request.uri_mut() = uri;
        }
    }
    request
}

fn app(state: Shared) -> Router {
    Router::new()
        .merge(plugin())
        .nest(PREFIX, plugin())
        .fallback(assets)
        .layer(axum::middleware::from_fn_with_state(state.clone(), server::remote::mirror))
        .layer(axum::middleware::map_request(first_values))
        .layer(CatchPanicLayer::custom(|payload: Box<dyn Any + Send>| internal(&panic_message(&*payload))))
        .layer(axum::middleware::map_response(|mut reply: Response| async move {
            reply.headers_mut().insert(HeaderName::from_static("content-security-policy"), HeaderValue::from_static(SECURITY_POLICY));
            reply.headers_mut().insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
            reply
        }))
        .layer(CompressionLayer::new().quality(CompressionLevel::Fastest))
        .layer(axum::middleware::from_fn(track))
        .with_state(state)
}

fn main() -> anyhow::Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from).min(MAX_THREADS);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let purge_delay = unsafe { libmimalloc_sys::mi_option_get(PURGE_DELAY) };
    // A command exits right after its output. To give freed memory back to the system before the exit only costs time.
    unsafe { libmimalloc_sys::mi_option_set(PURGE_DELAY, -1) };
    let argv: Vec<String> = std::env::args_os().skip(1).map(|argument| argument.to_string_lossy().into_owned()).collect();
    if let Some(code) = cli::run(&argv) {
        std::process::exit(code);
    }
    unsafe { libmimalloc_sys::mi_option_set(PURGE_DELAY, purge_delay) };
    if argv.iter().any(|argument| argument == "--help" || argument == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let settings = arguments(argv).unwrap_or_else(|error| {
        eprintln!("{USAGE}\nxprof-rs: error: {error}");
        std::process::exit(2);
    });
    tokio::runtime::Builder::new_multi_thread().worker_threads(threads).enable_all().build()?.block_on(listen(settings))
}

async fn listen(settings: Settings) -> anyhow::Result<()> {
    let Settings { port, host, .. } = &settings;
    let host = host.as_deref().unwrap_or("127.0.0.1");
    let listener = tokio::net::TcpListener::bind((host, *port)).await.with_context(|| format!("Cannot listen on {host}:{port}"))?;
    eprintln!(
        "xprof-rs runs at http://localhost:{port}/ and serves {}",
        settings.remote.as_ref().map_or_else(|| settings.logdir.display().to_string(), |remote| format!("{} (copy in {})", remote.url, remote.mirror.display()))
    );
    let state = state(&settings);
    tokio::spawn(prefetch(state.clone()));
    axum::serve(listener, app(state)).await?;
    Ok(())
}
