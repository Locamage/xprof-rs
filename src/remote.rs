use crate::{Params, RENDERED_FROM, Shared, blocking, confine, list, response, run_dir, sessions};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use futures_util::future::try_join_all;
use futures_util::{StreamExt, TryStreamExt, stream};
use object_store::path::{Path as Key, PathPart};
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, PutPayload};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

const RECHECK: Duration = Duration::from_secs(5);
const IN_USE: Duration = Duration::from_secs(60);
const LIST_TIMEOUT: Duration = Duration::from_secs(30);
const PART_BYTES: u64 = 8 << 20;
const PARALLEL_FILES: usize = 4;
const PARALLEL_PARTS: usize = 8;
const DEFAULT_CACHE_BYTES: u64 = 20 << 30;
const CACHE_DIR: &str = "XPROF_CACHE_DIR";
const CACHE_BYTES: &str = "XPROF_CACHE_BYTES";
const SESSION_PARAMS: [&str; 3] = ["run", "session_path", "run_path"];

#[derive(Debug)]
pub struct Remote {
    pub url: String,
    pub mirror: PathBuf,
    pub budget: u64,
    pub used: Mutex<HashMap<PathBuf, Instant>>,
    store: Arc<dyn ObjectStore>,
    root: Key,
    checked: Mutex<HashMap<PathBuf, Instant>>,
    locks: Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
}

impl PartialEq for Remote {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
    }
}

impl Remote {
    pub fn open(url: &str) -> Result<Remote, String> {
        let parsed = url::Url::parse(url).map_err(|error| format!("The log directory '{url}' is not a valid URL: {error}"))?;
        let (store, root) = object_store::parse_url_opts(&parsed, std::env::vars()).map_err(|error| format!("Cannot open {url}: {error}"))?;
        Remote::new(url, Arc::from(store), root)
    }

    pub fn new(url: &str, store: Arc<dyn ObjectStore>, root: Key) -> Result<Remote, String> {
        let digest: String = Sha256::digest(url.as_bytes()).iter().take(8).map(|byte| format!("{byte:02x}")).collect();
        #[cfg(not(test))]
        let temp_dir = std::env::temp_dir;
        #[cfg(test)]
        let temp_dir = crate::tests::temp_dir;
        let mirror = std::env::var_os(CACHE_DIR).map_or_else(temp_dir, PathBuf::from).join(format!("xprof-rs-{digest}"));
        let mirror = std::fs::create_dir_all(&mirror).and_then(|_| mirror.canonicalize()).map_err(|error| format!("Cannot create the mirror {}: {error}", mirror.display()))?;
        let budget = std::env::var(CACHE_BYTES).ok().and_then(|bytes| bytes.parse().ok()).unwrap_or(DEFAULT_CACHE_BYTES);
        Ok(Remote { url: url.to_string(), mirror, budget, used: Mutex::default(), store, root, checked: Mutex::default(), locks: Mutex::default() })
    }

    fn key(&self, dir: &Path) -> Option<Key> {
        dir.strip_prefix(&self.mirror).ok()?.components().try_fold(self.root.clone(), |key, part| match part {
            Component::Normal(name) => Some(key.join(PathPart::parse(name.to_str()?).ok()?)),
            _ => None,
        })
    }

    fn local(&self, key: &Key) -> Option<PathBuf> {
        key.prefix_match(&self.root)?.try_fold(self.mirror.clone(), |dir, part| (!matches!(part.as_ref(), "" | "." | "..")).then(|| dir.join(part.as_ref())))
    }

    pub async fn sync(&self, dir: &Path, touch: bool) -> Result<(), String> {
        let Some(key) = self.key(dir) else { return Ok(()) };
        if touch {
            self.used.lock().unwrap().insert(dir.to_path_buf(), Instant::now());
        }
        let lock = self.locks.lock().unwrap().entry(dir.to_path_buf()).or_default().clone();
        let _held = lock.lock().await;
        if self.checked.lock().unwrap().get(dir).is_some_and(|at| at.elapsed() < RECHECK) {
            return Ok(());
        }
        let unreadable = |error: String| format!("Cannot read {}: {error}", self.url);
        let unwritable = |error: std::io::Error| format!("Cannot write the mirror of {} in {}: {error}", self.url, self.mirror.display());
        let listing = async |prefix: &Key| match tokio::time::timeout(LIST_TIMEOUT, self.store.list_with_delimiter(Some(prefix))).await {
            Ok(listed) => listed.map_err(|error| unreadable(error.to_string())),
            Err(_) => Err(unreadable(format!("No answer in {LIST_TIMEOUT:?}"))),
        };
        if dir == self.mirror {
            let mut pending = vec![key];
            while let Some(prefix) = pending.pop() {
                for child in listing(&prefix).await?.common_prefixes {
                    if child.filename() != Some("plugins") {
                        pending.push(child);
                        continue;
                    }
                    for session in listing(&child.clone().join("profile")).await?.common_prefixes {
                        self.local(&session).map_or(Ok(()), std::fs::create_dir_all).map_err(unwritable)?;
                    }
                }
            }
        } else {
            let unchanged =
                |path: &Path, object: &ObjectMeta| std::fs::metadata(path).is_ok_and(|meta| meta.len() == object.size && meta.modified().ok() == Some(SystemTime::from(object.last_modified)));
            let wanted: Vec<(PathBuf, ObjectMeta)> = listing(&key)
                .await?
                .objects
                .into_iter()
                .filter(|object| object.location.filename().is_some_and(|name| RENDERED_FROM.iter().any(|suffix| name.ends_with(suffix))))
                .filter_map(|object| Some((self.local(&object.location)?, object)))
                .filter(|(path, object)| !unchanged(path, object))
                .collect();
            if !wanted.is_empty() {
                std::fs::create_dir_all(dir).map_err(unwritable)?;
            }
            let fetch = async |path: PathBuf, object: ObjectMeta| {
                let partial = path.with_file_name(format!(".{}.partial", object.location.filename().unwrap_or_default()));
                let file = Arc::new(std::fs::File::create(&partial).and_then(|file| file.set_len(object.size).map(|_| file)).map_err(unwritable)?);
                stream::iter((0..object.size).step_by(PART_BYTES as usize))
                    .map(|start| {
                        let (file, location) = (file.clone(), &object.location);
                        async move {
                            let bytes = self.store.get_range(location, start..(start + PART_BYTES).min(object.size)).await.map_err(|error| unreadable(error.to_string()))?;
                            blocking(move || file.write_all_at(&bytes, start)).await.map_err(unwritable)
                        }
                    })
                    .buffer_unordered(PARALLEL_PARTS)
                    .try_collect::<Vec<()>>()
                    .await?;
                file.set_modified(SystemTime::from(object.last_modified)).and_then(|_| std::fs::rename(&partial, &path)).map_err(unwritable)
            };
            stream::iter(wanted).map(|(path, object)| fetch(path, object)).buffer_unordered(PARALLEL_FILES).try_collect::<Vec<()>>().await?;
            let used = self.used.lock().unwrap().clone();
            let is_file = |entry: &std::fs::DirEntry| entry.file_type().is_ok_and(|kind| kind.is_file());
            let mut held: Vec<_> = sessions(&self.mirror)
                .into_iter()
                .map(|(_, session)| {
                    let files: Vec<(u64, PathBuf)> = list(&session, is_file).into_iter().filter_map(|file| Some((file.metadata().ok()?.len(), file))).collect();
                    (used.get(&session).copied(), session, files)
                })
                .collect();
            let mut total: u64 = held.iter().flat_map(|(_, _, files)| files).map(|(len, _)| len).sum();
            held.sort_by_key(|(at, _, _)| *at);
            for (_, session, files) in held.into_iter().filter(|(at, session, _)| session != dir && at.is_none_or(|at| at.elapsed() >= IN_USE)) {
                if total <= self.budget {
                    break;
                }
                for (len, file) in files {
                    total -= if std::fs::remove_file(file).is_ok() { len } else { 0 };
                }
                self.checked.lock().unwrap().remove(&session);
            }
        }
        self.checked.lock().unwrap().insert(dir.to_path_buf(), Instant::now());
        Ok(())
    }

    pub async fn upload(&self, dir: &Path) -> Result<(), String> {
        let Some(key) = self.key(dir) else { return Ok(()) };
        for file in list(dir, |entry| entry.file_type().is_ok_and(|kind| kind.is_file())) {
            let failed = |error: String| format!("Cannot upload {} to {}: {error}", file.display(), self.url);
            let name = file.file_name().and_then(|name| name.to_str()).ok_or_else(|| failed("The file name is not UTF-8".into()))?;
            let location = key.clone().join(PathPart::parse(name).map_err(|error| failed(error.to_string()))?);
            let bytes = std::fs::read(&file).map_err(|error| failed(error.to_string()))?;
            self.store.put(&location, PutPayload::from(bytes)).await.map_err(|error| failed(error.to_string()))?;
            let stored = self.store.head(&location).await.map_err(|error| failed(error.to_string()))?;
            std::fs::File::open(&file).and_then(|opened| opened.set_modified(SystemTime::from(stored.last_modified))).map_err(|error| failed(error.to_string()))?;
        }
        Ok(())
    }
}

pub async fn mirror(State(state): State<Shared>, request: Request, next: Next) -> Response {
    let Some(remote) = state.remote.clone() else { return next.run(request).await };
    let params: Params = form_urlencoded::parse(request.uri().query().unwrap_or_default().as_bytes()).into_owned().collect();
    let named = |key: &str| params.get(key).filter(|value| !value.is_empty());
    if !request.uri().path().ends_with("/runs") && SESSION_PARAMS.iter().all(|key| named(key).is_none()) {
        return next.run(request).await;
    }
    if let Err(message) = remote.sync(&remote.mirror, false).await {
        return response(StatusCode::BAD_GATEWAY, "text/plain", message);
    }
    let dirs: Vec<PathBuf> = match (named("session_path"), named("run_path"), named("run")) {
        (Some(path), _, _) => vec![state.logdir.join(path)],
        (None, Some(path), Some(run)) => vec![state.logdir.join(path).join(run)],
        (None, Some(path), None) => list(&state.logdir.join(path), |entry| entry.file_type().is_ok_and(|kind| kind.is_dir())),
        (None, None, run) => run.and_then(|run| run_dir(&state, run)).into_iter().collect(),
    };
    match try_join_all(dirs.into_iter().filter_map(|dir| confine(&state, dir)).map(async |dir| remote.sync(&dir, true).await)).await {
        Ok(_) => next.run(request).await,
        Err(message) => response(StatusCode::BAD_GATEWAY, "text/plain", message),
    }
}

#[cfg(test)]
#[path = "tests/inline/remote.rs"]
mod tests;
