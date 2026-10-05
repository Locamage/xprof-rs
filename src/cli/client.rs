use super::{Error, Kind, fail, json::py_repr};
use crate::tools::opstats::{Kept, OpStats, load_kept};
use crate::xplane::Plane;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// A parameter of the local client only. The table then has the first row only.
const TOTAL_ONLY: &str = "total_only";
const KNOWN_TOOLS: [&str; 20] = [
    "overview_page",
    "input_pipeline_analyzer",
    "framework_op_stats",
    "kernel_stats",
    "memory_profile",
    "pod_viewer",
    "op_profile",
    "hlo_op_profile",
    "hlo_stats",
    "roofline_model",
    "graph_viewer",
    "memory_viewer",
    "megascale_stats",
    "inference_profile",
    "perf_counters",
    "utilization_viewer",
    "kernel_utilization",
    "smart_suggestion",
    "trace_viewer",
    "trace_viewer@",
];
pub const TRACE_SUFFIXES: [&str; 2] = [".xplane.pb", ".xspace.pb"];

pub type Params<'a> = [(&'a str, String)];

pub trait Client {
    fn fetch(&self, tool: &str, session: &str, params: &Params) -> Result<Option<Vec<u8>>, Error>;
    fn run_dir(&self, session: &str) -> Result<PathBuf, Error>;
    fn logdir(&self) -> Option<&Path>;

    /// The client as one that threads can share, when it is one.
    fn shared(&self) -> Option<&(dyn Client + Sync)> {
        None
    }

    fn xspace_paths(&self, dir: &Path) -> Result<Vec<PathBuf>, Error> {
        let found = traces(dir);
        if found.is_empty() {
            return fail(Kind::FileNotFound, format!("No .xplane.pb or .xspace.pb files found in {}", dir.display()));
        }
        Ok(found)
    }

    fn hosts(&self, session: &str) -> Result<Vec<String>, Error> {
        let paths = self.xspace_paths(&self.run_dir(session)?)?;
        let mut hosts: Vec<String> = paths.iter().map(|path| host(path)).collect();
        hosts.sort();
        hosts.dedup();
        Ok(hosts)
    }

    fn barrier_durations(&self, session: &str) -> Option<Vec<f64>> {
        let mut params = vec![("trace_filter_config", r#"{"device_regexes": ["TPU:0", "GPU:0"]}"#.to_string())];
        params.extend(self.hosts(session).ok()?.first().map(|host| ("hosts", host.clone())));
        let trace = serde_json::from_slice::<serde_json::Value>(&self.fetch("trace_viewer.json", session, &params).ok().flatten()?).ok()?;
        let events = trace.get("traceEvents")?.as_array()?;
        Some(
            events
                .iter()
                .filter(|event| event.get("name").and_then(|name| name.as_str()) == Some("barrier-cores"))
                .map(|event| event.get("dur").and_then(serde_json::Value::as_f64).unwrap_or(0.0))
                .collect(),
        )
    }

    /// The roofline table, of which the overview reads the first row only.
    fn roofline_total(&self, session: &str) -> Option<String> {
        self.fetch_text("roofline_model.json", session, &[]).ok()?.or_else(|| self.fetch_text("roofline_model", session, &[]).ok()?)
    }

    fn fetch_text(&self, tool: &str, session: &str, params: &Params) -> Result<Option<String>, Error> {
        Ok(self.fetch(tool, session, params)?.filter(|data| !data.is_empty()).map(|data| String::from_utf8_lossy(&data).into_owned()))
    }

    fn fetch_either(&self, first: &str, second: &str, session: &str, params: &Params) -> Result<Option<String>, Error> {
        match self.fetch_text(first, session, params)? {
            Some(data) => Ok(Some(data)),
            None => self.fetch_text(second, session, params),
        }
    }
}

pub fn host(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    name.strip_suffix(".xplane.pb").unwrap_or(&name).rsplit('.').next().unwrap_or_default().to_string()
}

pub fn find(dir: &Path, suffixes: &[&str], recursive: bool) -> Vec<PathBuf> {
    let (mut found, mut pending) = (Vec::new(), vec![dir.to_path_buf()]);
    while let Some(next) = pending.pop() {
        for path in std::fs::read_dir(&next).into_iter().flatten().flatten().map(|entry| entry.path()) {
            if path.is_dir() {
                pending.extend(recursive.then_some(path));
            } else if suffixes.iter().any(|suffix| path.file_name().is_some_and(|name| name.to_string_lossy().ends_with(suffix))) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

pub fn traces(dir: &Path) -> Vec<PathBuf> {
    if dir.is_file() {
        return if TRACE_SUFFIXES.iter().any(|suffix| dir.to_string_lossy().ends_with(suffix)) { vec![dir.to_path_buf()] } else { Vec::new() };
    }
    find(dir, &TRACE_SUFFIXES, true)
}

pub fn kernel_utilization<'a>(path: &Path, option: impl Fn(&str) -> Option<&'a str>) -> Option<String> {
    let map = crate::read_file(path).ok().filter(|map| crate::tools::counters::valid_space(map))?;
    let filter = crate::tools::counters::Filter {
        kernel: option("kernel").or(option("kernel_name")).unwrap_or_default().to_string(),
        duration_us: option("duration_us").and_then(|value| value.trim().parse().ok()).unwrap_or(0.0),
        force: option("force_duration").is_some_and(|value| matches!(value, "True" | "true" | "1" | "yes" | "t" | "y")),
        device: option("device_id").and_then(|value| value.trim().parse().ok()).unwrap_or(-1),
    };
    Some(crate::tools::counters::kernel_utilization(&map, &filter))
}

pub fn temp_dir() -> PathBuf {
    ["TMPDIR", "TEMP", "TMP"].iter().filter_map(std::env::var_os).map(PathBuf::from).find(|path| path.is_dir()).unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn latest_run(dir: &Path) -> Option<PathBuf> {
    let mut runs: Vec<PathBuf> = std::fs::read_dir(dir.join("plugins/profile")).ok()?.flatten().map(|entry| entry.path()).filter(|path| path.is_dir()).collect();
    runs.sort();
    runs.pop()
}

#[derive(Default)]
pub struct Local {
    pub logdir: Option<PathBuf>,
    /// Loads traces with the fused children at once, for a command that needs them after other data.
    pub fused: bool,
    pub loaded: Mutex<HashMap<PathBuf, Option<Arc<OpStats>>>>,
    pub kept: RwLock<HashMap<PathBuf, Kept>>,
}

impl Local {
    /// Reads the prepared planes of a file. It reuses the planes of the op statistics when no GPU plane needs the trace derivation.
    fn prepared<T>(&self, path: &Path, read: impl FnOnce(&[Plane], &[u8]) -> T) -> Option<T> {
        if let Some(kept) = self.kept.read().unwrap().get(path).filter(|kept| !kept.planes.iter().any(|plane| plane.name.starts_with(crate::xplane::gpu::PREFIX))) {
            return Some(read(&kept.planes, kept.map()));
        }
        let (map, planes) = crate::prepare(path, true).ok()?;
        Some(read(&planes, &map))
    }
}

impl Client for Local {
    fn logdir(&self) -> Option<&Path> {
        self.logdir.as_deref()
    }

    fn shared(&self) -> Option<&(dyn Client + Sync)> {
        Some(self)
    }

    fn roofline_total(&self, session: &str) -> Option<String> {
        self.fetch_text("roofline_model.json", session, &[(TOTAL_ONLY, String::new())]).ok()?.or_else(|| self.fetch_text("roofline_model", session, &[]).ok()?)
    }

    fn barrier_durations(&self, session: &str) -> Option<Vec<f64>> {
        let [path] = <[PathBuf; 1]>::try_from(self.xspace_paths(&self.run_dir(session).ok()?).ok()?).ok()?;
        self.prepared(&path, crate::trace::legacy::barrier_durations)
    }

    fn run_dir(&self, session: &str) -> Result<PathBuf, Error> {
        if !session.is_empty() {
            let path = super::expand(session);
            if path.is_file() {
                return Ok(path.parent().map(Path::to_path_buf).unwrap_or_default());
            }
            if path.is_dir() {
                return Ok(latest_run(&path).unwrap_or(path));
            }
        }
        let Some(logdir) = &self.logdir else {
            return if session.is_empty() { fail(Kind::Value, "Logdir not set. Please configure logdir first.") } else { fail(Kind::FileNotFound, format!("Path not found: {session}")) };
        };
        if session.is_empty() {
            return Ok(latest_run(logdir).unwrap_or_else(|| logdir.clone()));
        }
        let run = logdir.join("plugins/profile").join(session);
        if run.exists() {
            return Ok(run);
        }
        let formatted = (session.len() == 14 && session.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| logdir.join("plugins/profile").join(format!("{}_{}_{}_{}_{}_{}", &session[..4], &session[4..6], &session[6..8], &session[8..10], &session[10..12], &session[12..])))
            .filter(|path| path.exists());
        let fallback = Some(logdir.join(session)).filter(|path| path.exists());
        formatted.or(fallback).map_or_else(|| fail(Kind::FileNotFound, format!("Run directory not found for session {} in {}", py_repr(session), logdir.display())), Ok)
    }

    fn fetch(&self, tool: &str, session: &str, params: &Params) -> Result<Option<Vec<u8>>, Error> {
        let name = tool.strip_suffix(".json").unwrap_or(tool);
        let name = if name == "hlo_op_profile" { "op_profile" } else { name };
        if !KNOWN_TOOLS.contains(&name) {
            return fail(Kind::Value, format!("Unknown XProf tool name: {}", py_repr(tool)));
        }
        let paths = self.xspace_paths(&self.run_dir(session)?)?;
        let flag = |key: &str, value: &str| if matches!(key, "show_metadata" | "merge_fusion") { value.trim().parse::<i64>().is_ok_and(|number| number != 0).to_string() } else { value.to_string() };
        let options: HashMap<String, String> = params.iter().map(|(key, value)| (key.to_string(), flag(key, value))).collect();
        let option = |key: &str| options.get(key).map(String::as_str);
        let dir = paths[0].parent().map(Path::to_path_buf).unwrap_or_default();
        let stats = |fused: bool| {
            let mut loaded = self.loaded.lock().unwrap();
            if paths.len() > 1 {
                return None;
            }
            let mut kept = self.kept.write().unwrap();
            let stats = loaded.entry(paths[0].clone()).or_insert_with(|| {
                let (stats, found) = load_kept(crate::read_file(&paths[0]).ok()?, fused || self.fused).ok()??;
                kept.insert(paths[0].clone(), found);
                Some(stats)
            });
            if let (Some(stats), Some(kept)) = (stats.as_mut(), kept.get_mut(&paths[0]).filter(|kept| fused && !kept.fused)) {
                kept.fuse(Arc::make_mut(stats));
            }
            stats.clone()
        };
        let rendered = match name {
            "memory_profile" if paths.len() != 1 => return fail(Kind::Assertion, ""),
            "memory_profile" => stats(false).map(|stats| stats.memory.clone()),
            "overview_page" => stats(false).map(|stats| crate::tools::overview_page::json(&stats, &paths)),
            "input_pipeline_analyzer" => stats(false).map(|stats| crate::tools::input_pipeline_analyzer::json(&stats)),
            "framework_op_stats" => stats(false).map(|stats| crate::tools::framework_op_stats::json(&stats)),
            "kernel_stats" => stats(false).map(|stats| crate::xplane::gpu::kernel_stats_json(&stats)),
            "pod_viewer" => stats(false).map(|stats| crate::tools::pod_viewer::json(&stats)),
            "op_profile" => stats(true).map(|stats| crate::tools::op_profile::json_trees(&stats, Some(option("group_by").unwrap_or("program")), false)),
            "hlo_stats" => stats(true).map(|stats| crate::tools::hlo_stats::json(&stats)),
            "roofline_model" => stats(false).map(|stats| crate::tools::roofline::json_rows(&stats, option(TOTAL_ONLY).is_some())),
            "memory_viewer" => crate::hlo::memory::serve(&dir, &options).map(|(body, _)| body),
            "graph_viewer" => return crate::hlo::graph::serve(&dir, &options).map(|(body, _)| Some(body)).map_err(|message| Error::new(Kind::Value, message)),
            "utilization_viewer" | "perf_counters" => {
                // The process does not read or check a trace again after it read the trace one time.
                let known =
                    (name == "utilization_viewer" && paths.len() == 1).then(|| self.kept.read().unwrap().get(&paths[0]).map(|kept| crate::tools::counters::utilization_viewer(kept.map()))).flatten();
                known.or_else(|| crate::tools::counters::serve(name, &paths))
            }
            "kernel_utilization" => <[PathBuf; 1]>::try_from(paths.clone()).ok().and_then(|[path]| kernel_utilization(&path, option)),
            "trace_viewer" => <[PathBuf; 1]>::try_from(paths.clone()).ok().and_then(|[path]| self.prepared(&path, crate::trace::legacy::render)),
            _ => None,
        };
        Ok(rendered.map(String::into_bytes))
    }
}
