use super::json::{J, py_repr};
use super::{Error, Kind, fail};
use crate::tools::opstats::{Kept, OpStats, Part, load_kept};
use crate::tools::roofline::Rows;
use crate::xplane::Plane;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// A parameter of the local client only. The table then has the first row only.
const TOTAL_ONLY: &str = "total_only";
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
        Ok(host_names(&self.xspace_paths(&self.run_dir(session)?)?))
    }

    /// The durations of the `barrier-cores` events of the first host.
    fn barrier_durations(&self, session: &str) -> Option<Vec<f64>>;

    /// The compute, HBM and ICI sums of the times of the leaves of the op profile by category.
    fn category_times(&self, session: &str, params: &Params) -> Option<[f64; 3]>;

    /// The roofline table, of which the device information reads the properties only.
    fn roofline_properties(&self, session: &str, params: &Params) -> Result<Option<String>, Error>;

    /// The roofline table, of which the overview reads the first row only.
    fn roofline_total(&self, session: &str) -> Option<String>;

    /// The op profile of a session, as the JSON reader gives it.
    fn op_profile(&self, session: &str, params: &Params) -> Result<Option<J>, Error>;

    fn fetch_text(&self, tool: &str, session: &str, params: &Params) -> Result<Option<String>, Error> {
        Ok(self
            .fetch(tool, session, params)?
            .filter(|data| !data.is_empty())
            .map(|data| String::from_utf8(data).unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())))
    }

    fn fetch_either(&self, first: &str, second: &str, session: &str, params: &Params) -> Result<Option<String>, Error> {
        match self.fetch_text(first, session, params)? {
            Some(data) => Ok(Some(data)),
            None => self.fetch_text(second, session, params),
        }
    }
}

pub fn tool_name(tool: &str) -> Result<&str, Error> {
    let name = tool.strip_suffix(".json").unwrap_or(tool);
    let name = if name == "hlo_op_profile" { "op_profile" } else { name };
    if crate::XPLANE_TOOLS.contains(&name) { Ok(name) } else { fail(Kind::Value, format!("Unknown XProf tool name: {}", py_repr(tool))) }
}

pub fn host_names(paths: &[PathBuf]) -> Vec<String> {
    let mut hosts: Vec<String> = paths.iter().map(|path| host(path)).collect();
    hosts.sort();
    hosts.dedup();
    hosts
}

pub fn host(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    TRACE_SUFFIXES.iter().find_map(|suffix| name.strip_suffix(suffix)).unwrap_or(&name).rsplit('.').next().unwrap_or_default().to_string()
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
        return if TRACE_SUFFIXES.iter().any(|suffix| dir.to_string_lossy().ends_with(suffix)) {
            vec![dir.to_path_buf()]
        } else {
            Vec::new()
        };
    }
    find(dir, &TRACE_SUFFIXES, true)
}

pub fn kernel_utilization<'a>(map: &[u8], option: impl Fn(&str) -> Option<&'a str>) -> Option<String> {
    let filter = crate::tools::counters::Filter {
        kernel: option("kernel").or(option("kernel_name")).unwrap_or_default().to_string(),
        duration_us: option("duration_us").and_then(|value| value.trim().parse().ok()).unwrap_or(0.0),
        force: option("force_duration").is_some_and(|value| matches!(value, "True" | "true" | "1" | "yes" | "t" | "y")),
        device: option("device_id").and_then(|value| value.trim().parse().ok()).unwrap_or(-1),
    };
    crate::tools::counters::checked(map, |map| crate::tools::counters::kernel_utilization(map, &filter))
}

pub fn temp_dir() -> PathBuf {
    ["TMPDIR", "TEMP", "TMP"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .find(|path| path.is_dir())
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// The last session of `dir/plugins/profile` in name order.
pub(super) fn latest_run(dir: &Path) -> Option<PathBuf> {
    let mut runs = crate::subdirs(&dir.join("plugins/profile"));
    runs.sort();
    runs.pop()
}

#[derive(Default)]
pub struct Local {
    pub logdir: Option<PathBuf>,
    /// Loads traces with the fused children at once, for a command that needs them after other
    /// data.
    pub fused: bool,
    pub loaded: Mutex<HashMap<PathBuf, Option<Arc<OpStats>>>>,
    pub kept: RwLock<HashMap<PathBuf, Kept>>,
}

impl Local {
    /// The op statistics of all the hosts. The process loads each file one time and extends it when
    /// a tool needs more.
    fn stats(&self, paths: &[PathBuf], fused: bool, part: Part) -> Option<Arc<OpStats>> {
        let (mut loaded, mut kept) = (self.loaded.lock().expect("lock poisoned"), self.kept.write().expect("lock poisoned"));
        let mut all: Vec<_> = paths
            .iter()
            .map(|path| {
                let stats = loaded.entry(path.clone()).or_insert_with(|| {
                    let map = crate::read_file(path).ok()?;
                    if map.is_empty() || !crate::tools::counters::valid_space(&map) {
                        return None;
                    }
                    let (stats, found) = load_kept(map, fused || self.fused, part).ok()??;
                    kept.insert(path.clone(), found);
                    Some(stats)
                });
                if let (Some(stats), Some(kept)) = (stats.as_mut(), kept.get_mut(path)) {
                    if fused && !kept.fused {
                        kept.fuse(Arc::make_mut(stats));
                    }
                    if kept.part < part {
                        kept.upgrade(Arc::make_mut(stats), part);
                    }
                }
                stats.clone()
            })
            .collect();
        all.retain(Option::is_some);
        OpStats::combine(&all)
    }

    /// Reads the prepared planes of a file. It reuses the planes of the op statistics when no GPU
    /// plane needs the trace derivation.
    fn prepared<T>(&self, path: &Path, read: impl FnOnce(&[Plane], &[u8]) -> T) -> Option<T> {
        if let Some(kept) = self
            .kept
            .write()
            .expect("lock poisoned")
            .get_mut(path)
            .filter(|kept| !kept.planes.iter().any(|plane| plane.name.starts_with(crate::xplane::gpu::PREFIX)))
        {
            kept.finish(true);
            return Some(read(&kept.planes, kept.map()));
        }
        let (map, planes) = crate::prepare(path).ok()?;
        Some(read(&planes, &map))
    }
}

impl Client for Local {
    fn logdir(&self) -> Option<&Path> {
        self.logdir.as_deref()
    }

    fn op_profile(&self, session: &str, params: &Params) -> Result<Option<J>, Error> {
        let paths = self.xspace_paths(&self.run_dir(session)?)?;
        let group_by = params.iter().find(|(key, _)| *key == "group_by").map_or("program", |(_, value)| value.as_str());
        Ok(self.stats(&paths, true, Part::Device).map(|stats| crate::tools::op_profile::value_trees(&stats, Some(group_by))))
    }

    fn shared(&self) -> Option<&(dyn Client + Sync)> {
        Some(self)
    }

    fn roofline_total(&self, session: &str) -> Option<String> {
        self.fetch_text("roofline_model.json", session, &[(TOTAL_ONLY, String::new())]).ok()?
    }

    fn roofline_properties(&self, session: &str, _: &Params) -> Result<Option<String>, Error> {
        let paths = self.xspace_paths(&self.run_dir(session)?)?;
        Ok(self.stats(&paths, false, Part::Device).map(|stats| crate::tools::roofline::json_rows(&stats, Rows::None)))
    }

    fn category_times(&self, session: &str, _: &Params) -> Option<[f64; 3]> {
        let stats = self.stats(&self.xspace_paths(&self.run_dir(session).ok()?).ok()?, true, Part::Device)?;
        Some(crate::tools::op_profile::category_times(&stats, super::steps::time_class))
    }

    fn barrier_durations(&self, session: &str) -> Option<Vec<f64>> {
        let mut paths = self.xspace_paths(&self.run_dir(session).ok()?).ok()?;
        paths.sort_by_key(|path| host(path));
        paths.into_iter().find_map(|path| {
            self.prepared(&path, |planes, map| {
                (crate::tools::counters::valid_space(map) && planes.iter().any(|plane| plane.name.starts_with("/device:")))
                    .then(|| crate::trace::legacy::barrier_durations(planes, map))
            })
            .flatten()
        })
    }

    fn run_dir(&self, session: &str) -> Result<PathBuf, Error> {
        if !session.is_empty() {
            let path = super::expand(session);
            if path.is_file() {
                return Ok(path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new(".")).to_path_buf());
            }
            if path.is_dir() {
                return Ok(latest_run(&path).unwrap_or(path));
            }
        }
        let Some(logdir) = &self.logdir else {
            return if session.is_empty() {
                fail(Kind::Value, "Logdir not set. Please configure logdir first.")
            } else {
                fail(Kind::FileNotFound, format!("Path not found: {session}"))
            };
        };
        if session.is_empty() {
            return Ok(latest_run(logdir).unwrap_or_else(|| logdir.clone()));
        }
        let run = logdir.join("plugins/profile").join(session);
        if run.exists() {
            return Ok(run);
        }
        let formatted = (session.len() == 14 && session.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| {
                logdir.join("plugins/profile").join(format!(
                    "{}_{}_{}_{}_{}_{}",
                    &session[..4],
                    &session[4..6],
                    &session[6..8],
                    &session[8..10],
                    &session[10..12],
                    &session[12..]
                ))
            })
            .filter(|path| path.exists());
        let fallback = Some(logdir.join(session)).filter(|path| path.exists());
        formatted.or(fallback).map_or_else(
            || fail(Kind::FileNotFound, format!("Run directory not found for session {} in {}", py_repr(session), logdir.display())),
            Ok,
        )
    }

    fn fetch(&self, tool: &str, session: &str, params: &Params) -> Result<Option<Vec<u8>>, Error> {
        let name = tool_name(tool)?;
        let paths = self.xspace_paths(&self.run_dir(session)?)?;
        let flag = |key: &str, value: &str| {
            if matches!(key, "show_metadata" | "merge_fusion") {
                value.trim().parse::<i64>().is_ok_and(|number| number != 0).to_string()
            } else {
                value.to_string()
            }
        };
        let options: HashMap<String, String> = params.iter().map(|(key, value)| (key.to_string(), flag(key, value))).collect();
        let option = |key: &str| options.get(key).map(String::as_str);
        let dir = paths[0].parent().map(Path::to_path_buf).unwrap_or_default();
        let stats = |fused: bool, part: Part| self.stats(&paths, fused, part);
        let rendered = match name {
            "memory_profile" if paths.len() != 1 && option("host").is_none() => {
                return fail(Kind::Assertion, "");
            }
            "memory_profile" => {
                let path = paths.iter().find(|path| option("host").is_none_or(|name| host(path) == name));
                let Some(path) = path else {
                    return fail(Kind::Assertion, "");
                };
                match self.kept.write().expect("lock poisoned").get_mut(path) {
                    Some(kept) => {
                        kept.finish(false);
                        Some(crate::tools::memory_profile::json(&kept.planes, kept.map()))
                    }
                    None => crate::tools::memory_profile::load(path).ok().flatten(),
                }
            }
            "overview_page" => stats(false, Part::All).map(|stats| {
                let loaded = self.loaded.lock().expect("lock poisoned");
                let paths = paths.iter().filter(|path| loaded.get(*path).is_some_and(Option::is_some)).cloned().collect::<Vec<_>>();
                crate::tools::overview_page::json(&stats, &paths)
            }),
            "input_pipeline_analyzer" => stats(false, Part::All).map(|stats| crate::tools::input_pipeline_analyzer::json(&stats)),
            "pod_viewer" => stats(false, Part::All).map(|stats| crate::tools::pod_viewer::json(&stats)),
            "hlo_stats" => stats(false, Part::Device).map(|stats| crate::tools::hlo_stats::json(&stats)),
            "roofline_model" => stats(false, Part::Programs)
                .map(|stats| crate::tools::roofline::json_rows(&stats, if option(TOTAL_ONLY).is_some() { Rows::Total } else { Rows::All })),
            "memory_viewer" => crate::hlo::memory::serve(&dir, &options).map(|(body, _)| body),
            "graph_viewer" => {
                return crate::hlo::graph::serve(&dir, &options)
                    .map(|(body, _)| Some(body))
                    .map_err(|message| Error::new(Kind::Value, message));
            }
            "utilization_viewer" | "perf_counters" => {
                // The process does not read or check a trace again after it read the trace one
                // time.
                let known = (name == "utilization_viewer" && paths.len() == 1)
                    .then(|| {
                        self.kept
                            .read()
                            .expect("lock poisoned")
                            .get(&paths[0])
                            .map(|kept| crate::tools::counters::utilization_viewer(kept.map()))
                    })
                    .flatten();
                known.or_else(|| crate::tools::counters::serve(name, &paths))
            }
            "kernel_utilization" => <[PathBuf; 1]>::try_from(paths.clone())
                .ok()
                .and_then(|[path]| crate::read_file(&path).ok().and_then(|map| kernel_utilization(&map, option))),
            _ => None,
        };
        Ok(rendered.map(String::into_bytes))
    }
}
