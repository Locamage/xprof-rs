use crate::tools::counters::{checked, events, first, planes, pretty};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

const CACHE_FILE: &str = ".cached_tools.json";
const HLO_SUFFIX: &str = ".hlo_proto.pb";
const CACHE_VERSION: i64 = 1;
const XPLANE_EXTENSIONS: [&str; 2] = ["xplane.pb", "xplane.riegeli"];
const TOOL_EXTENSIONS: [&str; 3] = ["xplane.pb", "hlo_proto.pb", "xplane.riegeli"];
const SORT_ORDER: [&str; 20] = [
    "overview_page",
    "trace_viewer",
    "trace_viewer@",
    "graph_viewer",
    "op_profile",
    "hlo_op_profile",
    "input_pipeline_analyzer",
    "input_pipeline",
    "kernel_stats",
    "memory_profile",
    "memory_viewer",
    "roofline_model",
    "perf_counters",
    "pod_viewer",
    "framework_op_stats",
    "tensorflow_stats",
    "hlo_op_stats",
    "hlo_stats",
    "inference_profile",
    "megascale_stats",
];
const BASE_TOOLS: [&str; 8] = ["trace_viewer@", "overview_page", "input_pipeline_analyzer", "framework_op_stats", "memory_profile", "op_profile", "hlo_stats", "roofline_model"];
const GPU_PREFIX: &str = "/device:GPU:";
const TPU_PREFIX: &str = "/device:TPU:";
const METADATA_PLANE: &str = "/host:metadata";
const HOST_PLANE: &str = "/host:CPU";
const MEGASCALE_PREFIX: &[u8] = b"MegaScale:";
const HLO_PROTO: &str = "Hlo Proto";
const COUNTER_VALUE: &str = "counter_value";

pub fn tools(map: &[u8]) -> Option<Vec<&'static str>> {
    checked(map, |map| -> Vec<&'static str> {
        let planes = planes(map, |_| true);
        let counters = planes.par_iter().filter(|plane| plane.name.starts_with(TPU_PREFIX)).any(|plane| {
            let id = plane.stat_id(COUNTER_VALUE);
            id.is_some() && plane.lines.par_iter().any(|line| events(line).any(|event| first(event.raw, 4, [id])[0].is_some()))
        });
        let found: [(bool, &[&str]); 4] = [
            (planes.iter().any(|plane| plane.name.starts_with(GPU_PREFIX)), &["kernel_stats"]),
            (planes.iter().any(|plane| plane.name == METADATA_PLANE && plane.stat_id(HLO_PROTO).is_some()), &["memory_viewer", "graph_viewer"]),
            (planes.iter().find(|plane| plane.name == HOST_PLANE).is_some_and(|plane| plane.metadata.values().any(|(name, _)| name.starts_with(MEGASCALE_PREFIX))), &["megascale_stats"]),
            (counters, &["perf_counters", "utilization_viewer", "kernel_utilization"]),
        ];
        BASE_TOOLS.into_iter().chain(found.into_iter().filter(|(present, _)| *present).flat_map(|(_, names)| names.iter().copied())).collect()
    })
}

pub fn sorted<'a>(tools: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut set: BTreeSet<&str> = tools.into_iter().collect();
    if set.contains("trace_viewer@") {
        set.remove("trace_viewer");
    }
    let mut out: Vec<String> = SORT_ORDER.iter().filter(|tool| set.contains(*tool)).map(std::string::ToString::to_string).collect();
    out.extend(set.into_iter().filter(|tool| !SORT_ORDER.contains(tool)).map(String::from));
    out
}

fn file_states(dir: &Path) -> Option<BTreeMap<String, String>> {
    let mut states = BTreeMap::new();
    for entry in std::fs::read_dir(dir).ok()? {
        let name = entry.ok()?.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || !XPLANE_EXTENSIONS.iter().any(|extension| name.len() > extension.len() && name.ends_with(&format!(".{extension}"))) {
            continue;
        }
        let meta = std::fs::metadata(dir.join(&name)).ok()?;
        let seconds = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs());
        states.insert(name, format!("{seconds}-{}", meta.len()));
    }
    Some(states)
}

pub fn cache_path(dir: &Path) -> PathBuf {
    let text = dir.to_string_lossy();
    let demo = text.match_indices("demo/plugins/profile").any(|(at, matched)| (at == 0 || text.as_bytes()[at - 1] == b'/') && matches!(text.as_bytes().get(at + matched.len()), None | Some(b'/')));
    if !demo {
        return dir.join(CACHE_FILE);
    }
    let key: String = Sha256::digest(text.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect::<String>()[..16].to_string();
    let Ok(folder) = crate::private_dir(&crate::cli::client::temp_dir().join(format!("xprof_{}", unsafe { libc::getuid() }))) else { return dir.join(CACHE_FILE) };
    folder.join(format!("xprof_{key}_{CACHE_FILE}"))
}

pub(crate) fn python_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    python_string_into(&mut out, text);
    out
}

/// Writes the text as a Python JSON string. It copies runs of plain characters without changes.
pub(crate) fn python_string_into(out: &mut String, text: &str) {
    out.push('"');
    let (bytes, mut run, mut index) = (text.as_bytes(), 0, 0);
    while index < bytes.len() {
        let byte = bytes[index];
        if (b' '..=b'~').contains(&byte) && byte != b'"' && byte != b'\\' {
            index += 1;
            continue;
        }
        out.push_str(&text[run..index]);
        let character = text[index..].chars().next().unwrap_or_default();
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            _ => character.encode_utf16(&mut [0; 2]).iter().for_each(|unit| write!(out, "\\u{unit:04x}").unwrap()),
        }
        index += character.len_utf8();
        run = index;
    }
    out.push_str(&text[run..]);
    out.push('"');
}

pub(crate) fn python_value(value: &Value) -> String {
    match value {
        Value::String(text) => python_string(text),
        Value::Array(items) => format!("[{}]", items.iter().map(python_value).collect::<Vec<_>>().join(", ")),
        Value::Object(entries) => format!("{{{}}}", entries.iter().map(|(key, value)| format!("{}: {}", python_string(key), python_value(value))).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

fn load(cache: &Path, dir: &Path) -> Option<Vec<Value>> {
    let text = std::fs::read(cache).ok()?;
    let Some(Value::Object(data)) = serde_json::from_slice::<Value>(&text).ok() else {
        _ = std::fs::remove_file(cache);
        return None;
    };
    let version_matches = match data.get("version") {
        Some(Value::Number(number)) => number.as_f64() == Some(CACHE_VERSION as f64),
        Some(Value::Bool(flag)) => *flag,
        _ => false,
    };
    let current = file_states(dir);
    let files_match = current.as_ref().is_some_and(|current| match data.get("files") {
        Some(Value::Object(files)) => files.len() == current.len() && current.iter().all(|(name, state)| files.get(name).and_then(Value::as_str) == Some(state.as_str())),
        _ => false,
    });
    if !version_matches || !files_match {
        _ = std::fs::remove_file(cache);
        return None;
    }
    match data.get("tools")? {
        Value::Array(items) => Some(items.clone()),
        Value::String(text) => Some(text.chars().map(|character| Value::String(character.to_string())).collect()),
        Value::Object(entries) => Some(entries.keys().map(|key| Value::String(key.clone())).collect()),
        _ => None,
    }
}

fn save(cache: &Path, dir: &Path, tools: &[String]) {
    let Some(states) = file_states(dir) else { return };
    let mut out = String::new();
    pretty(&mut out, &json!({"files": states, "tools": tools, "version": CACHE_VERSION}), 0, python_string_into);
    _ = std::fs::write(cache, out);
}

fn generate(dir: &Path) -> Option<Vec<String>> {
    let files: Vec<String> = std::fs::read_dir(dir).ok()?.filter_map(|entry| entry.ok().map(|entry| entry.file_name().to_string_lossy().into_owned())).collect();
    if files.is_empty() {
        return None;
    }
    let extension = |name: &str| TOOL_EXTENSIONS.into_iter().find(|extension| name == *extension || name.strip_suffix(extension).is_some_and(|host| host.ends_with('.')));
    let xspaces: Vec<PathBuf> = files.iter().filter(|name| extension(name).is_some_and(|extension| XPLANE_EXTENSIONS.contains(&extension))).map(|name| dir.join(name)).collect();
    if xspaces.is_empty() {
        return Some(Vec::new());
    }
    let extracted = files.iter().any(|name| name.ends_with(HLO_SUFFIX));
    let scan = |path: &PathBuf| {
        let map = crate::read_file(path).ok().filter(|_| path.to_string_lossy().ends_with(".pb"))?;
        tools(&map)
    };
    let found: Option<Vec<Vec<&str>>> = xspaces.iter().take(if extracted { 1 } else { xspaces.len() }).map(scan).collect();
    let found = found.filter(|_| extracted || crate::hlo::extracted(dir, &xspaces).is_some());
    Some(found.map_or_else(Vec::new, |found| sorted(found[0].iter().copied())))
}

pub fn json(dir: &Path) -> String {
    let cache = cache_path(dir);
    let tools = load(&cache, dir).unwrap_or_else(|| {
        let generated = generate(dir);
        if let Some(tools) = &generated {
            save(&cache, dir, tools);
        }
        generated.unwrap_or_default().into_iter().map(Value::String).collect()
    });
    python_value(&Value::Array(tools))
}

#[cfg(test)]
#[path = "../tests/inline/server/run_tools.rs"]
mod tests;
