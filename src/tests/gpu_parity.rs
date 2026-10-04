use crate::tests::legacy::{fetch, same};
use crate::{Settings, state};
use flate2::read::GzDecoder;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    GzDecoder::new(bytes).read_to_end(&mut out).unwrap();
    out
}

fn recorded(bytes: &[u8]) -> Vec<Value> {
    serde_json::from_slice(&gunzip(bytes)).unwrap()
}

fn params(case: &serde_json::Value) -> HashMap<String, String> {
    case["params"].as_object().unwrap().iter().map(|(key, value)| (key.clone(), value.as_str().unwrap().to_string())).collect()
}

fn canonical_id(text: &str, at: usize, prefix: &str, ids: &mut HashMap<String, usize>, out: &mut String) -> usize {
    let digits = text[at..].bytes().take_while(u8::is_ascii_digit).count();
    let next = ids.len();
    out.push_str(&format!("{prefix}{}", ids.entry(format!("{prefix}{}", &text[at..at + digits])).or_insert(next)));
    at + digits
}

fn canonical_line(line: &str, ids: &mut HashMap<String, usize>) -> String {
    let line = match line.split_once("Caller instructions: ") {
        Some((head, tail)) => {
            let (callers, rest) = tail.split_at(tail.find('<').unwrap_or(tail.len()));
            let mut callers: Vec<&str> = callers.split(", ").collect();
            callers.sort_unstable();
            format!("{head}Caller instructions: {}{rest}", callers.join(", "))
        }
        None => line.to_string(),
    };
    let mut out = String::new();
    let mut at = 0;
    let leading = line.len() - line.trim_start().len();
    while at < line.len() {
        let rest = &line[at..];
        let node_start = (at == leading || line[..at].ends_with("-> ")) && rest.starts_with(|c: char| c.is_ascii_digit());
        if rest.starts_with("cluster_") && rest[8..].starts_with(|c: char| c.is_ascii_digit()) {
            at = canonical_id(&line, at + 8, "cluster_", ids, &mut out);
        } else if node_start {
            at = canonical_id(&line, at, "node", ids, &mut out);
        } else {
            let character = rest.chars().next().unwrap();
            out.push(character);
            at += character.len_utf8();
        }
    }
    out
}

fn canonical_graph(text: &str) -> String {
    let mut ids = HashMap::new();
    let (mut hover, mut lines) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let trimmed = line.trim();
        if (trimmed.starts_with("%23node") || trimmed.starts_with("%23clust")) && trimmed.contains(":hover") {
            hover.push(trimmed.to_string());
        } else if !trimmed.is_empty() {
            lines.push(canonical_line(line, &mut ids));
        }
    }
    hover.sort();
    [lines, hover].concat().join("\n")
}

#[test]
fn graph_viewer_renders_real_gpu_modules_and_backend_configs_like_xprof() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/gpu/hlo");
    for case in recorded(include_bytes!("../../tests/data/gpu/hlo/expected.json.gz")) {
        let params = params(&case);
        let body = String::from_utf8(crate::graph_viewer::serve(&dir, &params).unwrap().0).unwrap();
        let expected = case["body"].as_str().unwrap();
        if params["type"] == "graph" {
            assert_eq!(canonical_graph(&body), canonical_graph(expected), "{params:?}");
        } else {
            assert_eq!(body, expected, "{params:?}");
        }
    }
}

fn canonical_trace(mut trace: Value, streaming: bool) -> Value {
    let mut flows: HashMap<String, usize> = HashMap::new();
    let mut canonical = |value: &Value| {
        let next = flows.len();
        Value::from(*flows.entry(value.to_string()).or_insert(next))
    };
    let frames = trace["stackFrames"].take();
    for event in trace["traceEvents"].as_array_mut().unwrap() {
        let event = event.as_object_mut().unwrap();
        if let Some(frame) = event.remove("sf") {
            event.insert("stack".into(), frames[frame.to_string()]["name"].clone());
        }
        if streaming && let Some(flow) = event.get("bind_id").map(&mut canonical) {
            event.insert("bind_id".into(), flow);
        }
        if let Some(flow) = event.get("args").and_then(|args| args.get("flow")).map(&mut canonical) {
            event["args"]["flow"] = flow;
        }
    }
    trace
}

fn cores_in_any_host_order(mut pod: Value) -> Value {
    for step in pod["podStatsSequence"]["podStatsMap"].as_array_mut().into_iter().flatten() {
        let mut cores: Vec<Value> = step["podStatsPerCore"].as_object().unwrap().values().cloned().collect();
        cores.sort_by_key(|core| core["hostName"].to_string());
        step["podStatsPerCore"] = Value::from(cores);
    }
    pod
}

fn equivalent(path: &str, body: &str, expected: &str) -> bool {
    let (Ok(actual), Ok(reference)) = (serde_json::from_str::<Value>(body), serde_json::from_str::<Value>(expected)) else { return body == expected };
    match ["tag=trace_viewer%40", "tag=trace_viewer&", "tag=pod_viewer"].iter().position(|tag| path.contains("profile/data?") && path.contains(tag)) {
        Some(2) => same(&cores_in_any_host_order(actual), &cores_in_any_host_order(reference)),
        Some(kind) => same(&canonical_trace(actual, kind == 0), &canonical_trace(reference, kind == 0)),
        None => same(&actual, &reference),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_gpu_profiles_answer_every_tool_like_xprof() {
    let root = crate::tests::temp_dir().join(format!("xprof-rs-gpu-parity-{}", std::process::id()));
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/gpu/xplane");
    for variant in std::fs::read_dir(&fixtures).unwrap().map(|entry| entry.unwrap().path()) {
        let session = root.join(variant.file_name().unwrap()).join("plugins/profile/s");
        std::fs::create_dir_all(&session).unwrap();
        for file in std::fs::read_dir(&variant).unwrap().map(|entry| entry.unwrap().path()) {
            std::fs::write(session.join(file.file_stem().unwrap()), gunzip(&std::fs::read(&file).unwrap())).unwrap();
        }
    }
    let state = state(&Settings { logdir: root.clone(), ..Default::default() });
    for case in recorded(include_bytes!("../../tests/data/gpu/expected.json.gz")) {
        let path = case["path"].as_str().unwrap();
        let (status, _, body) = fetch(&state, path).await;
        assert_eq!(u64::from(status.as_u16()), case["status"].as_u64().unwrap(), "{path}");
        assert!(status != 200 || equivalent(path, &body, case["body"].as_str().unwrap()), "{path}\n{body}");
    }
    std::fs::remove_dir_all(&root).unwrap();
}
