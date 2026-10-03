use super::legacy::{bytes, fetch, number};
use crate::{Settings, Shared, app, state};
use axum::body::Body;
use axum::http::{StatusCode, header};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tower::ServiceExt;

const PREFIX: &str = "/data/plugin/profile";
const RUN_TO_TOOLS: [(&str, &[&str]); 6] = [
    ("foo", &["trace_viewer", "trace_viewer@"]),
    ("bar", &["unsupported"]),
    ("baz", &["overview_page", "op_profile", "trace_viewer", "trace_viewer@"]),
    ("qux", &["overview_page", "input_pipeline_analyzer", "trace_viewer", "trace_viewer@"]),
    ("abc", &["xplane"]),
    ("empty", &[]),
];
const RUN_TO_HOSTS: [(&str, &[Option<&str>]); 6] =
    [("foo", &[Some("host0"), Some("host1")]), ("bar", &[Some("host1")]), ("baz", &[Some("host2")]), ("qux", &[None]), ("abc", &[Some("host1"), Some("host2")]), ("empty", &[])];
const XPLANE_TOOLS: [&str; 5] = ["trace_viewer", "trace_viewer@", "overview_page", "op_profile", "input_pipeline_analyzer"];

fn temp_logdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xprof-rs-plugin-{}-{name}", std::process::id()));
    _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.canonicalize().unwrap()
}

fn generate_testdata(logdir: &Path) {
    let plugin = logdir.join("plugins/profile");
    std::fs::create_dir_all(&plugin).unwrap();
    for (run, tools) in RUN_TO_TOOLS {
        std::fs::create_dir(plugin.join(run)).unwrap();
        let hosts = RUN_TO_HOSTS.iter().find(|(name, _)| *name == run).unwrap().1;
        for tool in tools.iter().filter(|tool| XPLANE_TOOLS.contains(tool) || **tool == "xplane") {
            for host in hosts {
                let data = if tool.starts_with("trace_viewer") { bytes(1, &[number(1, 0), bytes(2, &bytes(1, run.as_bytes()))].concat()) } else { tool.as_bytes().to_vec() };
                std::fs::write(plugin.join(run).join(host.map_or("xplane.pb".to_string(), |host| format!("{host}.xplane.pb"))), data).unwrap();
            }
        }
    }
    std::fs::write(plugin.join("noise"), "Not a dir, not a run.").unwrap();
}

fn write_empty_event_file(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("events.out.tfevents.profile-empty"), "").unwrap();
}

fn plugin(logdir: &Path) -> Shared {
    state(&Settings { logdir: logdir.to_path_buf(), ..Default::default() })
}

async fn json(state: &Shared, uri: &str) -> serde_json::Value {
    let (status, _, body) = fetch(state, &format!("{PREFIX}/{uri}")).await;
    assert_eq!(status, StatusCode::OK, "{uri}: {body}");
    serde_json::from_str(&body).unwrap()
}

async fn runs(state: &Shared) -> BTreeSet<String> {
    json(state, "runs").await.as_array().unwrap().iter().map(|run| run.as_str().unwrap().to_string()).collect()
}

fn expected_runs(prefixes: &[&str]) -> BTreeSet<String> {
    prefixes.iter().flat_map(|prefix| RUN_TO_TOOLS.iter().map(move |(run, _)| format!("{prefix}{run}"))).collect()
}

async fn post(state: &Shared, uri: &str) -> (StatusCode, String) {
    let reply = app(state.clone()).oneshot(axum::http::Request::builder().method("POST").uri(uri).body(Body::empty()).unwrap()).await.unwrap();
    let status = reply.status();
    (status, String::from_utf8_lossy(&axum::body::to_bytes(reply.into_body(), usize::MAX).await.unwrap()).into_owned())
}

fn settle(path: &Path) {
    std::fs::File::options().write(true).open(path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(60)).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_logdir_without_event_file() {
    let logdir = temp_logdir("runs-without-event-file");
    generate_testdata(&logdir);
    let state = plugin(&logdir);
    assert_eq!(runs(&state).await, expected_runs(&[""]));
    assert_eq!(json(&state, "run_tools?run=bar").await, serde_json::json!([]));
    assert_eq!(json(&state, "run_tools?run=empty").await, serde_json::json!([]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_logdir_with_event_file() {
    let logdir = temp_logdir("runs-with-event-file");
    write_empty_event_file(&logdir);
    generate_testdata(&logdir);
    assert_eq!(runs(&plugin(&logdir)).await, expected_runs(&[""]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_with_subdirectories() {
    let logdir = temp_logdir("runs-with-subdirectories");
    for dir in ["", "a", "b", "b/c"] {
        generate_testdata(&logdir.join(dir));
    }
    for dir in ["", "a", "b/c"] {
        write_empty_event_file(&logdir.join(dir));
    }
    assert_eq!(runs(&plugin(&logdir)).await, expected_runs(&["", "a/", "b/", "b/c/"]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_without_events() {
    let logdir = temp_logdir("runs-without-events");
    generate_testdata(&logdir);
    assert_eq!(runs(&plugin(&logdir)).await, expected_runs(&[""]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_with_nested_runs() {
    let logdir = temp_logdir("runs-with-nested-runs");
    generate_testdata(&logdir.join("2024-12-19"));
    write_empty_event_file(&logdir.join("2024-12-19/train"));
    write_empty_event_file(&logdir.join("2024-12-19/validation"));
    assert_eq!(runs(&plugin(&logdir)).await, expected_runs(&["2024-12-19/"]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hosts() {
    let logdir = temp_logdir("hosts");
    generate_testdata(&logdir);
    generate_testdata(&logdir.join("a"));
    write_empty_event_file(&logdir.join("a"));
    let state = plugin(&logdir);
    let hosts_abc = serde_json::json!([{"hostname": "host1"}, {"hostname": "host2"}]);
    let all_hosts_only = serde_json::json!([{"hostname": "ALL_HOSTS"}]);
    assert_eq!(json(&state, "hosts?run=qux&tag=framework_op_stats").await, serde_json::json!([]));
    assert_eq!(json(&state, "hosts?run=abc&tag=framework_op_stats").await, serde_json::json!([{"hostname": "ALL_HOSTS"}, {"hostname": "host1"}, {"hostname": "host2"}]));
    assert_eq!(json(&state, "hosts?run=abc&tag=trace_viewer").await, hosts_abc);
    assert_eq!(json(&state, "hosts?run=abc&tag=memory_profile").await, hosts_abc);
    assert_eq!(json(&state, "hosts?run=abc&tag=overview_page").await, all_hosts_only);
    assert_eq!(json(&state, "hosts?run=abc&tag=pod_viewer").await, all_hosts_only);
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "differs from upstream: xprof-rs reads only .xplane.pb files, it has no riegeli decoder, so host1.xplane.riegeli is not listed"]
async fn parse_filename_riegeli() {
    let logdir = temp_logdir("riegeli");
    let session = logdir.join("plugins/profile/s");
    std::fs::create_dir_all(&session).unwrap();
    for name in ["host0.xplane.pb", "host1.xplane.riegeli", "xplane.pb"] {
        std::fs::write(session.join(name), "").unwrap();
    }
    assert_eq!(json(&plugin(&logdir), "hosts?run=s&tag=trace_viewer@").await, serde_json::json!([{"hostname": "host0"}, {"hostname": "host1"}]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data() {
    let logdir = temp_logdir("data");
    generate_testdata(&logdir);
    generate_testdata(&logdir.join("a"));
    write_empty_event_file(&logdir.join("a"));
    let state = plugin(&logdir);
    for (query, expected) in [
        ("run=foo&tag=invalid_tool&host=host0", "No Data"),
        ("run=foo&tag=memory_viewer&host=host0", "No Data"),
        ("run=foo&tag=trace_viewer&host=invalid_host", "No xplane file found for host: invalid_host in run: foo"),
        ("run=bar&tag=unsupported&host=host1", "No Data"),
        ("run=bar&tag=trace_viewer&host=host0", "No xplane file found for run: bar, tool: trace_viewer"),
        ("run=qux&tag=trace_viewer&host=host", "No xplane file found for run: qux, tool: trace_viewer"),
        ("run=empty&tag=trace_viewer&host=", "No xplane file found for run: empty, tool: trace_viewer"),
        ("run=a/foo&tag=trace_viewer&host=invalid_host", "No xplane file found for host: invalid_host in run: a/foo"),
    ] {
        let (status, _, body) = fetch(&state, &format!("{PREFIX}/data?{query}")).await;
        assert_eq!((status, body.as_str()), (StatusCode::NOT_FOUND, expected), "{query}");
    }
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_names_only_happy_path() {
    let logdir = temp_logdir("names-only");
    let (status, headers, body) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?tag=perf_counters&names_only=1&device_type=v7x")).await;
    assert_eq!((status, &headers[header::CONTENT_TYPE]), (StatusCode::OK, &"application/json".parse::<header::HeaderValue>().unwrap()));
    let names: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(names[0].get("name").is_some() && names[0].get("val").is_some(), "{body}");
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_names_only_missing_device_type() {
    let logdir = temp_logdir("names-only-missing");
    let (status, _, body) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?tag=perf_counters&names_only=1")).await;
    assert!(status == StatusCode::INTERNAL_SERVER_ERROR && body.contains("device_type is required"), "{status} {body}");
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_impl_trace_viewer_options() {
    let logdir = temp_logdir("trace-viewer-options");
    generate_testdata(&logdir);
    let query = "run=foo&tag=trace_viewer@&host=host1&full_dma=true&resolution=10000&start_time_ms=100&end_time_ms=200&search_metadata=false";
    let (status, headers, body) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?{query}")).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "application/json"), "{body}");
    assert!(body.contains("\"returnedEventsSize\":0,"), "{body}");
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_impl_trace_viewer_format_pb_returns_octet_stream() {
    let logdir = temp_logdir("trace-viewer-pb");
    generate_testdata(&logdir);
    let (status, headers, _) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?run=foo&tag=trace_viewer&host=host1&format=pb")).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "application/octet-stream"));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_impl_trace_viewer_streaming_pb_returns_octet_stream() {
    let logdir = temp_logdir("trace-viewer-streaming-pb");
    generate_testdata(&logdir);
    let (status, headers, _) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?run=foo&tag=trace_viewer@&host=host1&format=pb")).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "application/octet-stream"));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_impl_trace_viewer_event_name_fallback_to_json() {
    let logdir = temp_logdir("trace-viewer-event-name");
    generate_testdata(&logdir);
    let (status, headers, body) = fetch(&plugin(&logdir), &format!("{PREFIX}/data?run=foo&tag=trace_viewer&host=host1&format=pb&event_name=mock_event")).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "application/json"), "{body}");
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_runs_no_logdir() {
    let state = state(&Settings::default());
    assert_eq!(runs(&state).await, BTreeSet::new());
    let elsewhere = temp_logdir("no-logdir-session");
    std::fs::write(elsewhere.join("host.xplane.pb"), "").unwrap();
    assert_eq!(json(&state, &format!("runs?session_path={}", elsewhere.display())).await, serde_json::json!([elsewhere.file_name().unwrap().to_string_lossy()]));
    std::fs::remove_dir_all(&elsewhere).unwrap();
    let (status, _, body) = fetch(&state, &format!("{PREFIX}/capture_profile?service_addr=localhost:1")).await;
    assert_eq!((status, body.as_str()), (StatusCode::INTERNAL_SERVER_ERROR, "{\"error\": \"logdir is not set, abort capturing.\"}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_impl_with_session() {
    let logdir = temp_logdir("runs-session");
    let session = logdir.join("session_run");
    std::fs::create_dir(&session).unwrap();
    std::fs::write(session.join("host.xplane.pb"), "dummy xplane data").unwrap();
    assert_eq!(json(&plugin(&logdir), &format!("runs?session_path={}", session.display())).await, serde_json::json!(["session_run"]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_impl_with_run_path() {
    let logdir = temp_logdir("runs-run-path");
    let run = logdir.join("base/run1");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("host.xplane.pb"), "dummy xplane data").unwrap();
    assert_eq!(json(&plugin(&logdir), &format!("runs?run_path={}", logdir.join("base").display())).await, serde_json::json!(["run1"]));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_static_file_with_xprof_static_dir_success_and_fallbacks() {
    let logdir = temp_logdir("static");
    let (statics, sibling) = (logdir.join("static"), logdir.join("static_sibling"));
    std::fs::create_dir_all(&statics).unwrap();
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(statics.join("bundle.js"), "console.log(\"xprof\");").unwrap();
    std::fs::write(statics.join("index.html"), "<html>index</html>").unwrap();
    std::fs::write(sibling.join("secret.txt"), "sensitive_data").unwrap();
    std::os::unix::fs::symlink(sibling.join("secret.txt"), statics.join("styles.css")).unwrap();
    let state = plugin(&logdir);
    let embedded = fetch(&state, "/bundle.js").await.2;
    unsafe { std::env::set_var("XPROF_STATIC_DIR", &statics) };
    let served = fetch(&state, "/bundle.js").await;
    let traversal = fetch(&state, "/styles.css").await;
    unsafe { std::env::set_var("XPROF_STATIC_DIR", logdir.join("does_not_exist")) };
    let missing_dir = fetch(&state, "/bundle.js").await;
    unsafe { std::env::remove_var("XPROF_STATIC_DIR") };
    assert_eq!((served.0, served.2.as_str()), (StatusCode::OK, "console.log(\"xprof\");"));
    assert_eq!((traversal.0, traversal.2.as_str()), (StatusCode::NOT_FOUND, "Fail to read the files."));
    assert_eq!((missing_dir.0, &missing_dir.2), (StatusCode::OK, &embedded));
    assert_ne!(embedded, served.2);
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_cache_task_generates_cache() {
    let logdir = temp_logdir("generate-cache-task");
    let session = logdir.join("plugins/profile/s");
    std::fs::create_dir_all(&session).unwrap();
    let file = session.join("host1.xplane.pb");
    std::fs::write(&file, include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    settle(&file);
    let state = plugin(&logdir);
    assert_eq!(post(&state, &format!("{PREFIX}/generate_cache?session_path={}", session.display())).await.0, StatusCode::ACCEPTED);
    for _ in 0..600 {
        if state.hosts.cache.contains_key(&file) && state.stats.cache.contains_key(&file) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(state.hosts.cache.contains_key(&file) && state.stats.cache.contains_key(&file));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_cache_impl() {
    let logdir = temp_logdir("generate-cache-impl");
    let (session, empty) = (logdir.join("plugins/profile/s"), logdir.join("plugins/profile/empty"));
    std::fs::create_dir_all(&session).unwrap();
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(session.join("host1.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = plugin(&logdir);
    assert_eq!(fetch(&state, &format!("{PREFIX}/generate_cache?session_path={}", session.display())).await.0, StatusCode::METHOD_NOT_ALLOWED);
    let (status, body) = post(&state, &format!("{PREFIX}/generate_cache")).await;
    assert!(status == StatusCode::BAD_REQUEST && body.contains("Missing \"session_path\" parameter"), "{status} {body}");
    let (status, body) = post(&state, &format!("{PREFIX}/generate_cache?session_path={}", empty.display())).await;
    assert!(status == StatusCode::NOT_FOUND && body.contains("No XPlane files found"), "{status} {body}");
    let (status, body) = post(&state, &format!("{PREFIX}/generate_cache?session_path={}&tools=unsupported_tool", session.display())).await;
    assert!(status == StatusCode::BAD_REQUEST && body.contains("No valid XPlane tools"), "{status} {body}");
    for tools in ["", "&tools=%20overview_page%20,%20trace_viewer@%20", "&tools=overview_page,trace_viewer@,graph_viewer", "&tools=overview_page"] {
        let (status, body) = post(&state, &format!("{PREFIX}/generate_cache?session_path={}{tools}", session.display())).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{tools}");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&body).unwrap()["status"], "ACCEPTED");
    }
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_csv_route_content_disposition() {
    let logdir = temp_logdir("csv-filename");
    for (run, tag, host, expected) in
        [("my/run/123", "roofline_model", "worker-pool-1-x2y3", "roofline_model_my_run_123_x2y3.csv"), ("clean_run", "hlo_stats", "localhost", "hlo_stats_clean_run_localhost.csv")]
    {
        let (prefix, session) = run.rsplit_once('/').unwrap_or((".", run));
        let dir = logdir.join(prefix).join("plugins/profile").join(session);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{host}.xplane.pb")), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
        let (status, headers, _) = fetch(&plugin(&logdir), &format!("{PREFIX}/data_csv?run={run}&tag={tag}&host={host}")).await;
        assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "text/csv"));
        assert!(headers[header::CONTENT_DISPOSITION].to_str().unwrap().contains(&format!("filename=\"{expected}\"")));
    }
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hlo_module_list_impl_sorts_module_names() {
    let logdir = temp_logdir("module-list");
    let run = logdir.join("plugins/profile/test_run");
    std::fs::create_dir_all(&run).unwrap();
    for name in ["jit_train_step(4869159985936022652)", "jit_add(8254229641153238180)", "jit__where(2141868631891211743)", "jit_add(3326385976583000095)"] {
        std::fs::write(run.join(format!("{name}.hlo_proto.pb")), "").unwrap();
    }
    let (status, _, body) = fetch(&plugin(&logdir), &format!("{PREFIX}/module_list?run=test_run")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "jit__where(2141868631891211743),jit_add(3326385976583000095),jit_add(8254229641153238180),jit_train_step(4869159985936022652)");
    std::fs::remove_dir_all(&logdir).unwrap();
}
