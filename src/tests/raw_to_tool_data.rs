use super::legacy::served;
use axum::http::{HeaderMap, StatusCode, header};

const DEMO: &[u8] = include_bytes!("../../tests/data/demo.xplane.pb");
const CORRUPT: &[u8] = &[0x0c];

fn content_type(headers: &HeaderMap) -> &str {
    headers[header::CONTENT_TYPE].to_str().unwrap()
}

async fn demo(name: &str, query: &str) -> (StatusCode, HeaderMap, String) {
    served(name, &[("h", DEMO.to_vec())], query).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_format_pb_returns_pb_data() {
    let (status, headers, body) = demo("raw-tv-pb", "tag=trace_viewer&host=h&format=pb").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "application/octet-stream"));
    assert!(!body.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_streaming_pb_returns_pb_data() {
    let (status, headers, _) = demo("raw-tvs-pb", "tag=trace_viewer@&host=h&format=pb").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "application/octet-stream"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_streaming_json_returns_raw_data() {
    let (status, headers, body) = demo("raw-tvs-json", "tag=trace_viewer@&host=h&format=json").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "application/json"));
    assert!(body.starts_with("{\"displayTimeUnit\":\"ns\""), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_json_returns_json_string() {
    let (status, headers, body) = demo("raw-tv-json", "tag=trace_viewer&host=h&format=json").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "application/json"));
    assert!(serde_json::from_str::<serde_json::Value>(&body).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_failure_returns_none() {
    let (status, _, body) = served("raw-tv-fail", &[("h", CORRUPT.to_vec())], "tag=trace_viewer&host=h&format=pb").await;
    assert_eq!((status, body.as_str()), (StatusCode::NOT_FOUND, "No Data"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_trace_viewer_streaming_fail_returns_none() {
    let (status, _, body) = served("raw-tvs-fail", &[("h", CORRUPT.to_vec())], "tag=trace_viewer@&host=h&format=pb").await;
    assert_eq!((status, body.as_str()), (StatusCode::NOT_FOUND, "No Data"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn json_to_csv_string_valid_json_returns_csv_string() {
    let (status, headers, body) = served("raw-csv", &[("h", DEMO.to_vec())], "tag=roofline_model&host=h").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(content_type(&headers), "application/json");
    let dir = super::legacy::logdir("raw-csv-route");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), DEMO).unwrap();
    let state = crate::state(&crate::Settings { logdir: dir.clone(), ..Default::default() });
    let (status, headers, csv) = super::legacy::fetch(&state, "/data/plugin/profile/data_csv?run=run/s&tag=roofline_model&host=h").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "text/csv"));
    assert!(csv.starts_with('"'), "{csv}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_names_valid_xspace_returns_tool_names() {
    let dir = super::legacy::logdir("raw-tool-names");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), DEMO).unwrap();
    let state = crate::state(&crate::Settings { logdir: dir.clone(), ..Default::default() });
    let (_, _, tools) = super::legacy::fetch(&state, "/data/plugin/profile/run_tools?run=run/s").await;
    let tools: Vec<String> = serde_json::from_str(&tools).unwrap();
    assert!(tools.contains(&"trace_viewer@".to_string()) && tools.contains(&"op_profile".to_string()), "{tools:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_names_invalid_xspace_returns_empty_list() {
    let dir = super::legacy::logdir("raw-tool-names-invalid");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), CORRUPT).unwrap();
    let state = crate::state(&crate::Settings { logdir: dir.clone(), ..Default::default() });
    assert_eq!(super::legacy::fetch(&state, "/data/plugin/profile/run_tools?run=run/s").await.2, "[]");
    std::fs::remove_dir_all(&dir).unwrap();
}

async fn graph_viewer(name: &str, graph_type: &str) -> (StatusCode, HeaderMap, String) {
    let dir = super::legacy::logdir(name);
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), DEMO).unwrap();
    let state = crate::state(&crate::Settings { logdir: dir.clone(), ..Default::default() });
    let (_, _, modules) = super::legacy::fetch(&state, "/data/plugin/profile/module_list?run=run/s").await;
    let module = modules.split(',').next().unwrap().replace('(', "%28").replace(')', "%29");
    let reply = super::legacy::fetch(&state, &format!("/data/plugin/profile/data?run=run/s&tag=graph_viewer&host=h&module_name={module}&{graph_type}")).await;
    std::fs::remove_dir_all(&dir).unwrap();
    reply
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_graph_viewer_pb_returns_octet_stream() {
    let (status, headers, _) = graph_viewer("raw-graph-pb", "type=pb").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "application/octet-stream"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_graph_viewer_graph_returns_html() {
    let (status, headers, body) = graph_viewer("raw-graph-html", "type=graph&node_name=copy-start&format=html").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "text/html"), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_graph_viewer_unknown_returns_plain_text() {
    let (status, headers, _) = graph_viewer("raw-graph-text", "type=adj_nodes&node_name=copy-start").await;
    assert_eq!((status, content_type(&headers)), (StatusCode::OK, "text/plain"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn xspace_to_tool_data_graph_viewer_failure_raises_value_error() {
    let (status, _, body) = graph_viewer("raw-graph-fail", "type=graph&node_name=nope").await;
    assert!(status == StatusCode::INTERNAL_SERVER_ERROR && body.contains("nope"), "{status} {body}");
}
