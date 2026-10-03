use super::legacy::{fetch, logdir};
use crate::{Settings, state};
use axum::http::StatusCode;

async fn csv(name: &str, tag: &str) -> (StatusCode, String) {
    let dir = logdir(name);
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let (status, _, body) = fetch(&state, &format!("/data/plugin/profile/data_csv?run=run/s&tag={tag}&host=h")).await;
    std::fs::remove_dir_all(&dir).unwrap();
    (status, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valid_json_list_input() {
    let (status, body) = csv("csv-list", "framework_op_stats").await;
    let tables: serde_json::Value = serde_json::from_str(include_str!("../../tests/data/framework_op_stats.json")).unwrap();
    let header: Vec<String> = tables[0]["cols"].as_array().unwrap().iter().map(|col| format!("\"{}\"", col["label"].as_str().unwrap())).collect();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.lines().next().unwrap(), header.join(","));
    assert_eq!(body.lines().count(), tables[0]["rows"].as_array().unwrap().len() + 1);
    assert!(body.lines().nth(1).unwrap().starts_with("\"1.0\",\"Device\","), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_handle_multiline_stack_traces() {
    let (status, body) = csv("csv-multiline", "hlo_stats").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains(",\"%") && body.split('"').skip(1).step_by(2).any(|cell| cell.contains('\n')), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_missing_cols_error() {
    assert_eq!(csv("csv-missing-cols", "trace_viewer@").await, (StatusCode::INTERNAL_SERVER_ERROR, "Data format not suitable for CSV (missing 'cols')".to_string()));
}
