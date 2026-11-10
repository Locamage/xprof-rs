use super::legacy::served;
use axum::http::{StatusCode, header};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_argument() {
    let (status, _, body) = served("unified-invalid", &[], "tag=trace_viewer&host=test_host").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_session_json_success() {
    let (status, headers, body) = served("unified-json", &[("test_host", Vec::new())], "tag=trace_viewer&host=test_host").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/json");
    assert!(!body.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_session_pb_success() {
    let (status, headers, body) = served("unified-pb", &[("test_host", Vec::new())], "tag=trace_viewer&host=test_host&format=pb").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/octet-stream");
    assert!(!body.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_registration() {
    let (status, _, _) = served("unified-streaming", &[("test_host", Vec::new())], "tag=trace_viewer@&host=test_host").await;
    assert_eq!(status, StatusCode::OK);
}
