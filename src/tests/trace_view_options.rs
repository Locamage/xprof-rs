use super::legacy::served;
use super::streaming_trace_viewer_processor::create_test_x_space;
use axum::http::StatusCode;
use prost::Message;

async fn view(name: &str, query: &str) -> (StatusCode, String) {
    let (status, _, body) = served(name, &[("host1", create_test_x_space(2).encode_to_vec())], &format!("tag=trace_viewer@&host=host1{query}")).await;
    (status, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_valid_options() {
    let query = "&start_time_ms=1.2&end_time_ms=200.0&resolution=1000&event_name=SessionRun&search_prefix=prefix&duration_ms=0.05&unique_id=0&format=proto&search_metadata=true";
    let (status, body) = view("options-valid", query).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = serde_json::from_str::<serde_json::Value>(&body).unwrap()["traceEvents"].as_array().unwrap().clone();
    assert_eq!(events.iter().filter(|event| event["ph"] == "X").map(|event| event["name"].as_str().unwrap()).collect::<Vec<_>>(), ["SessionRun"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_default_options() {
    let (status, body) = view("options-default", "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"returnedEventsSize\":2,"), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_invalid_double() {
    assert_eq!(view("options-invalid-double", "&start_time_ms=invalid").await.0, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_invalid_int() {
    assert_eq!(view("options-invalid-int", "&resolution=invalid").await.0, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_double_as_int() {
    let (status, body) = view("options-double-as-int", "&resolution=1000.0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, view("options-int", "&resolution=1000").await.1);
}
