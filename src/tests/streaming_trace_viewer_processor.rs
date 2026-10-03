use super::legacy::served;
use super::xspace::XSpace;
use axum::http::StatusCode;
use prost::Message;

pub fn create_test_x_space(num_events: usize) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.host();
    plane.named_line(1, "Test Line");
    let events = [("TraceContext", 1_000_000_000, 100_000_000, 123), ("SessionRun", 1_200_000_000, 50_000_000, 456)];
    for &(name, offset, duration, group) in events.iter().take(num_events) {
        plane.event(1, name, offset, duration, &[("group_id", group.into())]);
    }
    space
}

fn single_event_x_space(name: &str) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.host();
    plane.named_line(1, "Test Line");
    plane.event(1, name, 1_200_000_000, 50_000_000, &[]);
    space
}

fn trace_events(body: &str) -> Vec<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(body).unwrap()["traceEvents"].as_array().unwrap().clone()
}

fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Option<&'a serde_json::Value> {
    events.iter().find(|event| event["name"] == name)
}

async fn single_host(name: &str, space: &XSpace, query: &str) -> (StatusCode, String) {
    let (status, _, body) = served(name, &[("host1", space.encode_to_vec())], &format!("tag=trace_viewer@&host=host1&{query}")).await;
    (status, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_valid() {
    let query = "start_time_ms=1.2&end_time_ms=200.0&resolution=1000&event_name=SessionRun&search_prefix=prefix&duration_ms=0.05&unique_id=0&search_metadata=false";
    let (status, body) = single_host("view-option-valid", &create_test_x_space(2), query).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = trace_events(&body);
    assert_eq!(events.iter().filter(|event| event["ph"] == "X").count(), 1);
    assert_eq!(named(&events, "SessionRun").unwrap()["args"]["group_id"], 456);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_search_metadata_true() {
    let (status, body) = single_host("view-option-search-metadata-true", &create_test_x_space(2), "search_metadata=true").await;
    assert_eq!(status, StatusCode::OK);
    assert!(named(&trace_events(&body), "SessionRun").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_search_metadata() {
    let (status, body) = single_host("view-option-search-metadata", &create_test_x_space(2), "search_prefix=Sess&search_metadata=true").await;
    assert_eq!(status, StatusCode::OK);
    let events = trace_events(&body);
    assert_eq!(named(&events, "SessionRun").unwrap()["args"]["group_id"], 456);
    assert!(named(&events, "TraceContext").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_defaults() {
    let (status, body) = single_host("view-option-defaults", &create_test_x_space(2), "").await;
    assert_eq!(status, StatusCode::OK);
    let events = trace_events(&body);
    assert!(named(&events, "TraceContext").is_some() && named(&events, "SessionRun").is_some(), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_float_formatted() {
    let query = "start_time_ms=1.2&end_time_ms=200.0&resolution=1000.000000&event_name=SessionRun&search_prefix=prefix&duration_ms=0.05&unique_id=0.000000";
    let (status, body) = single_host("view-option-float", &create_test_x_space(2), query).await;
    assert_eq!(status, StatusCode::OK);
    assert!(named(&trace_events(&body), "SessionRun").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_trace_view_option_invalid_number() {
    assert_eq!(single_host("view-option-invalid", &create_test_x_space(2), "resolution=not_a_number").await.0, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_single_host() {
    let (status, body) = single_host("reduce-single-host", &create_test_x_space(2), "end_time_ms=2000.0&resolution=1").await;
    assert_eq!(status, StatusCode::OK);
    let ts = named(&trace_events(&body), "SessionRun").unwrap()["ts"].as_f64().unwrap();
    assert!((ts - 1200.0).abs() <= 1.0, "{ts}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_multi_host() {
    let hosts = [("host1", create_test_x_space(1).encode_to_vec()), ("host2", create_test_x_space(1).encode_to_vec())];
    let (status, _, body) = served("reduce-multi-host", &hosts, "tag=trace_viewer@&hosts=host1,host2&end_time_ms=2000.0&resolution=1").await;
    assert_eq!(status, StatusCode::OK);
    assert!(!trace_events(&body).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_multi_host_stress_test() {
    let names: Vec<String> = (0..10).map(|index| format!("host{index}")).collect();
    let hosts: Vec<(&str, Vec<u8>)> = names.iter().map(|name| (name.as_str(), single_event_x_space(&format!("EventFrom{name}")).encode_to_vec())).collect();
    let (status, _, body) = served("reduce-stress", &hosts, &format!("tag=trace_viewer@&hosts={}&end_time_ms=5000.0&resolution=1", names.join(","))).await;
    assert_eq!(status, StatusCode::OK);
    let events = trace_events(&body);
    for name in &names {
        assert!(named(&events, &format!("EventFrom{name}")).is_some(), "Could not find event from {name}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_with_search_metadata() {
    let (status, body) = single_host("reduce-search-metadata", &create_test_x_space(2), "search_prefix=Sess&search_metadata=true").await;
    assert_eq!(status, StatusCode::OK);
    assert!(serde_json::from_str::<serde_json::Value>(&body).unwrap().get("traceEvents").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_with_search_prefix() {
    let (status, body) = single_host("reduce-search-prefix", &create_test_x_space(2), "search_prefix=Sess").await;
    assert_eq!(status, StatusCode::OK);
    assert!(serde_json::from_str::<serde_json::Value>(&body).unwrap().get("traceEvents").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_session_end_to_end() {
    let hosts = [("host1", create_test_x_space(1).encode_to_vec()), ("host2", create_test_x_space(1).encode_to_vec())];
    let (status, _, _) = served("process-session-e2e", &hosts, "tag=trace_viewer@&host=ALL_HOSTS&end_time_ms=2000.0&resolution=1").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_session_single_host() {
    assert_eq!(single_host("process-session-single", &create_test_x_space(1), "end_time_ms=2000.0&resolution=1").await.0, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_with_event_name() {
    let (status, body) = single_host("reduce-event-name", &create_test_x_space(2), "event_name=SessionRun&start_time_ms=1.2&duration_ms=0.05&unique_id=0").await;
    assert_eq!(status, StatusCode::OK);
    let complete: Vec<serde_json::Value> = trace_events(&body).into_iter().filter(|event| event["ph"] == "X").collect();
    assert_eq!(complete.len(), 1);
    assert_eq!(complete[0]["name"], "SessionRun");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_session_with_search_prefix() {
    let (status, body) = single_host("process-session-search", &create_test_x_space(2), "search_prefix=Sess").await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty() && serde_json::from_str::<serde_json::Value>(&body).unwrap().get("traceEvents").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_with_empty_x_space() {
    let (status, body) = single_host("map-empty", &create_test_x_space(0), "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(serde_json::from_str::<serde_json::Value>(&body).unwrap().get("traceEvents").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduce_single_host_pb_format() {
    let (status, _, body) = served("reduce-pb", &[("host1", create_test_x_space(2).encode_to_vec())], "tag=trace_viewer@&host=host1&format=pb").await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());
    assert!(serde_json::from_str::<serde_json::Value>(&body).is_err());
}
