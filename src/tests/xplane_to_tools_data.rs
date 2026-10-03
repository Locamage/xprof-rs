use super::legacy::served;
use super::xspace::XSpace;
use prost::Message;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trace_viewer_search_metadata_option() {
    let mut space = XSpace::default();
    let plane = space.host();
    plane.named_line(1, "Test Line");
    plane.event(1, "TraceContext", 1_000_000_000, 100_000_000, &[("MetadataWithFoo", 123.into())]);
    let hosts = [("test_host", space.encode_to_vec())];
    let query = "tag=trace_viewer@&host=test_host&search_prefix=TraceContext&resolution=0&start_time_ms=0.0&end_time_ms=0.0&duration_ms=0.0";
    let (_, _, data) = served("tools-data-metadata", &hosts, &format!("{query}&search_metadata=true")).await;
    assert!(data.contains("TraceContext") && data.contains("MetadataWithFoo"), "{data}");
    let (_, _, data) = served("tools-data-no-metadata", &hosts, &format!("{query}&search_metadata=false")).await;
    assert!(data.contains("TraceContext") && !data.contains("MetadataWithFoo"), "{data}");
}
