use super::xspace::{V, XSpace};
use crate::trace::delta::{CounterValue, Response, render};
use crate::trace::json::View;
use crate::trace::{Options, Trace};
use prost::Message;
use std::io::Read;

fn decode(compressed: &[u8]) -> Response {
    let mut raw = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(compressed).unwrap().read_to_end(&mut raw).unwrap();
    let mut response = Response::decode(&raw[..]).unwrap();
    let metadata = response.metadata.as_mut().unwrap();
    metadata.processes.sort_by_key(|process| process.id);
    metadata.processes.iter_mut().for_each(|process| process.threads.sort_by_key(|thread| thread.id));
    for event in response.complete_events.iter_mut().chain(&mut response.async_events).flat_map(|series| &mut series.event_metadata) {
        event.flow_id = 0;
    }
    response
}

#[test]
fn delta_series_protobuf_matches_xprofs_response() {
    let dir = super::legacy::logdir("delta");
    let file = dir.join("run/plugins/profile/s/tpu-vm-demo-host-0.xplane.pb");
    std::fs::write(&file, include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let host = crate::load_host(&file).unwrap();
    let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 8000.0, full_dma: false };
    let view = View { trace: &host.trace, map: &host.map, planes: &host.planes, events: host.trace.load(&options) };
    assert_eq!(decode(&render(&[view], Some(false))), decode(include_bytes!("../../tests/data/trace_viewer_delta.pb.zst")));
    std::fs::remove_dir_all(&dir).unwrap();
}

const FLOW_START: u64 = 2;
const TF_EXECUTOR_CATEGORY: u64 = 2;

fn response(space: &XSpace, full_dma: Option<bool>) -> Response {
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma: true };
    let view = View { trace: &trace, map: &map, planes: &planes, events: trace.load(&options) };
    let mut raw = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(&render(&[view], full_dma)[..]).unwrap().read_to_end(&mut raw).unwrap();
    Response::decode(&raw[..]).unwrap()
}

fn async_flow(id: u64) -> V {
    V::Uint(TF_EXECUTOR_CATEGORY << 58 | id << 2 | FLOW_START)
}

#[test]
fn converts_complete_events_and_deltas() {
    let mut space = XSpace::default();
    let plane = space.host();
    plane.named_line(1, "Thread 1");
    plane.named_line(2, "Thread 2");
    for _ in 0..42 {
        plane.event(2, "Earlier", 1000, 600, &[]);
    }
    plane.event(1, "Compute", 1000, 500, &[]);
    plane.event(1, "Compute", 2000, 300, &[]);
    let response = response(&space, Some(false));
    let series = response.complete_events.iter().find(|series| series.metadata.as_ref().unwrap().thread_id == 1).unwrap();
    assert_eq!(series.metadata.as_ref().unwrap().process_id, 1701);
    assert_eq!(series.deltas, [1000, 1000]);
    assert_eq!(series.durations, [500, 300]);
    assert_eq!(series.event_metadata.iter().map(|metadata| metadata.serial).collect::<Vec<_>>(), [42, 0]);
    let process = &response.metadata.as_ref().unwrap().processes[0];
    assert_eq!(process.name, "localhost /host:CPU");
    assert_eq!(process.threads.iter().map(|thread| thread.name.as_str()).collect::<Vec<_>>(), ["Thread 1", "Thread 2"]);
}

#[test]
fn converts_counter_events() {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    plane.named_line(1, "_counters_");
    plane.event(1, "MyCounter", 1000, 0, &[("value", V::Uint(1234))]);
    plane.event(1, "MyCounter", 1500, 0, &[("value", V::Uint(5678))]);
    let response = response(&space, None);
    assert_eq!(response.counter_events.len(), 1);
    let series = &response.counter_events[0];
    assert_eq!(series.metadata.as_ref().unwrap().process_id, 1);
    assert_eq!(series.deltas, [1000, 500]);
    assert_eq!(series.event_metadata.iter().map(|metadata| metadata.counter_value.clone()).collect::<Vec<_>>(), [Some(CounterValue::Uint(1234)), Some(CounterValue::Uint(5678))]);
}

#[test]
fn converts_async_events() {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    plane.named_line(1, "Async XLA Ops");
    plane.event(1, "AsyncOp", 1000, 500, &[("flow", async_flow(1001)), ("_a", V::Uint(1)), ("group_id", 42.into())]);
    plane.event(1, "AsyncOp", 2000, 100, &[("flow", async_flow(1002)), ("_a", V::Uint(1)), ("group_id", 42.into())]);
    let response = response(&space, None);
    assert_eq!(response.async_events.len(), 1);
    let series = &response.async_events[0];
    assert_eq!(series.metadata.as_ref().unwrap().process_id, 1);
    assert_eq!(series.deltas, [1000, 1000]);
    assert_eq!(series.durations, [500, 100]);
    let (first, second) = (&series.event_metadata[0], &series.event_metadata[1]);
    assert_eq!((first.flow_id, first.group_id), (1001, 42));
    assert!(first.flow_category > 0 && (first.flow_category as usize) < response.interned_strings.len());
    assert_eq!((second.flow_id, second.flow_category), (1002, first.flow_category));
}

#[test]
#[allow(clippy::approx_constant)]
fn converts_mixed_events() {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    plane.named_line(1, "Stream");
    plane.named_line(2, "Async XLA Ops");
    plane.named_line(3, "_counters_");
    plane.event(1, "Compute", 1000, 100, &[]);
    plane.event(1, "Compute", 1200, 200, &[]);
    plane.event(2, "AsyncOp", 1500, 200, &[("flow", async_flow(1)), ("_a", V::Uint(1))]);
    plane.event(2, "AsyncOp", 1800, 100, &[("flow", async_flow(2)), ("_a", V::Uint(1))]);
    plane.event(3, "Memory", 2000, 0, &[("value", 3.14.into())]);
    plane.event(3, "Memory", 2500, 0, &[("value", 6.28.into())]);
    let response = response(&space, None);
    assert_eq!((response.complete_events.len(), response.async_events.len(), response.counter_events.len()), (1, 1, 1));
    assert_eq!(response.complete_events[0].deltas, [1000, 200]);
    assert_eq!(response.async_events[0].deltas, [1500, 300]);
    assert_eq!(response.counter_events[0].deltas, [2000, 500]);
    let values: Vec<Option<CounterValue>> = response.counter_events[0].event_metadata.iter().map(|metadata| metadata.counter_value.clone()).collect();
    assert_eq!(values, [Some(CounterValue::Double(3.14)), Some(CounterValue::Double(6.28))]);
}

#[test]
fn populates_details() {
    let mut space = XSpace::default();
    space.tpu(0, "TPU v4", 0.0, 0.0, None).event(1, "op", 0, 1, &[]);
    let details = |full_dma: Option<bool>| response(&space, full_dma).details.into_iter().map(|detail| (detail.name, detail.value)).collect::<Vec<_>>();
    assert_eq!(details(Some(true)), [("mpmd_pipeline_view".to_string(), false), ("full_dma".to_string(), true)]);
    assert!(details(None).is_empty());
}

#[test]
fn populates_full_timespan() {
    let mut space = XSpace::default();
    space.host().event(1, "op", 12_345_000, 67_890_000 - 12_345_000, &[]);
    let response = response(&space, None);
    assert_eq!((response.full_timespan_start_ps, response.full_timespan_end_ps), (Some(12_345_000), Some(67_890_000)));
    let empty = super::delta::response(&XSpace::default(), None);
    assert_eq!((empty.full_timespan_start_ps, empty.full_timespan_end_ps), (None, None));
}

#[test]
fn suppresses_sort_index_for_custom_sort_resources() {
    let mut space = XSpace::default();
    let plane = space.gpu(5);
    plane.named_line(30, "Standard Resource");
    plane.event(30, "op", 0, 1, &[]);
    let response = response(&space, None);
    let process = &response.metadata.as_ref().unwrap().processes[0];
    assert_eq!(process.threads.iter().map(|thread| (thread.id, thread.sort_index)).collect::<Vec<_>>(), [(30, Some(30))]);
}
