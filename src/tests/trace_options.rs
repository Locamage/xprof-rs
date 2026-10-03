use super::legacy::{fetch, logdir};
use super::xspace::{V, XSpace};
use crate::json::{View, render};
use crate::trace::{Options, Trace};
use crate::{Settings, state};

const FLOW_MID: u64 = 3;

fn trace_of(space: &XSpace) -> (Vec<u8>, Vec<crate::xplane::Plane>, Trace) {
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    (map, planes, trace)
}

fn details(space: &XSpace, full_dma: bool) -> serde_json::Value {
    let (map, planes, trace) = trace_of(space);
    let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma };
    let view = View { trace: &trace, map: &map, planes: &planes, events: trace.load(&options) };
    serde_json::from_slice::<serde_json::Value>(&render(&[view], full_dma, false)).unwrap()["details"].clone()
}

fn device_space(name: &str) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.plane(name);
    plane.named_line(1, "XLA Ops");
    plane.event(1, "flow_event", 1000, 100, &[("flow", V::Uint(123 << 2 | FLOW_MID))]);
    plane.event(1, "non_flow_event", 2000, 100, &[]);
    space
}

fn loaded(space: &XSpace, full_dma: bool) -> Vec<String> {
    let (_, _, trace) = trace_of(space);
    let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma };
    trace.load(&options).iter().map(|&index| trace.names[trace.events[index as usize].name as usize].to_string()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trace_options_from_tool_options_test() {
    let dir = logdir("trace-options");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let full_dma = |query: &'static str| {
        let state = state.clone();
        async move {
            let (_, _, body) = fetch(&state, &format!("/data/plugin/profile/data?run=run/s&tag=trace_viewer@&host=h{query}")).await;
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["details"].clone()
        }
    };
    let expected = |dma: bool| serde_json::json!([{"name": "mpmd_pipeline_view", "value": false}, {"name": "full_dma", "value": dma}]);
    assert_eq!(full_dma("").await, expected(false));
    assert_eq!(full_dma("&full_dma=true&enable_legacy_dcn=true&mpmd_pipeline_view=true").await, expected(true));
    assert_eq!(full_dma("&full_dma=True").await, expected(true));
    assert_eq!(full_dma("&full_dma=false&enable_legacy_dcn=false&mpmd_pipeline_view=false").await, expected(false));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn trace_options_to_details_test() {
    let mpmd_only = serde_json::json!([{"name": "mpmd_pipeline_view", "value": false}]);
    assert_eq!(details(&device_space("/host:CPU"), true), mpmd_only);
    assert_eq!(details(&device_space("/device:TPU:0"), true), serde_json::json!([{"name": "mpmd_pipeline_view", "value": false}, {"name": "full_dma", "value": true}]));
    assert_eq!(details(&device_space("/device:GPU:0"), true), mpmd_only);
    assert_eq!(details(&device_space("/device:TPU:0"), false), serde_json::json!([{"name": "mpmd_pipeline_view", "value": false}, {"name": "full_dma", "value": false}]));
}

#[test]
fn is_tpu_trace_test() {
    assert!(trace_of(&XSpace::default()).2.tpu_devices.is_empty());
    assert!(!trace_of(&device_space("/device:TPU:0")).2.tpu_devices.is_empty());
    assert!(trace_of(&device_space("/device:GPU:0")).2.tpu_devices.is_empty());
}

#[test]
fn filter_test() {
    let mut space = device_space("/device:TPU:0");
    space.plane("/device:TPU_COMPILER:0").event(1, "compiler_event", 3000, 100, &[]);
    assert_eq!(loaded(&space, false), ["non_flow_event"]);
    assert_eq!(loaded(&space, true), ["flow_event", "non_flow_event"]);
}

#[test]
fn non_tpu_trace_test() {
    assert_eq!(loaded(&device_space("/device:GPU:0"), false), ["flow_event", "non_flow_event"]);
}
