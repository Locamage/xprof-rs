use super::legacy::{fetch, logdir};
use super::xspace::{V, XSpace};
use crate::trace::json::{View, render};
use crate::trace::{Options, Trace};
use crate::{Settings, state};

const FLOW_MID: u64 = 3;

fn trace_of(space: &XSpace) -> (Vec<u8>, Vec<crate::xplane::Plane>, Trace) {
    let (map, mut planes) = space.parsed();
    let trace = Trace::build(&mut planes, "localhost", &map);
    (map, planes, trace)
}

fn details(space: &XSpace, full_dma: bool) -> serde_json::Value {
    let (map, planes, trace) = trace_of(space);
    let options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma };
    let view = View { trace: &trace, map: &map, planes: &planes, events: trace.load(&options) };
    serde_json::from_slice::<serde_json::Value>(&render(&[view], full_dma, crate::trace::json::Args::Brief)).unwrap()["details"].clone()
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

#[test]
fn simple_atod_test() {
    use crate::trace::atod;
    for (text, value) in [(" 2 ", 2.0), ("+5", 5.0), ("\x0b3", 3.0), ("0x10", 16.0), ("-0x1.8p1", -3.0), ("0X.8", 0.5), ("1e400", f64::INFINITY), ("-0", 0.0), ("5.", 5.0)] {
        assert_eq!(atod(text), Some(value), "{text}");
    }
    for text in ["", "+-5", "++5", "1_0", "abc", "0x", "0x1p", "0xinf", "١"] {
        assert_eq!(atod(text), None, "{text}");
    }
    assert!(atod("nan(abc)").unwrap().is_nan() && atod("-inf") == Some(f64::NEG_INFINITY));
}

#[test]
fn double_to_uint64_test() {
    use crate::trace::{atou, to_u64};
    assert_eq!(to_u64(-1e9), 0u64.wrapping_sub(1_000_000_000));
    assert_eq!([to_u64(f64::NAN), to_u64(f64::INFINITY), to_u64(f64::NEG_INFINITY), to_u64(1e30), to_u64(-1e30)], [1 << 63; 5]);
    assert_eq!((to_u64(1.9), to_u64(-0.5), to_u64(1e19)), (1, 0, 10_000_000_000_000_000_000));
    assert_eq!((atou("18446744073709551615"), atou("1e3"), atou("-1"), atou("x")), (Some(u64::MAX), Some(1000), Some(u64::MAX), None));
}

#[test]
fn start_after_end_gives_an_instant_test() {
    let (_, _, trace) = trace_of(&device_space("/device:GPU:0"));
    let names = |start_ms: f64| {
        let options = Options { start_ms, end_ms: 0.0, resolution: 0.0, full_dma: false };
        trace.load(&options).iter().map(|&index| trace.names[trace.events[index as usize].name as usize].to_string()).collect::<Vec<_>>()
    };
    assert_eq!(names(1.5e-6), ["flow_event"]);
    assert_eq!(names(0.0), ["flow_event", "non_flow_event"]);
}

#[test]
fn search_takes_the_stack_frames_of_the_full_trace_test() {
    use crate::trace::json::Args;
    let mut space = XSpace::default();
    let plane = space.host();
    plane.event(1, "first", 1000, 100, &[("long_name", "the first long name of the trace".into())]);
    plane.event(1, "second", 2000, 100, &[("long_name", "the second long name of the trace".into())]);
    let (map, planes, trace) = trace_of(&space);
    let view = View { trace: &trace, map: &map, planes: &planes, events: trace.search("second", false) };
    let json: serde_json::Value = serde_json::from_slice(&render(&[view], false, Args::Search)).unwrap();
    let event = json["traceEvents"].as_array().unwrap().iter().find(|event| event["name"] == "second").unwrap();
    assert_eq!(json["stackFrames"].as_object().unwrap().len(), 2);
    assert_eq!(json["stackFrames"][event["sf"].to_string()]["name"], "the second long name of the trace");
}
