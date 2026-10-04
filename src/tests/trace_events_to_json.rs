use super::legacy::same;
use super::xspace::{V, XSpace};
use crate::json::{View, micros, quoted, render, write_event};
use crate::trace::{Event, FLOW_END, FLOW_MID, FLOW_START, NONE_FLOW, NONE_RESOURCE, Options, Trace};
use crate::xplane::NONE_GROUP;
use std::collections::{BTreeMap, HashMap, HashSet};

fn trace_of(names: &[&str], flow_ids: &[u64]) -> Trace {
    Trace {
        devices: BTreeMap::new(),
        names: names.iter().map(|name| (*name).into()).collect(),
        events: Vec::new(),
        min_ps: 0,
        max_ps: 0,
        levels: Vec::new(),
        ties: Vec::new(),
        tpu_devices: HashSet::new(),
        dma_devices: HashSet::new(),
        long_names: HashMap::new(),
        steps: HashMap::new(),
        tracks: 0,
        flow_ids: flow_ids.to_vec(),
        args: Vec::new(),
        stack_frames: String::new(),
    }
}

fn event(resource: u32, ts: u64, dur: u64, flow: u64, flow_entry: u8) -> Event {
    Event { ts, dur, flow, flow_entry, group: NONE_GROUP, resource, device: 1, ..Default::default() }
}

fn written(trace: &Trace, event: &Event) -> String {
    let mut out = String::new();
    write_event(&mut out, trace, event, event.device, None, None, false);
    out
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap()
}

fn rendered(space: &XSpace) -> String {
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    let view = View { trace: &trace, map: &map, planes: &planes, events: trace.load(&Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma: false }) };
    String::from_utf8(render(&[view], false, false)).unwrap()
}

fn counter_space(events: &[(&str, &str, i64, i64)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.plane("/device:GPU:0");
    plane.named_line(1, "_counters_");
    for &(name, arg, ts, value) in events {
        plane.event(1, name, ts, 0, &[(arg, value.into())]);
    }
    space
}

#[test]
fn picos_to_micros_test() {
    for (ps, expected) in [(1_000_000, 1.0), (1, 1e-6), (1_234_567, 1.234567)] {
        let mut out = String::new();
        micros(&mut out, ps);
        assert_eq!(out.parse::<f64>().unwrap(), expected);
    }
}

#[test]
fn json_escape_test() {
    assert_eq!(quoted(""), r#""""#);
    assert_eq!(quoted("abc"), r#""abc""#);
    assert_eq!(quoted("a\"b\\c"), r#""a\"b\\c""#);
    assert_eq!(quoted("a\nb\rc\td\u{8}e\u{c}f"), r#""a\nb\rc\td\be\ff""#);
    assert_eq!(quoted("a<b"), format!("\"a{}u003cb\"", '\\'));
    assert_eq!(quoted("b>c"), format!("\"b{}u003ec\"", '\\'));
    assert_eq!(quoted("c&d"), format!("\"c{}u0026d\"", '\\'));
    assert_eq!(quoted("\u{2028}"), format!("\"{}u2028\"", '\\'));
    assert_eq!(quoted("\u{2029}"), format!("\"{}u2029\"", '\\'));
}

#[test]
fn json_event_counter_test() {
    let json = rendered(&counter_space(&[("counter_a", "arg1", 1_000_000, 1), ("counter_b", "arg1", 2_000_000, 2)]));
    assert!(json.ends_with("\"totalCounterEvents\":2}"), "{json}");
}

#[test]
fn write_details_test() {
    let mut space = XSpace::default();
    space.plane("/device:TPU:0").event(1, "op", 0, 1, &[]);
    assert!(rendered(&space).contains(r#""details":[{"name":"mpmd_pipeline_view","value":false},{"name":"full_dma","value":false}],"#));
}

#[test]
fn write_returned_events_size_test() {
    let mut space = XSpace::default();
    for offset in 0..123 {
        space.host().event(1, "op", offset * 10, 1, &[]);
    }
    assert!(rendered(&space).contains(r#""returnedEventsSize":123,"#));
}

#[test]
fn write_filtered_by_visibility_test() {
    let mut space = XSpace::default();
    space.host().event(1, "op", 0, 1, &[]);
    assert!(rendered(&space).contains(r#""filteredByVisibility":true,"#));
}

#[test]
fn write_trace_full_timespan_test() {
    let mut space = XSpace::default();
    space.host().event(1, "op", 1_000_000_000, 1_000_000_000, &[]);
    assert!(rendered(&space).contains(r#""fullTimespan":[1,2],"#));
}

#[test]
fn write_stack_frames_test() {
    let mut space = XSpace::default();
    let plane = space.host();
    plane.event(1, "first", 0, 10, &[("long_name", "stack1 is a long stack frame".into())]);
    plane.event(1, "second", 20, 10, &[("hlo_text", "stack2 is a long stack frame".into())]);
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    let view = View { trace: &trace, map: &map, planes: &planes, events: vec![0, 1] };
    let json = String::from_utf8(render(&[view], false, true)).unwrap();
    assert!(json.contains(r#""1":{"name":"stack1 is a long stack frame"}"#), "{json}");
    assert!(json.contains(r#""2":{"name":"stack2 is a long stack frame"}"#), "{json}");
}

#[test]
fn complete_event_test() {
    let trace = trace_of(&["complete_event"], &[]);
    let out = written(&trace, &event(2, 1_000_000, 500_000, NONE_FLOW, 0));
    assert!(same(&json(&out), &json(r#"{"pid":1,"tid":2,"name":"complete_event","ts":1,"dur":0.5,"ph":"X","args":{"uid":0}}"#)), "{out}");
}

#[test]
fn complete_event_with_flow_test() {
    let trace = trace_of(&["complete_event_with_flow"], &[123]);
    let out = written(&trace, &event(2, 1_000_000, 500_000, 0, FLOW_START));
    assert!(same(&json(&out), &json(r#"{"pid":1,"tid":2,"name":"complete_event_with_flow","ts":1,"dur":0.5,"bind_id":123,"flow_out":true,"ph":"X","args":{"uid":0}}"#)), "{out}");
}

#[test]
fn counter_event_test() {
    let json = rendered(&counter_space(&[("counter_event", "arg1", 1_000_000, 100)]));
    assert!(json.contains(r#"{"pid":1001,"name":"counter_event","ph":"C","event_stats":"arg1","entries":[[1,100]]}"#), "{json}");
}

#[test]
fn async_event_test() {
    let trace = trace_of(&["async_event"], &[456]);
    let out = written(&trace, &event(NONE_RESOURCE, 1_000_000, 0, 0, FLOW_START));
    assert!(same(&json(&out), &json(r#"{"pid":1,"name":"async_event","ts":1,"id":456,"cat":"","ph":"b","args":{"uid":0}}"#)), "{out}");
}

#[test]
fn is_matching_last_counter_event_test() {
    let json = rendered(&counter_space(&[("counter_event", "arg1", 1_000_000, 100), ("counter_event", "arg1", 2_000_000, 200)]));
    assert_eq!(json.matches(r#""name":"counter_event","ph":"C""#).count(), 1, "{json}");
    let mut space = counter_space(&[("counter_event", "arg1", 1_000_000, 100)]);
    space.plane("/device:GPU:1").named_line(1, "_counters_");
    space.plane("/device:GPU:1").event(1, "counter_event", 2_000_000, 0, &[("arg1", 200.into())]);
    assert_eq!(rendered(&space).matches(r#""name":"counter_event","ph":"C""#).count(), 2);
}

#[test]
fn write_event_flows() {
    let trace = trace_of(&["flow_event"], &[100]);
    let start = written(&trace, &Event { flow_cat: 1, ..event(2, 1_000_000, 1000, 0, FLOW_START) });
    assert!(start.contains(r#""flow_out":true"#) && start.contains(r#""bind_id":100"#), "{start}");
    let mid = written(&trace, &event(2, 2_000_000, 1000, 0, FLOW_MID));
    assert!(mid.contains(r#""flow_in":true"#) && mid.contains(r#""flow_out":true"#), "{mid}");
    let end = written(&trace, &event(2, 3_000_000, 1000, 0, FLOW_END));
    assert!(end.contains(r#""flow_in":true"#) && !end.contains(r#""flow_out":true"#), "{end}");
}

#[test]
fn write_event_async_flow_mid() {
    let trace = trace_of(&["async_event"], &[456]);
    let out = written(&trace, &event(NONE_RESOURCE, 1_000_000, 500_000, 0, FLOW_MID));
    assert!(out.contains(r#""ph":"b""#) && out.contains(r#""ph":"e""#), "{out}");
    assert!(out.contains(r#""ts":1"#) && out.contains(r#""ts":1.5"#), "{out}");
}

#[test]
#[allow(clippy::approx_constant)]
fn write_event_args() {
    let mut space = XSpace::default();
    let plane = space.host();
    let stats = [
        ("str_arg", V::from("value")),
        ("int_arg", 42.into()),
        ("uint_arg", 100u64.into()),
        ("double_arg", 3.14.into()),
        ("ref_arg", V::Ref("ref_value".into())),
        ("long_name", "@@stack_frame and more text".into()),
    ];
    plane.event(1, "args_event", 1000, 1000, &stats);
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    let (mut out, mut frames) = (String::new(), Vec::new());
    let args_event = Event { group: 10, serial: 123_456, ..trace.events[0] };
    write_event(&mut out, &trace, &args_event, 1, Some((&planes[0], &map, &mut frames)), None, false);
    for expected in
        [r#""args":{"#, r#""group_id":10"#, r#""str_arg":"value""#, r#""int_arg":42"#, r#""uint_arg":100"#, r#""double_arg":3.14"#, r#""ref_arg":"ref_value""#, r#""sf":1"#, r#""z":123456"#]
    {
        assert!(out.contains(expected), "{expected} in {out}");
    }
    assert_eq!(frames, ["@@stack_frame and more text"]);
}

#[test]
fn counter_events_grouping_test() {
    let json = rendered(&counter_space(&[("counter", "val", 1_000_000, 10), ("counter", "val", 2_000_000, 20)]));
    assert!(json.contains(r#""ph":"C""#), "{json}");
    assert!(json.contains(r#""entries":[[1,10],[2,20]]}"#), "{json}");
}

#[test]
fn integration_test() {
    let mut space = XSpace::default();
    space.plane("/device:GPU:0").named_line(1, "thread1");
    space.plane("/device:GPU:0").event(1, "event1", 1000, 1000, &[]);
    let json = rendered(&space);
    for expected in [r#""traceEvents":["#, r#""process_name""#, r#""thread_name""#] {
        assert!(json.contains(expected), "{expected} in {json}");
    }
}
