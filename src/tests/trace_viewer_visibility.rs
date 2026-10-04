use crate::trace::{Event, NONE_FLOW, Row, Visibility};

const RESOURCE_ID: u32 = 1;
const SRC_RESOURCE_ID: u32 = 2;
const DST_RESOURCE_ID: u32 = 4;
const FLOW_MID_DIRECTION: u64 = 3;

type TimedEvent<'a> = (&'a str, i64, i64, Option<u64>);

fn visibility(resolution_ps: u64) -> Visibility {
    Visibility { resolution: resolution_ps, rows: vec![Row::default(); 8], flows: vec![None; 8] }
}

fn complete(begin_ps: u64, duration_ps: u64) -> Event {
    Event { ts: begin_ps, dur: duration_ps, resource: RESOURCE_ID, track: RESOURCE_ID, flow: NONE_FLOW, ..Default::default() }
}

fn flow(begin_ps: u64, duration_ps: u64, flow_id: u64, resource_id: u32) -> Event {
    Event { ts: begin_ps, dur: duration_ps, resource: resource_id, track: resource_id, flow: flow_id, ..Default::default() }
}

#[test]
fn complete_events_downsampling() {
    let mut v = visibility(100);
    assert!(v.visible_at_resolution(&complete(950, 50)));
    assert!(!v.visible_at_resolution(&complete(1050, 50)));
    assert!(v.visible_at_resolution(&complete(1055, 200)));
    assert!(v.visible_at_resolution(&complete(1355, 50)));
}

#[test]
fn complete_nested_events_downsampling() {
    let mut v = visibility(100);
    assert!(v.visible_at_resolution(&complete(1000, 200)));
    assert!(v.visible_at_resolution(&complete(1200, 190)));
    assert!(v.visible_at_resolution(&complete(1250, 20)));
    assert!(!v.visible_at_resolution(&complete(1270, 20)));
    assert!(v.visible_at_resolution(&complete(1290, 100)));
}

#[test]
fn flow_events_downsampling() {
    let mut v = visibility(100);
    assert!(v.visible_at_resolution(&flow(1000, 50, 1, SRC_RESOURCE_ID)));
    assert!(!v.visible_at_resolution(&flow(1050, 50, 2, SRC_RESOURCE_ID)));
    assert!(v.visible_at_resolution(&flow(1100, 50, 3, SRC_RESOURCE_ID)));
    assert!(v.visible_at_resolution(&flow(1100, 50, 1, DST_RESOURCE_ID)));
    assert!(!v.visible_at_resolution(&flow(1200, 52, 2, DST_RESOURCE_ID)));
    assert!(v.visible_at_resolution(&flow(1252, 10, 3, DST_RESOURCE_ID)));
    assert!(v.visible_at_resolution(&complete(1300, 50)));
    assert!(!v.visible_at_resolution(&complete(1350, 50)));
    assert!(!v.visible_at_resolution(&complete(1400, 50)));
    assert!(v.visible_at_resolution(&flow(1600, 50, 4, RESOURCE_ID)));
    assert!(v.visible_at_resolution(&flow(1700, 52, 5, RESOURCE_ID)));
    assert!(!v.visible_at_resolution(&flow(1752, 10, 6, RESOURCE_ID)));
}

fn visible(planes: &[(&str, &[TimedEvent])], start_ps: u64, end_ps: u64, full_dma: bool) -> Vec<String> {
    let mut space = super::xspace::XSpace::default();
    for (plane, events) in planes {
        let plane = space.plane(plane);
        plane.named_line(1, "XLA Ops");
        for &(name, begin, duration, flow) in *events {
            let stats: Vec<(&str, super::xspace::V)> = flow.map(|flow| ("flow", super::xspace::V::Uint(flow << 2 | FLOW_MID_DIRECTION))).into_iter().collect();
            plane.event(1, name, begin, duration, &stats);
        }
    }
    let (map, planes) = space.parsed();
    let trace = crate::trace::Trace::build(&planes, "localhost", &map);
    let options = crate::trace::Options { start_ms: start_ps as f64 / 1e9, end_ms: end_ps as f64 / 1e9, resolution: 0.0, full_dma };
    let mut names: Vec<String> = trace.load(&options).iter().map(|&index| trace.names[trace.events[index as usize].name as usize].to_string()).collect();
    names.sort();
    names
}

#[test]
fn visibility_no_downsampling() {
    let events: &[TimedEvent] = &[
        ("i999", 999, 0, None),
        ("i1000", 1000, 0, None),
        ("i1500", 1500, 0, None),
        ("i2000", 2000, 0, None),
        ("i2001", 2001, 0, None),
        ("c900_99", 900, 99, None),
        ("c900_100", 900, 100, None),
        ("c1450_100", 1450, 100, None),
        ("c2000_50", 2000, 50, None),
        ("c2001_50", 2001, 50, None),
    ];
    assert_eq!(visible(&[("/host:CPU", events)], 1000, 2000, false), ["c1450_100", "c2000_50", "c900_100", "i1000", "i1500", "i2000"]);
}

#[test]
fn test_trace_visibility_filter() {
    let events: &[TimedEvent] = &[("event", 1000, 100, None), ("event2", 2000, 100, None), ("event3", 900, 50, None), ("event4", 1000, 100, Some(1))];
    assert_eq!(visible(&[("/device:TPU:0", events)], 1000, 2000, false), ["event", "event2"]);
    assert_eq!(visible(&[("/device:TPU:0", events)], 1000, 2000, true), ["event", "event2", "event4"]);
    assert_eq!(visible(&[("/device:GPU:0", events)], 1000, 2000, true), ["event", "event2", "event4"]);
    let non_core: &[TimedEvent] = &[("event4", 1000, 100, Some(1))];
    let core: &[TimedEvent] = &[("event", 1000, 100, None)];
    assert_eq!(visible(&[("/device:TPU:0", core), ("/device:CUSTOM:#Chip TPU Non-Core HBM", non_core)], 1000, 2000, false), ["event"]);
}
