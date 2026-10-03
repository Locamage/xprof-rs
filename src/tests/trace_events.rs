use super::legacy::{fetch, logdir};
use super::xspace::{V, XSpace};
use crate::trace::Trace;
use crate::{Settings, state};
use prost::Message;
use std::sync::atomic::{AtomicUsize, Ordering};

static SEARCHES: AtomicUsize = AtomicUsize::new(0);

const FLOW_END: u64 = 1;
const FLOW_START: u64 = 2;
const FLOW_MID: u64 = 3;

type FlowEvent<'a> = (&'a str, i64, i64, u64, u64);

fn levels(lines: &[(i64, &[FlowEvent])]) -> impl Fn(&str) -> u8 {
    let mut space = XSpace::default();
    let plane = space.plane("/host:CPU");
    for &(line, events) in lines {
        plane.named_line(line, "TestResource");
        for &(name, begin, end, flow, direction) in events {
            plane.event(line, name, begin, end - begin, &[("flow", V::Uint(flow << 2 | direction))]);
        }
    }
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    move |name: &str| trace.events.iter().find(|event| &*trace.names[event.name as usize] == name).unwrap().level
}

async fn search(space: &XSpace, query: &str) -> Vec<serde_json::Value> {
    let dir = logdir(&format!("search-{}", SEARCHES.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), space.encode_to_vec()).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let (_, _, body) = fetch(&state, &format!("/data/plugin/profile/data?run=run/s&tag=trace_viewer@&host=h&{query}")).await;
    std::fs::remove_dir_all(&dir).unwrap();
    let events = serde_json::from_str::<serde_json::Value>(&body).unwrap()["traceEvents"].as_array().unwrap().clone();
    events.into_iter().filter(|event| event["ph"] == "X").collect()
}

fn names(events: &[serde_json::Value]) -> Vec<&str> {
    events.iter().map(|event| event["name"].as_str().unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_metadata_keys_in_trie() {
    let mut space = XSpace::default();
    let plane = space.plane("/host:CPU");
    plane.named_line(2, "TestResource");
    plane.event(2, "ModelForward", 100, 100, &[("request_id1", "target_req_123".into()), ("bytes", V::Uint(1024))]);
    plane.event(2, "OtherEvent", 300, 100, &[("some_other_key", "req_ignored".into())]);
    let found = search(&space, "search_prefix=Model").await;
    assert_eq!(names(&found), ["ModelForward"]);
    assert!(found[0]["args"].get("request_id1").is_none());
    let found = search(&space, "search_prefix=Model&search_metadata=true").await;
    assert_eq!(names(&found), ["ModelForward"]);
    assert_eq!(found[0]["args"]["request_id1"], "target_req_123");
    assert!(names(&search(&space, "search_prefix=req_ignored").await).is_empty());
    let mut deduplication = XSpace::default();
    let plane = deduplication.plane("/host:CPU");
    plane.named_line(2, "TestResource");
    plane.event(2, "req_event", 500, 100, &[("request_id1", "req_val".into())]);
    assert_eq!(names(&search(&deduplication, "search_prefix=req").await), ["req_event"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_metadata_interned_and_edge_cases() {
    let mut space = XSpace::default();
    let plane = space.plane("/host:CPU");
    plane.named_line(2, "TestResource");
    plane.event(2, "ShortEventName", 100, 100, &[("request_id1", "SuperLongRequestIdValueThatWillBeInterned".into())]);
    plane.event(2, "IntReqEvent", 300, 100, &[("request_id1", V::Uint(99999))]);
    plane.event(2, "NoRawDataEvent", 500, 100, &[]);
    assert_eq!(names(&search(&space, "search_prefix=ShortEventName").await), ["ShortEventName"]);
    assert_eq!(names(&search(&space, "search_prefix=NoRawDataEvent").await), ["NoRawDataEvent"]);
}

#[test]
fn flow_events_zoom_level_assignment_and_safety_guard() {
    let level = levels(&[(2, &[("Dummy", 0, 100, 99, FLOW_START), ("EventA", 1000, 1100, 1, FLOW_START), ("EventB", 2000, 2000 + 2_000_000_000_000, 1, FLOW_END)])]);
    assert_eq!(level("EventB"), 0);
    assert_eq!(level("EventA"), 9);
}

#[test]
fn safety_guard_for_fallback_flow_events() {
    let level = levels(&[
        (2, &[("Dummy1", 0, 100, 99, FLOW_START), ("EventA", 1000, 1100, 1, FLOW_START), ("EventB", 2000, 2000 + 2_000_000_000_000, 1, FLOW_MID)]),
        (3, &[("Dummy2", 3_500_000_000_000, 3_500_000_000_100, 98, FLOW_START), ("EventC", 4_000_000_000_000, 4_000_000_000_100, 1, FLOW_END)]),
    ]);
    assert_eq!(level("EventB"), 0);
    assert_eq!(level("EventC"), 9);
}
