use super::xspace::XSpace;
use crate::trace::Trace;
use crate::xplane::{slice, stats};
use std::collections::{BTreeMap, HashMap};

const HOST_THREADS_DEVICE_ID: u32 = 701;
const PICOS_PER_SECOND: u64 = 1_000_000_000_000;

fn trace_of(text: &str) -> (Vec<u8>, Trace) {
    let (map, planes) = XSpace::text(text).parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    (map, trace)
}

fn events_by_name(trace: &Trace) -> Vec<(&str, &crate::trace::Event)> {
    trace.events.iter().map(|event| (&*trace.names[event.name as usize], event)).collect()
}

#[test]
fn counter_line() {
    let (map, trace) = trace_of(
        r#"planes {
             name: "/device:GPU:0"
             lines {
               name: "_counters_"
               events { metadata_id: 100 offset_ps: 1000000000000 stats { metadata_id: 200 uint64_value: 100 } }
               events { metadata_id: 100 offset_ps: 2000000000000 stats { metadata_id: 200 uint64_value: 200 } }
               events { metadata_id: 101 offset_ps: 1000000000000 stats { metadata_id: 201 uint64_value: 300 } }
               events { metadata_id: 101 offset_ps: 2000000000000 stats { metadata_id: 201 uint64_value: 400 } }
             }
             lines {
               id: 14
               name: "Stream #14(MemcpyH2D)"
               timestamp_ns: 500000000000
               events { metadata_id: 10 offset_ps: 0 duration_ps: 2000000000000 stats { metadata_id: 8 uint64_value: 100 } stats { metadata_id: 9 str_value: "$1" } }
               events { metadata_id: 10 offset_ps: 1000000000000 duration_ps: 500000000000 stats { metadata_id: 8 uint64_value: 200 } stats { metadata_id: 9 str_value: "abcd" } }
             }
             event_metadata { key: 10 value: { id: 10 name: "MemcpyD2D" } }
             event_metadata { key: 100 value: { id: 100 name: "Counter 1" } }
             event_metadata { key: 101 value: { id: 101 name: "Counter 2" } }
             stat_metadata { key: 8 value: { id: 8 name: "RemoteCall" } }
             stat_metadata { key: 9 value: { id: 8 name: "context_id" } }
             stat_metadata { key: 200 value: { id: 200 name: "counter_1" } }
             stat_metadata { key: 201 value: { id: 201 name: "counter_2" } }
           }"#,
    );
    let mut counters: HashMap<&str, BTreeMap<u64, Option<i64>>> = HashMap::new();
    for (name, event) in events_by_name(&trace).into_iter().filter(|(name, _)| name.contains("Counter")) {
        let value = stats(slice(&map, event.raw), 4, |_| true).next().and_then(|stat| stat.value.int());
        counters.entry(name).or_default().insert(event.ts, value);
    }
    let expected = HashMap::from([
        ("Counter 1", BTreeMap::from([(PICOS_PER_SECOND, Some(100)), (2 * PICOS_PER_SECOND, Some(200))])),
        ("Counter 2", BTreeMap::from([(PICOS_PER_SECOND, Some(300)), (2 * PICOS_PER_SECOND, Some(400))])),
    ]);
    assert_eq!(counters, expected);
}

#[test]
fn async_line() {
    let (_, trace) = trace_of(
        r#"planes {
             name: "/device:GPU:0"
             lines { id: 15 name: "Async XLA Ops" timestamp_ns: 2000000000 events { metadata_id: 11 offset_ps: 0 duration_ps: 1000000000000 } }
             event_metadata { key: 11 value: { id: 11 name: "Async Op" } }
           }"#,
    );
    let found: Vec<(u64, u64)> = events_by_name(&trace).into_iter().filter(|(name, _)| *name == "Async Op").map(|(_, event)| (event.ts, event.dur)).collect();
    assert_eq!(found, [(2 * PICOS_PER_SECOND, PICOS_PER_SECOND)]);
}

#[test]
fn subprocess_host_trace_converts_to_trace_events() {
    let (_, trace) = trace_of(
        r#"planes {
             name: "/host:CPU [100]"
             lines { id: 2 name: "Thread 2" timestamp_ns: 2000 events { metadata_id: 2 offset_ps: 0 duration_ps: 200 } }
             event_metadata { key: 2 value: { id: 2 name: "Event 2" } }
           }
           planes {
             name: "/host:CPU"
             lines { id: 1 name: "Thread 1" timestamp_ns: 1000 events { metadata_id: 1 offset_ps: 0 duration_ps: 100 } }
             event_metadata { key: 1 value: { id: 1 name: "Event 1" } }
           }"#,
    );
    let devices: HashMap<&str, u32> = events_by_name(&trace).into_iter().map(|(name, event)| (name, event.device)).collect();
    assert_eq!(devices, HashMap::from([("Event 1", HOST_THREADS_DEVICE_ID), ("Event 2", HOST_THREADS_DEVICE_ID + 1)]));
}
