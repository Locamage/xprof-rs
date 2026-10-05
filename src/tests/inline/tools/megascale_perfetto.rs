use super::*;

#[test]
fn tiny_events_at_the_end_of_time() {
    let mut trace = Trace::default();
    let name = trace.strings.intern("fusion");
    let events = [i64::MAX - 5, i64::MAX - 4].map(|ts| Event { name, ts, dur: 100, ..Default::default() }).to_vec();
    trace.tpu.insert(0, vec![Track { name: "XLA Ops".into(), events }]);
    group_tiny_events(&mut trace);
    let events = &trace.tpu[&0][0].events;
    assert_eq!((events.len(), trace.strings.get(events[0].name), events[0].dur), (1, "2 events are hidden", 101));
}

#[test]
fn counter_sum_that_is_too_large() {
    let points = counter(vec![(0, i64::MAX), (1, i64::MAX), (1, 2)], i64::wrapping_add, Sample::Int);
    let values: Vec<(i64, i64)> = points.into_iter().map(|(ts, sample)| (ts, if let Sample::Int(value) = sample { value } else { 0 })).collect();
    assert_eq!(values, [(0, i64::MAX), (1, 0)]);
}
