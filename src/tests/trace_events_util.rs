use super::xspace::XSpace;
use crate::trace::Trace;

const BAD_TIMESTAMP: i64 = i64::MAX;

fn bounds(spans: &[(i64, i64)]) -> (u64, u64) {
    let mut space = XSpace::default();
    let plane = space.plane("/device:TPU:0");
    plane.named_line(1, "XLA Ops");
    for &(begin, end) in spans {
        plane.event(1, "op", begin, end.wrapping_sub(begin).max(0), &[]);
    }
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    (trace.min_ps, trace.max_ps)
}

#[test]
fn empty_trace_sets_bounds() {
    assert_eq!(bounds(&[(10, 20)]), (10, 20));
}

#[test]
fn larger_span_expands_bounds() {
    assert_eq!(bounds(&[(10, 20), (5, 25)]), (5, 25));
    assert_eq!(bounds(&[(10, 20), (5, 25), (5, BAD_TIMESTAMP)]), (5, BAD_TIMESTAMP as u64));
}

#[test]
fn smaller_span_does_not_change_bounds() {
    assert_eq!(bounds(&[(10, 20), (12, 18)]), (10, 20));
}

#[test]
fn bad_timestamps_are_ignored() {
    assert_eq!(bounds(&[(10, 20), (5, BAD_TIMESTAMP.wrapping_add(1))]), (5, 20));
    assert_eq!(bounds(&[(10, 20), (5, BAD_TIMESTAMP.wrapping_add(1)), (BAD_TIMESTAMP.wrapping_add(1), 25)]), (5, 20));
}
