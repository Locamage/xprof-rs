use super::xspace::{V, XSpace, op_stats, proto_bytes};

pub type Interval = (i64, i64, bool);

pub fn add_tracker(space: &mut XSpace, chip: Option<u32>, intervals: &[Interval]) {
    let ordinal = space.planes.len() as i32;
    let plane = space.tpu(ordinal, "TPU v4", 0.0, 0.0, None);
    if let Some(chip) = chip {
        plane.add_stat("core_details", V::Bytes(proto_bytes("tensorflow.profiler.CoreDetails", &format!("local_chip_id: {chip}"))));
    }
    plane.named_line(0, "XLA Ops");
    for (index, &(begin, end, active)) in intervals.iter().enumerate() {
        plane.event(0, &format!("op.{index}"), begin, end - begin, &[("hlo_category", if active { "call" } else { "infeed" }.into())]);
    }
}

pub fn active_and_idle(space: XSpace) -> (u64, u64) {
    let stats = op_stats(&[space]).unwrap();
    (stats.extra.busy_ps[0], stats.extra.idle_ps[0])
}

fn tracker(intervals: &[Interval]) -> (u64, u64) {
    let mut space = XSpace::default();
    add_tracker(&mut space, None, intervals);
    active_and_idle(space)
}

#[test]
fn non_overlapping_intervals_test() {
    let (active, idle) = tracker(&[(10, 20, true), (30, 40, true)]);
    assert_eq!((active, idle, active + idle), (20, 10, 30));
    assert!((active as f64 / (active + idle) as f64 - 0.6666).abs() <= 0.0001);
}

#[test]
fn overlapping_intervals_test() {
    let (active, idle) = tracker(&[(10, 20, true), (30, 40, true), (20, 35, true)]);
    assert_eq!((active, idle, active + idle), (30, 0, 30));
    assert_eq!(active as f64 / (active + idle) as f64, 1.0);
}

#[test]
fn duty_cycle_test_with_included_intervals() {
    let (active, idle) = tracker(&[(10, 40, true), (20, 30, true)]);
    assert_eq!((active, idle, active + idle), (30, 0, 30));
    assert_eq!(active as f64 / (active + idle) as f64, 1.0);
}

#[test]
fn union_test() {
    let mut space = XSpace::default();
    add_tracker(&mut space, Some(0), &[(0, 10, true), (20, 30, true)]);
    add_tracker(&mut space, Some(0), &[(10, 20, true), (30, 40, true)]);
    let (active, idle) = active_and_idle(space);
    assert_eq!((active, idle, active + idle), (40, 0, 40));
}

#[test]
fn overlapping_mixed_intervals_test() {
    assert_eq!(active_and_idle(XSpace::default()).0, 0);
    assert_eq!(tracker(&[(10, 20, true), (20, 30, false)]), (10, 10));
}
