use super::*;
use crate::tests::legacy::{bytes, entry, number};

#[test]
fn byte_counts_and_steps_that_are_too_large() {
    let int = |id: u64, value: i64| bytes(4, &[number(1, id), number(4, value as u64)].concat());
    let event = |offset: u64, stats: &[Vec<u8>]| bytes(4, &[number(1, 1), number(2, offset), number(3, 1), stats.concat()].concat());
    let first = event(0, &[int(1, i64::MAX), int(2, i64::MAX), int(3, i64::MAX), int(4, i64::MAX)]);
    let line = [number(1, 1), bytes(2, b"allocator"), first, event(10, &[int(5, i64::MIN)])].concat();
    let names = ["bytes_reserved", "bytes_allocated", "bytes_available", "group_id", "allocation_bytes"];
    let plane =
        [bytes(2, HOST_PLANE.as_bytes()), bytes(3, &line), entry(4, 1, "MemoryAllocation")].into_iter().chain((1..).zip(names).map(|(id, name)| entry(5, id, name))).collect::<Vec<_>>().concat();
    let map = bytes(1, &plane);
    let json = json(&crate::xplane::parse(&map).unwrap(), &map);
    assert!(json.contains(r#""allocationBytes":"-9223372036854775808","address":"0","tfOpName":"","stepId":"-9223372036854775808""#), "{json}");
}
