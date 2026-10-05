use super::*;
use crate::tests::legacy::{bytes, entry, number};

#[test]
fn enqueue_op_inside_an_enqueue_op() {
    let event = |offset: u64, duration: u64| bytes(4, &[number(1, 1), number(2, offset), number(3, duration)].concat());
    let line = [number(1, 7), bytes(2, b"worker"), event(0, 100), event(10, 10), event(200, 10)].concat();
    let space = bytes(1, &[bytes(2, HOST_PLANE.as_bytes()), bytes(3, &line), entry(4, 1, "x:InfeedEnqueueTuple")].concat());
    assert_eq!(host_db(&crate::xplane::parse(&space).unwrap(), &space).1, (110, 190));
}
