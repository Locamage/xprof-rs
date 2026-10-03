use super::duty_cycle_tracker::{active_and_idle, add_tracker};
use super::xspace::XSpace;

#[test]
fn combine_multi_core_chip_test() {
    let mut space = XSpace::default();
    add_tracker(&mut space, Some(0), &[(10, 20, true), (20, 30, false)]);
    add_tracker(&mut space, Some(0), &[(10, 20, false), (20, 30, true)]);
    assert_eq!(active_and_idle(space), (20, 0));
}

#[test]
fn combine_multi_chip_test() {
    let mut space = XSpace::default();
    add_tracker(&mut space, None, &[(10, 20, true), (20, 30, false)]);
    add_tracker(&mut space, None, &[(10, 20, true), (20, 30, false)]);
    assert_eq!(active_and_idle(space), (20, 20));
}

#[test]
fn combine_multi_chip_and_core_test() {
    let mut space = XSpace::default();
    add_tracker(&mut space, Some(0), &[(10, 20, false), (20, 30, true)]);
    add_tracker(&mut space, Some(0), &[(10, 20, true), (20, 30, false)]);
    add_tracker(&mut space, None, &[(15, 25, true), (10, 30, false)]);
    assert_eq!(active_and_idle(space), (30, 10));
}
