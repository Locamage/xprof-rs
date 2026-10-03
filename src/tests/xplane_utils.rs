use super::xspace::XSpace;
use crate::derive::is_grouped;

fn plane_with_events(space: &mut XSpace, name: &str, events: &[(i64, &str)]) {
    let plane = space.add_plane();
    plane.name = name.into();
    for &(line, stat) in events {
        plane.line(line).timestamp_ns = 100;
        plane.event(line, &format!("event{line}"), 1, 2, &[(stat, 2.0.into())]);
    }
}

#[test]
fn partially_grouped_x_space() {
    let mut space = XSpace::default();
    plane_with_events(&mut space, "/host:CPU", &[(1, "non_group_id")]);
    plane_with_events(&mut space, "/device:TPU", &[(1, "group_id")]);
    assert!(!is_grouped(&space.parsed().1));
}

#[test]
fn test_fully_grouped_x_space() {
    let mut space = XSpace::default();
    plane_with_events(&mut space, "/host:CPU", &[(1, "group_id")]);
    plane_with_events(&mut space, "/device:TPU", &[(1, "group_id"), (2, "non_group_id")]);
    assert!(is_grouped(&space.parsed().1));
}
