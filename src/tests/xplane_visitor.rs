use super::xspace::{V, XSpace};
use crate::xplane::Value;

#[test]
fn get_stat_test() {
    let mut space = XSpace::default();
    space.add_plane().event(0, "test_event", 0, 0, &[("test_stat", 42.into())]);
    let (map, planes) = space.parsed();
    let (plane, event) = (&planes[0], &planes[0].lines[0].events[0]);
    assert_eq!(plane.stat(&map, event.meta, event.raw, "test_stat").and_then(|value| value.int()), Some(42));
    assert!(plane.stat(&map, event.meta, event.raw, "missing_stat").is_none());
}

#[test]
fn get_event_or_metadata_stat_test() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.metadata_stats(
        "test_event",
        &[
            ("test_stat", 10.into()),
            ("meta_only_stat", 30.into()),
            ("string_stat", "hello".into()),
            ("double_stat", 1.5.into()),
            ("uint64_bool_stat", 1u64.into()),
            ("bytes_stat", V::Bytes(b"test_bytes".to_vec())),
        ],
    );
    plane.event(0, "test_event", 0, 0, &[("test_stat", 20.into())]);
    let (map, planes) = space.parsed();
    let (plane, event) = (&planes[0], &planes[0].lines[0].events[0]);
    let stat = |name: &str| plane.stat(&map, event.meta, event.raw, name);
    let int = |name: &str| stat(name).and_then(|value| value.int());
    let double = |name: &str| match stat(name) {
        Some(Value::Double(value)) => Some(value),
        _ => None,
    };
    let text = |name: &str| match stat(name) {
        Some(Value::Str(bytes) | Value::Bytes(bytes)) => Some(String::from_utf8_lossy(bytes).into_owned()),
        _ => None,
    };
    assert_eq!((int("test_stat"), int("meta_only_stat")), (Some(20), Some(30)));
    assert_eq!(text("string_stat").as_deref(), Some("hello"));
    assert_eq!((double("double_stat"), double("test_stat")), (Some(1.5), None));
    assert_eq!(text("bytes_stat").as_deref(), Some("test_bytes"));
    assert_eq!(int("uint64_bool_stat").map(|value| value != 0), Some(true));
    assert!(stat("absent_stat").is_none());
    assert_eq!((int("string_stat"), text("test_stat")), (None, None));
}
