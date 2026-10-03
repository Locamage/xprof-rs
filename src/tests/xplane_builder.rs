use super::xspace::{V, XSpace};
use crate::xplane::Value;

#[test]
fn non_instant_span_includes_single_time_tests() {
    let mut space = XSpace::default();
    let stats = [
        ("bool stat", V::Int(1)),
        ("int32 stat", 1234.into()),
        ("int64 stat", (1234i64 << 32).into()),
        ("uint32 stat", 5678u64.into()),
        ("uint64 stat", (5678u64 << 32).into()),
        ("string stat", "abc".into()),
        ("float stat", 0.5.into()),
        ("double stat", 1.0.into()),
        ("ref stat", V::Ref("referenced abc".into())),
    ];
    space.add_plane().event(0, "1st event", 0, 0, &stats);
    let (map, planes) = space.parsed();
    let (plane, event) = (&planes[0], &planes[0].lines[0].events[0]);
    assert_eq!(planes[0].lines.len(), 1);
    assert_eq!(&*plane.meta[event.meta as usize].name, "1st event");
    let stat = |name: &str| plane.stat(&map, event.meta, event.raw, name).unwrap();
    assert_eq!(stat("bool stat").int().map(|value| value != 0), Some(true));
    assert_eq!(stat("int32 stat").int(), Some(1234));
    assert_eq!(stat("int64 stat").int(), Some(1234 << 32));
    assert!(matches!(stat("uint32 stat"), Value::Uint(5678)));
    assert!(matches!(stat("uint64 stat"), Value::Uint(value) if value == 5678 << 32));
    assert_eq!(plane.text(&stat("string stat")), "abc");
    assert!(matches!(stat("float stat"), Value::Double(0.5)));
    assert!(matches!(stat("double stat"), Value::Double(1.0)));
    assert_eq!(plane.text(&stat("ref stat")), "referenced abc");
}
