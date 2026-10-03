use super::xspace::XSpace;

#[test]
fn process_session_test() {
    let mut xspace = XSpace::default();
    xspace.plane("/host:CPU");
    let (map, planes) = xspace.parsed();
    assert!(!crate::memory_profile::json(&planes, &map).is_empty());
}
