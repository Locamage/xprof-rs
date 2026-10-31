use super::xspace::{V, XSpace};
use crate::hlo::protos;

const METADATA: &str = "/host:metadata";
const EVENT_NAME: &str = "train_step (12345)";
const HLO: &[u8] = b"\x0a\x03\x0a\x01m";

fn hlo_protos(space: &XSpace) -> Vec<(u64, Vec<u8>)> {
    let (map, planes) = space.parsed();
    protos(&planes, &map)
        .into_iter()
        .map(|(id, body)| (id, body.to_vec()))
        .collect()
}

#[test]
fn fixes_hlo_metadata_success() {
    let mut space = XSpace::default();
    let metadata_plane = space.plane(METADATA);
    metadata_plane.metadata_stats(EVENT_NAME, &[("HLO Proto", V::Bytes(HLO.to_vec()))]);
    assert_ne!(metadata_plane.event_metadata(EVENT_NAME).id, 12345);
    assert_eq!(hlo_protos(&space), [(12345, HLO.to_vec())]);
}

#[test]
fn does_not_fix_if_no_legacy_hlo_stat() {
    let mut space = XSpace::default();
    space.plane(METADATA).event_metadata(EVENT_NAME);
    assert!(hlo_protos(&space).is_empty());
}

#[test]
fn fixes_even_if_hlo_stat_already_has_expected_name() {
    let mut space = XSpace::default();
    let metadata_plane = space.plane(METADATA);
    metadata_plane.metadata_stats(EVENT_NAME, &[("Hlo Proto", V::Bytes(HLO.to_vec()))]);
    let original_event_id = metadata_plane.event_metadata(EVENT_NAME).id as u64;
    let fixed = hlo_protos(&space);
    assert!(fixed.iter().all(|(id, _)| *id != original_event_id));
    assert_eq!(fixed, [(12345, HLO.to_vec())]);
}

#[test]
fn fixes_metadata_with_a_display_name() {
    let mut space = XSpace::default();
    let metadata_plane = space.plane(METADATA);
    metadata_plane.metadata_stats(EVENT_NAME, &[("Hlo Proto", V::Bytes(HLO.to_vec()))]);
    metadata_plane.event_metadata(EVENT_NAME).display_name = EVENT_NAME.into();
    assert_eq!(hlo_protos(&space), [(12345, HLO.to_vec())]);
}

#[test]
fn skips_if_no_hlo_stat_in_event() {
    let mut space = XSpace::default();
    space
        .plane(METADATA)
        .metadata_stats(EVENT_NAME, &[("Other Stat", V::Str("value".into()))]);
    assert!(hlo_protos(&space).is_empty());
}

#[test]
fn skips_a_proto_that_does_not_parse() {
    for body in [
        &b"hlo_content"[..],
        b"\x0a\x03\x0a\x01\xff",
        b"\x0a\x02\x1a\x05",
        b"\x0b\x0c\x0a\x01m",
    ] {
        let mut space = XSpace::default();
        space
            .plane(METADATA)
            .metadata_stats(EVENT_NAME, &[("Hlo Proto", V::Bytes(body.to_vec()))]);
        assert!(hlo_protos(&space).is_empty(), "{body:?}");
    }
}

#[test]
fn keeps_a_proto_with_unknown_fields_and_groups() {
    for body in [
        &b"\x0a\x03\x0a\x01m\xf8\x01\x07"[..],
        b"\x0a\x03\x0a\x01m\xfb\x01\x08\x01\xfc\x01",
        b"\x0a\x02\x10\x01",
    ] {
        let mut space = XSpace::default();
        space
            .plane(METADATA)
            .metadata_stats(EVENT_NAME, &[("Hlo Proto", V::Bytes(body.to_vec()))]);
        assert_eq!(hlo_protos(&space), [(12345, body.to_vec())], "{body:?}");
    }
}
