use super::xspace::{V, XSpace};
use crate::hlo::protos;

const METADATA: &str = "/host:metadata";
const EVENT_NAME: &str = "train_step (12345)";

fn hlo_protos(space: &XSpace) -> Vec<(u64, Vec<u8>)> {
    let (map, planes) = space.parsed();
    protos(&planes, &map).into_iter().map(|(id, body)| (id, body.to_vec())).collect()
}

#[test]
fn fixes_hlo_metadata_success() {
    let mut space = XSpace::default();
    let metadata_plane = space.plane(METADATA);
    metadata_plane.metadata_stats(EVENT_NAME, &[("HLO Proto", V::Bytes(b"hlo_content".to_vec()))]);
    assert_ne!(metadata_plane.event_metadata(EVENT_NAME).id, 12345);
    assert_eq!(hlo_protos(&space), [(12345, b"hlo_content".to_vec())]);
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
    metadata_plane.metadata_stats(EVENT_NAME, &[("Hlo Proto", V::Bytes(b"hlo_content".to_vec()))]);
    let original_event_id = metadata_plane.event_metadata(EVENT_NAME).id as u64;
    let fixed = hlo_protos(&space);
    assert!(fixed.iter().all(|(id, _)| *id != original_event_id));
    assert_eq!(fixed, [(12345, b"hlo_content".to_vec())]);
}

#[test]
fn skips_if_no_hlo_stat_in_event() {
    let mut space = XSpace::default();
    space.plane(METADATA).metadata_stats(EVENT_NAME, &[("Other Stat", V::Str("value".into()))]);
    assert!(hlo_protos(&space).is_empty());
}
