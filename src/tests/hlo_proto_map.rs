use super::hlo_fixture::session_dir;
use super::xspace::{V, XSpace};
use crate::hlo::xla::{HloModuleProto, HloProto};
use crate::hlo::{by_options, extract, load, modules, protos};
use prost::Message;
use std::collections::HashMap;
use std::path::PathBuf;

fn add_original_hlo_proto(space: &mut XSpace, name: &str) {
    let hlo_proto = HloProto { hlo_module: Some(HloModuleProto { name: name.into(), ..Default::default() }), ..Default::default() };
    space.plane("/host:metadata").metadata_stats(name, &[("Hlo Proto", V::Bytes(hlo_proto.encode_to_vec()))]);
}

fn extracted(space: &XSpace) -> PathBuf {
    let dir = session_dir("hlo-proto-map");
    let xspace = dir.join("host.xplane.pb");
    std::fs::write(&xspace, space.encode_to_vec()).unwrap();
    extract(&dir, &[xspace]).unwrap();
    dir
}

#[test]
fn get_original_module_list() {
    let mut space = XSpace::default();
    let (map, planes) = space.parsed();
    assert!(protos(&planes, &map).is_empty());
    add_original_hlo_proto(&mut space, "module1");
    add_original_hlo_proto(&mut space, "module2");
    let mut module_list = modules(&extracted(&space));
    module_list.sort();
    assert_eq!(module_list, ["module1(1)", "module2(2)"]);
}

#[test]
fn get_original_hlo_proto() {
    let mut space = XSpace::default();
    add_original_hlo_proto(&mut space, "module");
    let dir = extracted(&space);
    let by_program_id = |program_id: &str| by_options(&dir, &HashMap::from([("program_id".to_string(), program_id.to_string())]));
    assert_eq!(by_program_id("1").unwrap().proto.name, "module");
    assert!(by_program_id("2").is_none());
    assert_eq!(load(&dir, "module(1)").unwrap().proto.name, "module");
    assert!(load(&dir, "module(2)").is_none());
    assert!(load(&dir, "module2(1)").is_none());
}
