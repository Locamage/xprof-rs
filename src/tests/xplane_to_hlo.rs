use super::hlo_fixture::{session_dir, write_module};
use crate::graph_viewer::serve;
use crate::hlo::by_options;
use crate::hlo::xla::{HloComputationProto, HloInstructionProto, HloModuleProto, HloProto};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn set_up() -> PathBuf {
    let profile_dir = session_dir("xplane-to-hlo");
    std::fs::write(profile_dir.join("hostname0.xplane.pb"), b"").unwrap();
    profile_dir
}

fn named(name: &str) -> HloProto {
    HloProto { hlo_module: Some(HloModuleProto { name: name.into(), ..Default::default() }), ..Default::default() }
}

fn write_dummy_hlo_with_node(profile_dir: &Path, module_name: &str, node_name: &str) {
    let mut hlo = named(module_name);
    let instruction = HloInstructionProto { name: node_name.into(), ..Default::default() };
    hlo.hlo_module.as_mut().unwrap().computations.push(HloComputationProto { instructions: vec![instruction], ..Default::default() });
    write_module(profile_dir, module_name, &hlo);
}

fn options(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|&(key, value)| (key.to_string(), value.to_string())).collect()
}

fn by_node_name(profile_dir: &Path, node_name: &str) -> Result<(Vec<u8>, &'static str), String> {
    serve(profile_dir, &options(&[("type", "adj_nodes"), ("node_name", node_name)]))
}

#[test]
fn get_hlo_proto_by_node_name_success() {
    let profile_dir = set_up();
    write_module(&profile_dir, "empty_module", &HloProto::default());
    write_dummy_hlo_with_node(&profile_dir, "other_module", "other_node");
    write_dummy_hlo_with_node(&profile_dir, "my_module", "my_target_node");
    assert_eq!(by_node_name(&profile_dir, "my_target_node"), Err("No program shape found in the proto".to_string()));
}

#[test]
fn get_hlo_proto_by_node_name_not_found() {
    let profile_dir = set_up();
    std::fs::write(profile_dir.join("bad_module.hlo_proto.pb"), "this is not a valid proto").unwrap();
    write_dummy_hlo_with_node(&profile_dir, "my_module", "other_node");
    assert!(by_node_name(&profile_dir, "my_target_node").unwrap_err().contains("not found"));
}

#[test]
fn get_hlo_proto_by_node_name_ignores_other_files() {
    let profile_dir = set_up();
    std::fs::write(profile_dir.join("some_other_file.txt"), "not a proto").unwrap();
    std::fs::write(profile_dir.join(".hlo_proto.pb"), "not a proto").unwrap();
    assert!(by_node_name(&profile_dir, "my_target_node").is_err());
}

#[test]
fn get_hlo_proto_by_node_name_invalid_dir() {
    let invalid_profile_dir = crate::tests::temp_dir().join("xprof-rs-non_existent_dir");
    assert!(by_node_name(&invalid_profile_dir, "my_target_node").unwrap_err().contains("not found"));
}

#[test]
fn get_hlo_proto_by_program_id_success() {
    let profile_dir = set_up();
    write_module(&profile_dir, "my_module_prog_123", &named("my_module_prog_123"));
    let hlo_proto = by_options(&profile_dir, &options(&[("program_id", "prog_123")])).unwrap();
    assert_eq!(hlo_proto.proto.name, "my_module_prog_123");
}

#[test]
fn get_hlo_proto_by_program_id_not_found() {
    let profile_dir = set_up();
    assert!(by_options(&profile_dir, &options(&[("program_id", "prog_456")])).is_none());
    assert!(serve(&profile_dir, &options(&[("type", "pbtxt"), ("program_id", "prog_456")])).unwrap_err().contains("not found"));
}

#[test]
fn get_hlo_proto_by_options_success_module() {
    let profile_dir = set_up();
    write_module(&profile_dir, "my_module", &named("my_module"));
    let hlo_proto = by_options(&profile_dir, &options(&[("module_name", "my_module")])).unwrap();
    assert_eq!(hlo_proto.proto.name, "my_module");
}

#[test]
fn get_hlo_proto_by_options_success_program_id() {
    let profile_dir = set_up();
    write_module(&profile_dir, "my_module_prog_123", &named("my_module_prog_123"));
    let hlo_proto = by_options(&profile_dir, &options(&[("program_id", "prog_123")])).unwrap();
    assert_eq!(hlo_proto.proto.name, "my_module_prog_123");
}

#[test]
fn get_hlo_proto_by_options_failure() {
    let profile_dir = set_up();
    assert!(by_options(&profile_dir, &options(&[])).is_none());
    assert!(serve(&profile_dir, &options(&[("type", "pbtxt")])).unwrap_err().contains("Can not load"));
}
