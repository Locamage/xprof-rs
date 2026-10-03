use super::cli_support::{Fake, args, scratch, text};
use super::xspace::XSpace;
use crate::cli::Kind;
use crate::cli::hlo::{get_hlo_module_content, get_hlo_neighborhood, get_hlo_text, list_hlo_modules, proto_files};
use crate::cli::json::J;
use prost::Message;

const MODULE: &str = "module_0001.jit_compute.hlo_proto.pb";
const GRAPH: &str = "%entry (\n  %x = f32[10] parameter(0)\n  %w = f32[10] parameter(1)\n  %mul = f32[10] multiply(%x, %w)\n  %add = f32[10] add(%mul, %x)\n  ROOT %neg = f32[10] negate(%add)\n)\n";

fn session(name: &str, modules: &[&str]) -> std::path::PathBuf {
    let dir = scratch(name);
    for module in modules {
        std::fs::write(dir.join(module), b"dummy").unwrap();
    }
    dir
}

#[test]
fn test_list_hlo_modules_empty() {
    let dir = session("hlo-empty", &[]);
    std::fs::write(dir.join("host.xplane.pb"), XSpace::default().encode_to_vec()).unwrap();
    let result = text(list_hlo_modules(&Fake::fixed("").in_dir(&dir), &args("empty_session", &[])));
    assert!(result.contains("No HLO modules found"), "{result}");
}

#[test]
fn test_list_hlo_modules_success() {
    let dir = session("hlo-list", &[MODULE, "module_0002.jit_eval.hlo_proto.pb"]);
    let result = text(list_hlo_modules(&Fake::fixed("").in_dir(&dir), &args(&dir.to_string_lossy(), &[])));
    assert!(result.contains("Found 2 HLO modules:"));
    assert!(result.contains("0. module_0001.jit_compute"));
    assert!(result.contains("1. module_0002.jit_eval"));
}

#[test]
fn test_get_hlo_module_content_success() {
    let dir = session("hlo-content", &[MODULE]);
    let fake = Fake::fixed("HloModule jit_compute\n\n%entry (\n  %x = f32[10] parameter(0)\n  ROOT %neg = f32[10] negate(%x)\n)\n").in_dir(&dir);
    let content = text(get_hlo_module_content(&fake, &args(&dir.to_string_lossy(), &[("module_name", J::from("module_0001.jit_compute"))])));
    assert!(content.contains("HloModule jit_compute"));
    assert!(content.contains("%neg = f32[10] negate(%x)"));
}

#[test]
fn test_get_hlo_neighborhood_bfs() {
    let dir = session("hlo-bfs", &[MODULE]);
    let neighborhood = text(get_hlo_neighborhood(&Fake::fixed(GRAPH).in_dir(&dir), &args(&dir.to_string_lossy(), &[("instruction_name", J::from("mul")), ("radius", J::Int(1))])));
    assert!(neighborhood.contains("%mul"));
    assert!(neighborhood.contains("%x"));
    assert!(neighborhood.contains("%w"));
}

#[test]
fn test_get_hlo_neighborhood_op_name_alias() {
    let dir = session("hlo-alias", &[MODULE]);
    let neighborhood = text(get_hlo_neighborhood(&Fake::fixed(GRAPH).in_dir(&dir), &args(&dir.to_string_lossy(), &[("op_name", J::from("mul")), ("radius", J::Int(1))])));
    assert!(neighborhood.contains("%mul"));
    assert!(neighborhood.contains("%x"));
}

#[test]
fn test_get_hlo_neighborhood_missing_name_raises_value_error() {
    let error = get_hlo_neighborhood(&Fake::fixed(GRAPH), &args("session", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}

#[test]
fn test_get_hlo_neighborhood_conflicting_names_raises_value_error() {
    let error = get_hlo_neighborhood(&Fake::fixed(GRAPH), &args("session", &[("instruction_name", J::from("mul")), ("op_name", J::from("add"))])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
    assert!(error.message.contains("Conflicting arguments"));
}

#[test]
fn test_get_hlo_text_file_export() {
    let dir = session("hlo-export", &[MODULE]);
    let out = dir.join("exported_hlo.txt");
    let content = text(get_hlo_text(&Fake::fixed("HloModule test_export\n").in_dir(&dir), &args(&dir.to_string_lossy(), &[("path", J::from(out.to_string_lossy().as_ref()))])));
    assert_eq!(content, "HloModule test_export\n");
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "HloModule test_export\n");
}

#[test]
fn test_get_hlo_proto_files_finds_nested_hlo_protos() {
    let dir = session("hlo-nested", &[]);
    let nested = dir.join("plugins/profile/2026_08_18_01_02_03");
    std::fs::create_dir_all(&nested).unwrap();
    let module = nested.join("module_nested.hlo_proto.pb");
    std::fs::write(&module, b"dummy").unwrap();
    assert_eq!(proto_files(&Fake::fixed("").in_dir(&dir), &dir.to_string_lossy()).unwrap(), vec![module]);
}
