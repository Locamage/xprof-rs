use crate::hlo::Module;
use crate::hlo_text::{Printer, Style};
use std::borrow::Cow;
use std::sync::LazyLock;

static LITERALS: LazyLock<Module<'static>> = LazyLock::new(|| Module::parse(Cow::Owned(std::fs::read(format!("{}/tests/data/literal/literals.pb", env!("CARGO_MANIFEST_DIR"))).unwrap())));

fn to_string(name: &str) -> String {
    let module = &*LITERALS;
    let literal = module.inst(module.find(name).unwrap()).literal.unwrap();
    let mut text = format!("{} ", literal.shape.as_ref().unwrap().text(false));
    Printer::new(module, Style::Long, false).literal(&literal, false, &mut text);
    text
}

#[test]
fn literal_scalar_to_string() {
    let expected = [
        ("true_lit", "pred[] true"),
        ("false_lit", "pred[] false"),
        ("u1_lit", "u1[] 1"),
        ("u2_lit", "u2[] 0"),
        ("u4_lit", "u4[] 5"),
        ("u32_lit", "u32[] 42"),
        ("s1_lit", "s1[] -1"),
        ("s2_lit", "s2[] 1"),
        ("s4_lit", "s4[] -3"),
        ("s32_lit", "s32[] -999"),
        ("f32_lit", "f32[] 3.14"),
        ("f16_lit", "f16[] 0.5"),
        ("c64_lit", "c64[] (3.14, 2.78)"),
        ("c128_lit", "c128[] (3.14, 2.78)"),
        ("bf16_lit", "bf16[] 0.5"),
        ("bf16_lit_truncated", "bf16[] 3.141"),
        ("bf16_lit_truncated2", "bf16[] 9"),
        ("f4e2m1fn_lit", "f4e2m1fn[] 0.5"),
        ("f8e5m2_lit", "f8e5m2[] 0.5"),
        ("f8e5m2_lit_truncated", "f8e5m2[] 3"),
        ("f8e4m3_lit", "f8e4m3[] 0.5"),
        ("f8e4m3fn_lit", "f8e4m3fn[] 0.5"),
        ("f8e4m3b11fnuz_lit", "f8e4m3b11fnuz[] 0.5"),
        ("f8e4m3fnuz_lit", "f8e4m3fnuz[] 0.5"),
        ("f8e5m2fnuz_lit", "f8e5m2fnuz[] 0.5"),
        ("f8e3m4_lit", "f8e3m4[] 0.5"),
        ("f8e8m0fnu_lit", "f8e8m0fnu[] 0.5"),
    ];
    for (name, text) in expected {
        assert_eq!(to_string(name), text);
    }
}

#[test]
fn literal_vector_to_string() {
    assert_eq!(to_string("pred_vec"), "pred[3] {1, 0, 1}");
}

#[test]
fn r2_to_string() {
    assert_eq!(to_string("r2"), "s32[3,2] {\n  { 1, 2 },\n  { 3, 4 },\n  { 5, 6 }\n}");
}

#[test]
fn r3_to_string() {
    assert_eq!(to_string("r3"), "s32[3,2,1] {\n{\n  {1},\n  {2}\n},\n{\n  {3},\n  {4}\n},\n{\n  {5},\n  {6}\n}\n}");
}

#[test]
fn r6_to_string() {
    let expected = "s32[2,2,1,1,1,2] {\n{ /*i0=0*/\n{ /*i1=0*/\n{ /*i2=0*/\n{ /*i3=0*/\n  { 0, 0 }\n}\n}\n},\n{ /*i1=1*/\n{ /*i2=0*/\n{ /*i3=0*/\n  { 0, 0 }\n}\n}\n}\n},\n{ /*i0=1*/\n{ /*i1=0*/\n{ /*i2=0*/\n{ /*i3=0*/\n  { 0, 0 }\n}\n}\n},\n{ /*i1=1*/\n{ /*i2=0*/\n{ /*i3=0*/\n  { 0, 0 }\n}\n}\n}\n}\n}";
    assert_eq!(to_string("r6"), expected);
}

#[test]
fn create_r3_from_array3d() {
    assert_eq!(LITERALS.nodes[LITERALS.find("r3_from_array").unwrap()].shape.dimensions, [2, 3, 2]);
    assert_eq!(to_string("r3_from_array"), "f32[2,3,2] {\n{\n  { 1, 2 },\n  { 3, 4 },\n  { 5, 6 }\n},\n{\n  { 7, 8 },\n  { 9, 10 },\n  { 11, 12 }\n}\n}");
}

#[test]
fn literal_r4_f32_projected_stringifies() {
    assert_eq!(LITERALS.nodes[LITERALS.find("r4_projected").unwrap()].shape.dimensions, [1, 2, 3, 2]);
    let expected = "f32[1,2,3,2] {\n{ /*i0=0*/\n{ /*i1=0*/\n  { 1, 2 },\n  { 1001, 1002 },\n  { 2001, 2002 }\n},\n{ /*i1=1*/\n  { 1, 2 },\n  { 1001, 1002 },\n  { 2001, 2002 }\n}\n}\n}";
    assert_eq!(to_string("r4_projected"), expected);
}

#[test]
fn literal_r4_f32_stringifies() {
    assert_eq!(LITERALS.nodes[LITERALS.find("r4").unwrap()].shape.dimensions, [2, 2, 3, 3]);
    let expected = "f32[2,2,3,3] {\n{ /*i0=0*/\n{ /*i1=0*/\n  { 1, 2, 3 },\n  { 4, 5, 6 },\n  { 7, 8, 9 }\n},\n{ /*i1=1*/\n  { 11, 12, 13 },\n  { 14, 15, 16 },\n  { 17, 18, 19 }\n}\n},\n{ /*i0=1*/\n{ /*i1=0*/\n  { 101, 102, 103 },\n  { 104, 105, 106 },\n  { 107, 108, 109 }\n},\n{ /*i1=1*/\n  { 201, 202, 203 },\n  { 204, 205, 206 },\n  { 207, 208, 209 }\n}\n}\n}";
    assert_eq!(to_string("r4"), expected);
}
