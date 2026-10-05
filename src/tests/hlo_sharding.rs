use super::hlo_fixture::module;
use crate::hlo::text::{Printer, Style};
use crate::hlo::xla::mesh_proto::MeshAxis;
use crate::hlo::xla::named_sharding_proto::DimensionSharding;
use crate::hlo::xla::{AxisRefProto, HloComputationProto, HloInstructionProto, HloModuleProto, HloProto, MeshProto, NamedShardingProto, OpMetadata, OpSharding, ProgramShapeProto, ShapeProto};

const REPLICATED: i32 = 0;
const MAXIMAL: i32 = 1;
const TUPLE: i32 = 2;
const OTHER: i32 = 3;
const F32: i32 = 11;
const TUPLE_TYPE: i32 = 13;

fn to_string(sharding: OpSharding, include_metadata: bool) -> String {
    let shape = match sharding.r#type {
        TUPLE => ShapeProto { element_type: TUPLE_TYPE, tuple_shapes: vec![ShapeProto { element_type: F32, ..Default::default() }; sharding.tuple_shardings.len()], ..Default::default() },
        _ => ShapeProto { element_type: F32, ..Default::default() },
    };
    let parameter = HloInstructionProto { name: "p".into(), opcode: "parameter".into(), id: 1, shape: Some(Box::new(shape)), sharding: Some(Box::new(sharding)), ..Default::default() };
    let computation = HloComputationProto { name: "main".into(), id: 1, root_id: 1, instructions: vec![parameter], ..Default::default() };
    let hlo_module = HloModuleProto { name: "m".into(), entry_computation_id: 1, computations: vec![computation], host_program_shape: Some(ProgramShapeProto::default()), ..Default::default() };
    let module = module(&HloProto { hlo_module: Some(hlo_module), ..Default::default() });
    let mut text = String::new();
    Printer::new(&module, Style::Long, include_metadata).instruction(0, &mut text);
    text[text.find("sharding=").unwrap() + "sharding=".len()..].to_string()
}

fn metadata(op_name: &str) -> OpMetadata {
    OpMetadata { op_name: op_name.into(), ..Default::default() }
}

fn single_metadata() -> Vec<OpMetadata> {
    vec![metadata("a")]
}

fn list_metadata() -> Vec<OpMetadata> {
    vec![metadata("b"), metadata("c")]
}

fn replicate(metadata: Vec<OpMetadata>) -> OpSharding {
    OpSharding { r#type: REPLICATED, metadata, ..Default::default() }
}

fn single_device(device: i64, metadata: Vec<OpMetadata>) -> OpSharding {
    OpSharding { r#type: MAXIMAL, tile_assignment_dimensions: vec![1], tile_assignment_devices: vec![device], metadata, ..Default::default() }
}

fn tile(dimensions: &[i64], devices: &[i64], metadata: Vec<OpMetadata>) -> OpSharding {
    OpSharding { r#type: OTHER, tile_assignment_dimensions: dimensions.to_vec(), tile_assignment_devices: devices.to_vec(), metadata, ..Default::default() }
}

#[test]
fn to_string_replicated_test() {
    assert_eq!(to_string(replicate(Vec::new()), false), "{replicated}");
}

#[test]
fn to_string_replicate_sharding_with_metadata_test() {
    for (metadata, expected) in
        [(Vec::new(), "{replicated}"), (single_metadata(), "{replicated metadata={op_name=\"a\"}}"), (list_metadata(), "{replicated metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")]
    {
        assert_eq!(to_string(replicate(metadata.clone()), false), "{replicated}");
        assert_eq!(to_string(replicate(metadata), true), expected);
    }
}

#[test]
fn to_string_assign_device_test() {
    assert_eq!(to_string(single_device(7, Vec::new()), false), "{maximal device=7}");
}

#[test]
fn to_string_assign_device_sharding_with_metadata_test() {
    for (metadata, expected) in
        [(Vec::new(), "{maximal device=7}"), (single_metadata(), "{maximal device=7 metadata={op_name=\"a\"}}"), (list_metadata(), "{maximal device=7 metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")]
    {
        assert_eq!(to_string(single_device(7, metadata.clone()), false), "{maximal device=7}");
        assert_eq!(to_string(single_device(7, metadata), true), expected);
    }
}

#[test]
fn to_string_tiled_test() {
    assert_eq!(to_string(tile(&[2, 1, 2], &[2, 3, 5, 7], Vec::new()), false), "{devices=[2,1,2]2,3,5,7}");
}

#[test]
fn to_string_iota_tiled_test() {
    let sharding = OpSharding { r#type: OTHER, tile_assignment_dimensions: vec![3, 4], iota_reshape_dims: vec![2, 2, 3], iota_transpose_perm: vec![2, 1, 0], ..Default::default() };
    assert_eq!(to_string(sharding, false), "{devices=[3,4]<=[2,2,3]T(2,1,0)}");
}

#[test]
fn to_string_tiled_sharding_with_metadata_test() {
    for (metadata, expected) in [
        (Vec::new(), "{devices=[2,1,2]2,3,5,7}"),
        (single_metadata(), "{devices=[2,1,2]2,3,5,7 metadata={op_name=\"a\"}}"),
        (list_metadata(), "{devices=[2,1,2]2,3,5,7 metadata={{op_name=\"b\"}, {op_name=\"c\"}}}"),
    ] {
        assert_eq!(to_string(tile(&[2, 1, 2], &[2, 3, 5, 7], metadata.clone()), false), "{devices=[2,1,2]2,3,5,7}");
        assert_eq!(to_string(tile(&[2, 1, 2], &[2, 3, 5, 7], metadata), true), expected);
    }
}

#[test]
fn to_string_tuple_test() {
    let sharding = OpSharding { r#type: TUPLE, tuple_shardings: vec![replicate(Vec::new()), tile(&[1, 2], &[3, 5], Vec::new()), single_device(3, Vec::new())], ..Default::default() };
    assert_eq!(to_string(sharding, false), "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
}

#[test]
fn to_string_tuple_with_metadata_test() {
    let sharding = OpSharding { r#type: TUPLE, tuple_shardings: vec![replicate(vec![metadata("d")]), tile(&[1, 2], &[3, 5], Vec::new()), single_device(3, vec![metadata("e")])], ..Default::default() };
    assert_eq!(to_string(sharding.clone(), false), "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
    assert_eq!(to_string(sharding, true), "{{replicated metadata={op_name=\"d\"}}, {devices=[1,2]3,5}, {maximal device=3 metadata={op_name=\"e\"}}}");
}

#[test]
fn ostream_test() {
    assert_eq!(to_string(tile(&[1, 1, 2, 2], &[0, 1, 2, 3], Vec::new()), false), "{devices=[1,1,2,2]0,1,2,3}");
}

fn from_axis_names(metadata: Vec<OpMetadata>) -> OpSharding {
    let mesh = MeshProto { axes: vec![MeshAxis { name: "a".into(), size: 2 }, MeshAxis { name: "b".into(), size: 4 }], ..Default::default() };
    let dimension = |index: i64| DimensionSharding { axes: vec![AxisRefProto { mesh_axis_index: index, sub_axis_info: None }], is_closed: true };
    let named = NamedShardingProto { mesh: Some(mesh), dim_shardings: vec![dimension(0), dimension(1)], metadata, ..Default::default() };
    OpSharding { r#type: OTHER, named_sharding: Some(named), ..Default::default() }
}

#[test]
fn to_string_with_named_sharding_test() {
    assert_eq!(to_string(from_axis_names(Vec::new()), false), "{mesh['a'=2,'b'=4], [{'a'}, {'b'}]}");
    assert_eq!(to_string(from_axis_names(list_metadata()), true), "{mesh['a'=2,'b'=4], [{'a'}, {'b'}], metadata={{op_name=\"b\"}, {op_name=\"c\"}}}");
    let tuple_sharding = OpSharding { r#type: TUPLE, tuple_shardings: vec![from_axis_names(Vec::new()), from_axis_names(Vec::new()), from_axis_names(list_metadata())], ..Default::default() };
    assert_eq!(
        to_string(tuple_sharding, true),
        "{{mesh['a'=2,'b'=4], [{'a'}, {'b'}]}, {mesh['a'=2,'b'=4], [{'a'}, {'b'}]}, {mesh['a'=2,'b'=4], [{'a'}, {'b'}], metadata={{op_name=\"b\"}, {op_name=\"c\"}}}}"
    );
}
