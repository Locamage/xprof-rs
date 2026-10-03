use super::{Cost, costs};
use crate::hlo::Module;
use crate::hlo::xla::{
    DotDimensionNumbers, HloComputationProto, HloInstructionProto, HloModuleProto, HloProto, LayoutProto, ProgramShapeProto, ReplicaGroup, ShapeProto, hlo_instruction_proto::SliceDimensions,
};
use prost::Message;
use std::borrow::Cow;

const F32: i32 = 11;
const S8: i32 = 2;

pub fn shape(kind: i32, dimensions: &[i64]) -> ShapeProto {
    let layout = LayoutProto { minor_to_major: (0..dimensions.len() as i64).rev().collect(), ..Default::default() };
    ShapeProto { element_type: kind, dimensions: dimensions.to_vec(), layout: Some(Box::new(layout)), ..Default::default() }
}

pub fn inst(id: i64, opcode: &str, shape: ShapeProto, operands: &[i64]) -> HloInstructionProto {
    HloInstructionProto { id, name: format!("{opcode}.{id}"), opcode: opcode.into(), shape: Some(Box::new(shape)), operand_ids: operands.to_vec(), ..Default::default() }
}

pub fn parameter(id: i64, number: i64, shape: ShapeProto) -> HloInstructionProto {
    HloInstructionProto { parameter_number: number, ..inst(id, "parameter", shape, &[]) }
}

pub fn computation(id: i64, instructions: Vec<HloInstructionProto>) -> HloComputationProto {
    let root_id = instructions.last().unwrap().id;
    HloComputationProto { id, name: format!("computation.{id}"), root_id, instructions, ..Default::default() }
}

pub fn analyze(computations: Vec<HloComputationProto>) -> Option<Vec<Cost>> {
    let entry_computation_id = computations.last().unwrap().id;
    let proto = HloProto {
        hlo_module: Some(HloModuleProto { name: "test".into(), entry_computation_id, computations, host_program_shape: Some(ProgramShapeProto::default()), ..Default::default() }),
        ..Default::default()
    };
    costs(&Module::parse(Cow::Owned(proto.encode_to_vec())))
}

fn adder(id: i64) -> HloComputationProto {
    computation(id, vec![parameter(id * 10, 0, shape(F32, &[])), parameter(id * 10 + 1, 1, shape(F32, &[])), inst(id * 10 + 2, "add", shape(F32, &[]), &[id * 10, id * 10 + 1])])
}

#[test]
fn elementwise_flops_follow_the_sm86_profile() {
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(F32, &[4])), inst(2, "cosine", shape(F32, &[4]), &[1])])]).unwrap();
    assert_eq!(result[1], Cost { model_flops: 2648, device_flops: 2648, bytes_accessed: 32, memory: vec![(true, 1, 16), (false, 1, 16)] });
    assert_eq!(result[0], Cost::default());
}

#[test]
fn fusion_reads_only_the_sliced_part_of_its_parameter() {
    let slice = HloInstructionProto { slice_dimensions: vec![SliceDimensions { start: 0, limit: 2, stride: 1 }], ..inst(11, "slice", shape(F32, &[2]), &[10]) };
    let fused = computation(1, vec![parameter(10, 0, shape(F32, &[8])), slice, inst(12, "add", shape(F32, &[2]), &[11, 11])]);
    let fusion = HloInstructionProto { fusion_kind: "kLoop".into(), called_computation_ids: vec![1], ..inst(21, "fusion", shape(F32, &[2]), &[20]) };
    let result = analyze(vec![fused, computation(2, vec![parameter(20, 0, shape(F32, &[8])), fusion])]).unwrap();
    assert_eq!(result[4], Cost { model_flops: 6, device_flops: 6, bytes_accessed: 16, memory: vec![(true, 1, 8), (false, 1, 8)] });
}

#[test]
fn slice_counts_output_bytes_twice() {
    let slice = HloInstructionProto { slice_dimensions: vec![SliceDimensions { start: 2, limit: 4, stride: 1 }], ..inst(2, "slice", shape(F32, &[2]), &[1]) };
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(F32, &[8])), slice])]).unwrap();
    assert_eq!(result[1], Cost { model_flops: 0, device_flops: 0, bytes_accessed: 16, memory: vec![(true, 1, 8), (false, 1, 8)] });
}

#[test]
fn reduce_scales_the_reducer_and_reads_init_per_output() {
    let reduce = HloInstructionProto { dimensions: vec![1], called_computation_ids: vec![1], ..inst(23, "reduce", shape(F32, &[4]), &[21, 22]) };
    let result = analyze(vec![adder(1), computation(2, vec![parameter(21, 0, shape(F32, &[4, 8])), parameter(22, 1, shape(F32, &[])), reduce])]).unwrap();
    assert_eq!(result[5], Cost { model_flops: 84, device_flops: 84, bytes_accessed: 160, memory: vec![(true, 1, 144), (false, 1, 16)] });
}

#[test]
#[allow(deprecated)]
fn flattened_id_collective_without_groups_fails_the_module() {
    let all_reduce = |groups: Vec<ReplicaGroup>| HloInstructionProto {
        channel_id: 1,
        use_global_device_ids: true,
        called_computation_ids: vec![1],
        replica_groups: groups,
        ..inst(22, "all-reduce-start", shape(F32, &[4]), &[21])
    };
    assert!(analyze(vec![adder(1), computation(2, vec![parameter(21, 0, shape(F32, &[4])), all_reduce(Vec::new())])]).is_none());
    let result = analyze(vec![adder(1), computation(2, vec![parameter(21, 0, shape(F32, &[4])), all_reduce(vec![ReplicaGroup { replica_ids: vec![0, 1] }])])]).unwrap();
    assert_eq!(result[4], Cost { model_flops: 12, device_flops: 12, bytes_accessed: 16, memory: vec![(true, 1, 16), (false, 1, 16)] });
}

#[test]
fn eight_bit_inputs_halve_device_flops() {
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(S8, &[4])), parameter(2, 1, shape(S8, &[4])), inst(3, "multiply", shape(S8, &[4]), &[1, 2])])]).unwrap();
    assert_eq!((result[2].model_flops, result[2].device_flops), (12, 6));
}

#[test]
fn cublas_lt_gemm_uses_the_backend_dot_dimensions() {
    let config = br#"{"gemm_backend_config":{"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}}"#;
    let gemm = HloInstructionProto { custom_call_target: "__cublas$lt$matmul".into(), backend_config: config.to_vec(), ..inst(3, "custom-call", shape(F32, &[2, 3]), &[1, 2]) };
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(F32, &[2, 4])), parameter(2, 1, shape(F32, &[4, 3])), gemm])]).unwrap();
    assert_eq!(result[2].model_flops, 48);
    let dot = HloInstructionProto {
        dot_dimension_numbers: Some(Box::new(DotDimensionNumbers { lhs_contracting_dimensions: vec![1], rhs_contracting_dimensions: vec![0], ..Default::default() })),
        ..inst(3, "dot", shape(F32, &[2, 3]), &[1, 2])
    };
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(F32, &[2, 4])), parameter(2, 1, shape(F32, &[4, 3])), dot])]).unwrap();
    assert_eq!(result[2].model_flops, 48);
}

#[test]
fn legacy_cublas_gemm_has_no_cost() {
    let gemm = HloInstructionProto { custom_call_target: "__cublas$gemm".into(), ..inst(3, "custom-call", shape(F32, &[2, 3]), &[1, 2]) };
    let result = analyze(vec![computation(1, vec![parameter(1, 0, shape(F32, &[2, 4])), parameter(2, 1, shape(F32, &[4, 3])), gemm])]).unwrap();
    assert_eq!(result[2], Cost::default());
}

fn fixture(name: &str) -> (Module<'static>, Vec<Cost>) {
    let data = std::fs::read(format!("{}/tests/data/gpu_cost/{name}.pb", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let module = Module::parse(Cow::Owned(data));
    let analysis = costs(&module).unwrap();
    (module, analysis)
}

fn flops_and_adjustment(name: &str, instruction: &str) -> (i64, i64) {
    let (module, analysis) = fixture(name);
    let cost = &analysis[module.find(instruction).unwrap()];
    (cost.model_flops, cost.model_flops - cost.device_flops)
}

#[test]
fn fp16_gemm_no_adjustment() {
    let gold_flops = 65536 * 32800 * 32 * 2;
    assert_eq!(flops_and_adjustment("fp16_gemm_no_adjustment", "gemm"), (gold_flops, 0));
}

#[test]
fn s8_gemm_adjustment() {
    let gold_flops = 65536 * 32800 * 32 * 2;
    assert_eq!(flops_and_adjustment("s8_gemm_adjustment", "gemm"), (gold_flops, gold_flops / 2));
}

#[test]
fn fp8_gemm_with_fp32_parameter_adjustment() {
    let gold_flops = 2048 * 5120 * 5120 * 2;
    assert_eq!(flops_and_adjustment("fp8_gemm_with_fp32_parameter_adjustment", "gemm"), (gold_flops, gold_flops / 2));
}

#[test]
fn custom_call_with_zero_operands() {
    fixture("custom_call_with_zero_operands");
}

#[test]
fn call_registered_factory() {
    let (module, analysis) = fixture("call_registered_factory");
    let root = &analysis[module.graphs[module.entry].root];
    assert_eq!(root.device_flops, root.model_flops);
}

#[test]
fn test16_bit_pricision() {
    let (module, analysis) = fixture("test16_bit_pricision");
    let root = &analysis[module.graphs[module.entry].root];
    assert_eq!(root.device_flops, root.model_flops);
    assert!(root.device_flops > 0);
}

#[test]
fn test4_bit_pricision() {
    let (module, analysis) = fixture("test4_bit_pricision");
    let root = &analysis[module.graphs[module.entry].root];
    assert_eq!(root.device_flops, root.model_flops / 4);
    assert!(root.device_flops > 0);
}
