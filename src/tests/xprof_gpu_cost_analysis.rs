use crate::gpu_cost::tests::{analyze, computation, inst, parameter, shape};
use crate::hlo::xla::{HloInstructionProto, ShapeProto};

const S8: i32 = 2;
const S32: i32 = 4;
const F16: i32 = 10;
const F32: i32 = 11;
const TUPLE: i32 = 13;
const BF16: i32 = 16;
const F8E4M3FN: i32 = 20;
const GEMM_BACKEND_CONFIG: &str = r#"{
        "gemm_backend_config": {
            "alpha_real":1,
            "beta":0,
            "dot_dimension_numbers":{
                "lhs_contracting_dimensions":["1"],
                "rhs_contracting_dimensions":["0"],
                "lhs_batch_dimensions":[],
                "rhs_batch_dimensions":[]
            },
            "alpha_imag":0,
            "precision_config":{
                "operand_precision":["DEFAULT","DEFAULT"]
            },
            "epilogue":"DEFAULT"
        }
    }"#;

fn tuple(elements: Vec<ShapeProto>) -> ShapeProto {
    ShapeProto { element_type: TUPLE, tuple_shapes: elements, ..Default::default() }
}

fn gemm(target: &str, output: ShapeProto, operands: &[i64], backend_config: &str) -> Vec<HloInstructionProto> {
    let result = output.tuple_shapes[0].clone();
    let gemm = HloInstructionProto { name: "gemm".into(), custom_call_target: target.into(), backend_config: backend_config.as_bytes().to_vec(), ..inst(10, "custom-call", output, operands) };
    vec![gemm, HloInstructionProto { tuple_index: 0, ..inst(11, "get-tuple-element", result, &[10]) }]
}

fn flop_count_and_device_flops_adjustment(instructions: Vec<HloInstructionProto>) -> (i64, i64) {
    let costs = analyze(vec![computation(1, instructions)]).unwrap();
    let gemm = costs.len() - 2;
    (costs[gemm].model_flops, costs[gemm].model_flops - costs[gemm].device_flops)
}

#[test]
fn fp16_gemm_no_adjustment() {
    let mut instructions = vec![parameter(1, 0, shape(F16, &[65536, 32800])), parameter(2, 1, shape(F16, &[32800, 32]))];
    instructions.extend(gemm("__cublas$lt$matmul", tuple(vec![shape(F16, &[65536, 32]), shape(S8, &[0])]), &[1, 2], GEMM_BACKEND_CONFIG));
    let gold_flops = 65536i64 * 32800 * 32 * 2;
    assert_eq!(flop_count_and_device_flops_adjustment(instructions), (gold_flops, 0));
}

#[test]
fn s8_gemm_adjustment() {
    let mut instructions = vec![parameter(1, 0, shape(S8, &[65536, 32800])), parameter(2, 1, shape(S8, &[32800, 32]))];
    instructions.extend(gemm("__cublas$lt$matmul", tuple(vec![shape(S32, &[65536, 32]), shape(S8, &[0])]), &[1, 2], GEMM_BACKEND_CONFIG));
    let gold_flops = 65536i64 * 32800 * 32 * 2;
    assert_eq!(flop_count_and_device_flops_adjustment(instructions), (gold_flops, gold_flops / 2));
}

#[test]
fn fp8_gemm_with_fp32_parameter_adjustment() {
    let config =
        GEMM_BACKEND_CONFIG.replace(r#""epilogue":"DEFAULT""#, r#""epilogue":"DEFAULT", "lhs_stride":"10485760", "rhs_stride":"26214400", "grad_x":false, "grad_y":false, "damax_output":false"#);
    let mut rhs = shape(F8E4M3FN, &[5120, 5120]);
    rhs.layout.as_mut().unwrap().minor_to_major = vec![0, 1];
    let mut instructions = vec![parameter(1, 0, shape(F8E4M3FN, &[2048, 5120])), parameter(2, 1, rhs), parameter(3, 2, shape(F32, &[])), parameter(4, 3, shape(F32, &[]))];
    instructions.extend(gemm("__cublas$lt$matmul$f8", tuple(vec![shape(BF16, &[2048, 5120]), shape(S8, &[33554432])]), &[1, 2, 3, 4], &config));
    let gold_flops = 2048i64 * 5120 * 5120 * 2;
    assert_eq!(flop_count_and_device_flops_adjustment(instructions), (gold_flops, gold_flops / 2));
}

#[test]
fn custom_call_with_zero_operands() {
    let custom = HloInstructionProto { custom_call_target: "AllocateBuffer".into(), ..inst(1, "custom-call", shape(F32, &[2, 32, 256]), &[]) };
    assert!(analyze(vec![computation(1, vec![custom])]).is_some());
}
