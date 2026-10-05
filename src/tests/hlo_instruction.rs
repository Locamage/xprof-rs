use super::hlo_fixture::{hlo_proto, module};
use crate::hlo::Module;
use crate::hlo::text::{Printer, Style, block_scaling_config_text};
use crate::hlo::xla::BlockScalingConfig;
use crate::hlo::xla::block_scaling_config::TensorBlockScalingConfig;
use std::borrow::Cow;

const CONVOLUTION: &str = r#"
hlo_module {
  name: "SparsityConfig"
  entry_computation_name: "main"
  entry_computation_id: 1
  host_program_shape { result { element_type: BF16 dimensions: [256, 256] layout { minor_to_major: [1, 0] } } }
  computations {
    name: "main"
    id: 1
    root_id: 5
    instructions { name: "lhs" opcode: "parameter" id: 1 parameter_number: 0 shape { element_type: BF16 dimensions: [256, 256] layout { minor_to_major: [1, 0] } } }
    instructions { name: "rhs" opcode: "parameter" id: 2 parameter_number: 1 shape { element_type: BF16 dimensions: [64, 256] layout { minor_to_major: [1, 0] } } }
    instructions { name: "lhs_indices" opcode: "parameter" id: 3 parameter_number: 2 shape { element_type: S32 dimensions: [256, 256] layout { minor_to_major: [1, 0] } } }
    instructions { name: "rhs_indices" opcode: "parameter" id: 4 parameter_number: 3 shape { element_type: S32 dimensions: [64, 256] layout { minor_to_major: [1, 0] } } }
    instructions {
      name: "convolution"
      opcode: "convolution"
      id: 5
      operand_ids: %s
      shape { element_type: BF16 dimensions: [256, 256] layout { minor_to_major: [1, 0] } }
      window {}
      convolution_dimension_numbers { input_batch_dimension: 0 input_feature_dimension: 1 kernel_input_feature_dimension: 0 kernel_output_feature_dimension: 1 output_batch_dimension: 0 output_feature_dimension: 1 }
      feature_group_count: 1
      batch_group_count: 1
      sparsity_config { %s }
    }
  }
}
"#;

fn to_string(module: &Module, name: &str) -> String {
    let mut out = String::new();
    Printer::new(module, Style::Long, false).instruction(module.find(name).unwrap(), &mut out);
    out
}

fn fixture(name: &str) -> Module<'static> {
    Module::parse(Cow::Owned(std::fs::read(format!("{}/tests/data/hlo_printing/{name}.pb", env!("CARGO_MANIFEST_DIR"))).unwrap()))
}

fn convolution(operands: &str, sparsity_config: &str) -> Module<'static> {
    module(&hlo_proto(&CONVOLUTION.replacen("%s", operands, 1).replacen("%s", sparsity_config, 1)))
}

#[test]
fn sparsity_config_to_string_rhs_only() {
    let conv = convolution("[1, 2, 4]", "rhs { block_size: 4 num_non_zero: 1 dimension: 0 stride: 1 idx: 2 }");
    assert_eq!(
        to_string(&conv, "convolution"),
        "%convolution = bf16[256,256]{1,0} convolution(%lhs, %rhs, %rhs_indices), dim_labels=bf_io->bf, sparsity_config={rhs={sparsity=1x4 dimension=0 stride=1 idx=2}}"
    );
}

#[test]
fn sparsity_config_to_string_lhs_and_rhs() {
    let conv = convolution("[1, 2, 3, 4]", "rhs { block_size: 4 num_non_zero: 1 dimension: 0 stride: 1 idx: 3 } lhs { block_size: 4 num_non_zero: 1 dimension: 0 stride: 1 idx: 2 }");
    assert_eq!(
        to_string(&conv, "convolution"),
        "%convolution = bf16[256,256]{1,0} convolution(%lhs, %rhs, %lhs_indices, %rhs_indices), dim_labels=bf_io->bf, sparsity_config={lhs={sparsity=1x4 dimension=0 stride=1 idx=2} rhs={sparsity=1x4 dimension=0 stride=1 idx=3}}"
    );
}

#[test]
fn block_scaling_config_to_string() {
    let side = |scale_idx: i32, zero_idx: Option<i32>, strides: &[i64], steps: &[i64]| Some(TensorBlockScalingConfig { scale_idx, zero_idx, strides: strides.to_vec(), steps: steps.to_vec() });
    assert_eq!(block_scaling_config_text(&BlockScalingConfig::default()), "");
    assert_eq!(block_scaling_config_text(&BlockScalingConfig { lhs: side(2, None, &[], &[]), rhs: None }), "lhs={scale_idx=2}");
    assert_eq!(block_scaling_config_text(&BlockScalingConfig { lhs: side(2, Some(0), &[], &[]), rhs: None }), "lhs={scale_idx=2 zero_idx=0}");
    assert_eq!(block_scaling_config_text(&BlockScalingConfig { lhs: side(2, Some(3), &[1, 4], &[1, 2]), rhs: None }), "lhs={scale_idx=2 zero_idx=3 strides=1x4 steps=1x2}");
    assert_eq!(block_scaling_config_text(&BlockScalingConfig { lhs: None, rhs: side(3, None, &[2, 4], &[]) }), "rhs={scale_idx=3 strides=2x4}");
    assert_eq!(
        block_scaling_config_text(&BlockScalingConfig { lhs: side(2, Some(0), &[1], &[]), rhs: side(3, Some(1), &[2], &[4]) }),
        "lhs={scale_idx=2 zero_idx=0 strides=1} rhs={scale_idx=3 zero_idx=1 strides=2 steps=4}"
    );
}

#[test]
fn print_compare_op_works_if_dead() {
    assert_eq!(to_string(&fixture("print_compare_op_works_if_dead"), "result"), "%result = pred[] compare(%p0, %p1), direction=GT, type=TOTALORDER");
}
