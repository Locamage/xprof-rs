use crate::hlo::cost::bit_width;
use crate::hlo::cost::tests::{parameter, shape};

const BF16: i32 = 16;
const S4: i32 = 21;

#[test]
fn get_input_bitwidths() {
    let operands = [parameter(1, 0, shape(BF16, &[2, 4])), parameter(2, 1, shape(S4, &[2, 4]))];
    assert_eq!(operands.map(|operand| bit_width(operand.shape.unwrap().element_type)), [16, 4]);
}
