use crate::xplane::gpu::tf_op_fullname;

#[test]
fn tf_op_fullname_test() {
    assert_eq!(tf_op_fullname("", ""), "");
    assert_eq!(tf_op_fullname("", "XLA_Args"), "XLA_Args:XLA_Args");
    assert_eq!(tf_op_fullname("op_type", "XLA_Retvals"), "XLA_Retvals:op_type");
    assert_eq!(tf_op_fullname("op_type", "op_name"), "op_name:op_type");
}
