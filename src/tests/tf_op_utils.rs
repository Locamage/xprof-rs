use crate::derive::{Category, tf_op};
use crate::framework_op_stats::parse_tf_op;

fn check(full: &str, category: Category, name: &str, kind: &str, event_name: &str) {
    let op = tf_op(full);
    assert_eq!((op.category, op.name, op.kind, op.event_name().as_str()), (category, name, kind, event_name), "{full}");
    let parsed = parse_tf_op(full);
    assert_eq!((parsed.known, parsed.name.as_str(), parsed.kind.as_str()), (category != Category::Unknown, name, kind), "{full}");
}

#[test]
fn tf_op_test() {
    check("OpName:OpType", Category::TensorFlow, "OpName", "OpType", "OpType");
}

#[test]
fn internal_tf_op_test() {
    check("OpName:_InternalOpType", Category::TensorFlow, "OpName", "_InternalOpType", "_InternalOpType");
}

#[test]
fn tf_op_with_path_test() {
    check("path/to/name:OpType", Category::TensorFlow, "path/to/name", "OpType", "OpType");
}

#[test]
fn short_dataset_op_test() {
    check("Iterator::Batch", Category::TfData, "Iterator::Batch", "Dataset", "Iterator::Batch");
}

#[test]
fn long_dataset_op_test() {
    check("Iterator::Batch::Map::TfRecord", Category::TfData, "Iterator::Batch::Map::TfRecord", "Dataset", "Iterator::TfRecord");
}

#[test]
fn trace_me_test() {
    check("MyTraceMe", Category::Unknown, "MyTraceMe", "", "MyTraceMe");
}

#[test]
fn trace_me_with_colon_test() {
    check("RunStep/Server:54635", Category::Unknown, "RunStep/Server:54635", "", "RunStep/Server:54635");
}

#[test]
fn trace_me_with_double_colon_test() {
    check("XLA::StartProgram", Category::Unknown, "XLA::StartProgram", "", "XLA::StartProgram");
}

#[test]
fn trace_me_with_trailing_whitespace_test() {
    check("SessionRun ", Category::Unknown, "SessionRun ", "", "SessionRun");
}

#[test]
fn infeed_enqueue_test() {
    let full = "input_pipeline_task0/while/body/_1/InfeedQueue/enqueue/1:InfeedEnqueueTuple";
    check(full, Category::TensorFlow, "input_pipeline_task0/while/body/_1/InfeedQueue/enqueue/1", "InfeedEnqueueTuple", "InfeedEnqueueTuple");
    assert!(tf_op(full).kind.starts_with("InfeedEnqueue"));
}

#[test]
fn memcpy_h_to_d_test() {
    check("MemcpyHToD", Category::Memcpy, "MemcpyHToD", "MemcpyHToD", "MemcpyHToD");
}

#[test]
fn memcpy_d_to_h_test() {
    check("MemcpyDToH", Category::Memcpy, "MemcpyDToH", "MemcpyDToH", "MemcpyDToH");
}

#[test]
fn memcpy_d_to_d_test() {
    check("MemcpyDToD", Category::Memcpy, "MemcpyDToD", "MemcpyDToD", "MemcpyDToD");
}

#[test]
fn memcpy_h_to_h_test() {
    check("MemcpyHToH", Category::Memcpy, "MemcpyHToH", "MemcpyHToH", "MemcpyHToH");
}

#[test]
fn jax_op_test() {
    check("op_name:op_type", Category::Jax, "op_name", "op_type", "op_type");
}

#[test]
fn jax_op_with_colon_test() {
    check("op_name/op_type:", Category::Jax, "op_name/op_type", "op_type", "op_type");
}

#[test]
fn jax_op_with_bracket_test() {
    check("op_name:op_type[array=([])]", Category::Jax, "op_name", "op_type", "op_type");
}

#[test]
fn jax_op_with_bracket_and_trailing_colon_test() {
    check("op_name/op_type[array=([])]:", Category::Jax, "op_name/op_type[array=([])]", "op_type", "op_type");
}

#[test]
fn other_xla_op_test() {
    check("namescope.1/namespace__opname2d:namespace__opname2d", Category::Jax, "namescope.1/namespace__opname2d", "namespace__opname2d", "namespace__opname2d");
}

#[test]
fn op_without_type_test() {
    check("namescope/OpName_1:", Category::TensorFlow, "namescope/OpName_1", "OpName", "OpName");
}

#[test]
fn op_type_with_understsl_test() {
    check("namescope/OpName_a:", Category::TensorFlow, "namescope/OpName_a", "OpName_a", "OpName_a");
}

#[test]
fn name_scope_test() {
    let op = tf_op("scope-1/scope2/OpName:OpType");
    assert_eq!((op.category, op.name, op.kind), (Category::TensorFlow, "scope-1/scope2/OpName", "OpType"));
    assert_eq!(op.scopes(), ["scope-1", "scope2"]);
}
