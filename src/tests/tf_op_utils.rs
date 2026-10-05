use crate::tools::framework_op_stats::parse_tf_op;
use crate::xplane::derive::{Category, Category::*, tf_op};

fn check(full: &str, category: Category, name: &str, kind: &str, event_name: &str) {
    let op = tf_op(full);
    assert_eq!((op.category, op.name, op.kind, op.event_name().as_str()), (category, name, kind, event_name), "{full}");
    let parsed = parse_tf_op(full);
    assert_eq!((parsed.known, parsed.name.as_str(), parsed.kind.as_str()), (category != Category::Unknown, name, kind), "{full}");
}

macro_rules! checks { ($($name:ident: $($arg:expr),*;)*) => { $(#[test] fn $name() { check($($arg),*) })* } }

checks! {
    tf_op_test: "OpName:OpType", TensorFlow, "OpName", "OpType", "OpType";
    internal_tf_op_test: "OpName:_InternalOpType", TensorFlow, "OpName", "_InternalOpType", "_InternalOpType";
    tf_op_with_path_test: "path/to/name:OpType", TensorFlow, "path/to/name", "OpType", "OpType";
    short_dataset_op_test: "Iterator::Batch", TfData, "Iterator::Batch", "Dataset", "Iterator::Batch";
    long_dataset_op_test: "Iterator::Batch::Map::TfRecord", TfData, "Iterator::Batch::Map::TfRecord", "Dataset", "Iterator::TfRecord";
    trace_me_test: "MyTraceMe", Unknown, "MyTraceMe", "", "MyTraceMe";
    trace_me_with_colon_test: "RunStep/Server:54635", Unknown, "RunStep/Server:54635", "", "RunStep/Server:54635";
    trace_me_with_double_colon_test: "XLA::StartProgram", Unknown, "XLA::StartProgram", "", "XLA::StartProgram";
    trace_me_with_trailing_whitespace_test: "SessionRun ", Unknown, "SessionRun ", "", "SessionRun";
    memcpy_h_to_d_test: "MemcpyHToD", Memcpy, "MemcpyHToD", "MemcpyHToD", "MemcpyHToD";
    memcpy_d_to_h_test: "MemcpyDToH", Memcpy, "MemcpyDToH", "MemcpyDToH", "MemcpyDToH";
    memcpy_d_to_d_test: "MemcpyDToD", Memcpy, "MemcpyDToD", "MemcpyDToD", "MemcpyDToD";
    memcpy_h_to_h_test: "MemcpyHToH", Memcpy, "MemcpyHToH", "MemcpyHToH", "MemcpyHToH";
    jax_op_test: "op_name:op_type", Jax, "op_name", "op_type", "op_type";
    jax_op_with_colon_test: "op_name/op_type:", Jax, "op_name/op_type", "op_type", "op_type";
    jax_op_with_bracket_test: "op_name:op_type[array=([])]", Jax, "op_name", "op_type", "op_type";
    jax_op_with_bracket_and_trailing_colon_test: "op_name/op_type[array=([])]:", Jax, "op_name/op_type[array=([])]", "op_type", "op_type";
    other_xla_op_test: "namescope.1/namespace__opname2d:namespace__opname2d", Jax, "namescope.1/namespace__opname2d", "namespace__opname2d", "namespace__opname2d";
    op_without_type_test: "namescope/OpName_1:", TensorFlow, "namescope/OpName_1", "OpName", "OpName";
    op_type_with_understsl_test: "namescope/OpName_a:", TensorFlow, "namescope/OpName_a", "OpName_a", "OpName_a";
}

#[test]
fn infeed_enqueue_test() {
    let full = "input_pipeline_task0/while/body/_1/InfeedQueue/enqueue/1:InfeedEnqueueTuple";
    check(full, Category::TensorFlow, "input_pipeline_task0/while/body/_1/InfeedQueue/enqueue/1", "InfeedEnqueueTuple", "InfeedEnqueueTuple");
    assert!(tf_op(full).kind.starts_with("InfeedEnqueue"));
}

#[test]
fn name_scope_test() {
    let op = tf_op("scope-1/scope2/OpName:OpType");
    assert_eq!((op.category, op.name, op.kind), (Category::TensorFlow, "scope-1/scope2/OpName", "OpType"));
    assert_eq!(op.scopes(), ["scope-1", "scope2"]);
}
