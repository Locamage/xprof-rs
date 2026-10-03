use super::xspace::{XPlane, XSpace, op_stats};
use serde_json::Value;

const KERNEL_DETAILS: &str = "regs:32\nstatic_shared:0\ndynamic_shared:16384\ngrid:2,1,1\nblock:32,1,1\nocc_pct:100";

fn add_tensor_flow_op_event(tf_op_fullname: String, start_timestamp_ns: i64, duration_ns: i64, kernel_name: &str, kernel_details: Option<&str>, plane: &mut XPlane, line: i64) {
    let mut stats = vec![("tf_op", tf_op_fullname.into())];
    stats.extend(kernel_details.map(|details| ("kernel_details", details.into())));
    plane.event(line, kernel_name, start_timestamp_ns * 1000, duration_ns * 1000, &stats);
}

fn nano_to_micro(ns: i64) -> f64 {
    ns as f64 / 1e3
}

#[test]
fn gpu_tf_stats() {
    let (tf_op1, tf_op2, tf_op3) = ("TfOp1", "TfOp2", "Conv2D");
    let (kernel1_start_ns, kernel1_duration_ns, kernel2_start_ns, kernel2_duration_ns) = (100000, 8000, 110000, 10000);
    let (kernel3_start_ns, kernel3_duration_ns, kernel4_start_ns, kernel4_duration_ns, kernel5_start_ns, kernel5_duration_ns) = (120000, 10000, 130000, 10000, 150000, 10000);
    let mut space = XSpace::default();
    let device_plane = space.gpu(0);
    device_plane.line(10);
    add_tensor_flow_op_event(format!("{tf_op1}:{tf_op1}"), kernel1_start_ns, kernel1_duration_ns, "kernel1", None, device_plane, 10);
    add_tensor_flow_op_event(format!("{tf_op1}:{tf_op1}"), kernel2_start_ns, kernel2_duration_ns, "kernel2", None, device_plane, 10);
    device_plane.line(20);
    add_tensor_flow_op_event(format!("{tf_op1}:{tf_op1}"), kernel1_start_ns, kernel1_duration_ns, "kernel1", None, device_plane, 20);
    add_tensor_flow_op_event(format!("{tf_op1}:{tf_op1}"), kernel2_start_ns, kernel2_duration_ns, "kernel2", None, device_plane, 20);
    add_tensor_flow_op_event(format!("{tf_op2}:{tf_op2}"), kernel3_start_ns, kernel3_duration_ns, "kernel3", None, device_plane, 20);
    add_tensor_flow_op_event(format!("{tf_op3}:{tf_op3}"), kernel4_start_ns, kernel4_duration_ns, "volta_fp16_s884gemm", Some(KERNEL_DETAILS), device_plane, 20);
    add_tensor_flow_op_event(format!("{tf_op3}:{tf_op3}"), kernel5_start_ns, kernel5_duration_ns, "kernel5", Some(KERNEL_DETAILS), device_plane, 20);
    let op_stats = op_stats(&[space]).unwrap();
    let tf_stats: Value = serde_json::from_str(&crate::framework_op_stats::json(&op_stats)).unwrap();
    let records: Vec<Vec<Value>> = tf_stats[0]["rows"].as_array().unwrap().iter().map(|row| row["c"].as_array().unwrap().iter().map(|cell| cell["v"].clone()).collect()).collect();
    assert_eq!(records.len(), 4);
    let (op_type, op_name, occurrences, total_self_time, gpu_tensorcore_utilization) = (2, 3, 4, 7, 17);
    assert_eq!((&records[0][op_name], &records[0][op_type], &records[0][occurrences]), (&tf_op1.into(), &tf_op1.into(), &2.0.into()));
    assert_eq!(records[0][total_self_time], nano_to_micro(kernel1_duration_ns) * 2.0 + nano_to_micro(kernel2_duration_ns) * 2.0);
    assert_eq!((&records[1][op_name], &records[1][op_type], &records[1][occurrences]), (&tf_op3.into(), &tf_op3.into(), &1.0.into()));
    assert_eq!(records[1][total_self_time], nano_to_micro(kernel4_duration_ns) + nano_to_micro(kernel5_duration_ns));
    assert_eq!(records[1][gpu_tensorcore_utilization], 0.5);
    assert_eq!((&records[2][op_name], &records[2][op_type], &records[2][occurrences]), (&tf_op2.into(), &tf_op2.into(), &1.0.into()));
    assert_eq!(records[2][total_self_time], nano_to_micro(kernel3_duration_ns));
}
