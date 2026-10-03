use super::xspace::{V, XPlane, XSpace};
use crate::counter_ids::{V6E_IDS, V7X_IDS};
use crate::counters::{kernel_utilization, utilization_viewer};
use prost::Message;
use serde_json::Value;

const V7X_CYCLES: &str = "VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT";
const V7X_TC: &str = "VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_";
const V6E_TC: &str = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_";

fn v7x(name: &str) -> u64 {
    V7X_IDS[name]
}

fn v7x_tc(suffix: &str) -> u64 {
    v7x(&format!("{V7X_TC}{suffix}"))
}

fn v6e_tc(suffix: &str) -> u64 {
    V6E_IDS[format!("{V6E_TC}{suffix}").as_str()]
}

fn device(space: &mut XSpace, name: &str, device_id: Option<i64>, device_type: &str) -> usize {
    let plane = space.plane(name);
    if let Some(device_id) = device_id {
        plane.add_stat("device_id", device_id);
    }
    plane.add_stat("device_type_string", device_type);
    space.planes.len() - 1
}

fn counter(plane: &mut XPlane, line: i64, name: &str, id: u64, value: f64, duration_ns: i64) {
    plane.event(line, name, 0, duration_ns * 1000, &[("performance_counter_id", id.into()), ("counter_value", V::Double(value))]);
}

fn tpu(device_type: &str, line_name: &str, counters: &[(&str, u64, f64)]) -> XSpace {
    let mut space = XSpace::default();
    let index = device(&mut space, "/device:TPU:0", Some(0), device_type);
    let plane = &mut space.planes[index];
    plane.named_line(0, line_name);
    for &(name, id, value) in counters {
        counter(plane, 0, name, id, value, 0);
    }
    space
}

fn utilization(space: &XSpace) -> String {
    utilization_viewer(&space.encode_to_vec())
}

fn kernels(space: &XSpace) -> Value {
    serde_json::from_str(&kernel_utilization(&space.encode_to_vec(), &Default::default())).unwrap()
}

#[test]
fn basic_tpu_v7x() {
    let json_str = utilization(&tpu("TPU v7x", "", &[("CYCLES", v7x(V7X_CYCLES), 1000.0), ("SCALAR_ALU_0", v7x_tc("SCALAR_ALU_INSTRUCTION_0"), 500.0)]));
    assert!(json_str.contains("Scalar Unit") && json_str.contains("1000") && json_str.contains("500"), "{json_str}");
}

#[test]
fn vpu_util_tpu_v7x() {
    let json_str = utilization(&tpu("TPU v7x", "", &[("CYCLES", v7x(V7X_CYCLES), 1000.0), ("VPU_FADD_0", v7x_tc("VPU_VALU_FADD_OPS_0"), 250.0)]));
    assert!(json_str.contains("VPU Util") && json_str.contains("250") && json_str.contains("4000"), "{json_str}");
}

#[test]
fn vpu_util_tpu_v6_e() {
    let json_str = utilization(&tpu("TPU v6 Lite", "", &[("CYCLES", v6e_tc("CYCLES"), 1000.0), ("VPU_FADD_0", v6e_tc("VPU_VALU_FADD_OPS_0"), 250.0)]));
    assert!(json_str.contains("VPU Util") && json_str.contains("250") && json_str.contains("4000"), "{json_str}");
}

#[test]
fn counter_value_out_of_bounds_of_uint64() {
    let json_str = utilization(&tpu(
        "TPU v7x",
        "",
        &[("CYCLES", v7x(V7X_CYCLES), 1000.0), ("SCALAR_ALU_0", v7x_tc("SCALAR_ALU_INSTRUCTION_0"), -500.0), ("SCALAR_ALU_1", v7x_tc("SCALAR_ALU_INSTRUCTION_1"), 2.0e19)],
    ));
    assert!(json_str.contains("Scalar Unit") && json_str.contains("1.84467") && json_str.contains("e+19"), "{json_str}");
}

#[test]
fn unsupported_device_ignored() {
    let mut space = XSpace::default();
    device(&mut space, "/device:TPU:0", Some(0), "TPU v5e");
    assert!(utilization(&space).contains("\"rows\":[]"));
}

#[test]
fn na_n_counter_value_handled_safely() {
    let json_str = utilization(&tpu("TPU v7x", "", &[("CYCLES", v7x(V7X_CYCLES), 1000.0), ("SCALAR_ALU_0", v7x_tc("SCALAR_ALU_INSTRUCTION_0"), f64::NAN)]));
    assert!(json_str.contains("Scalar Unit") && json_str.contains('0'), "{json_str}");
}

#[test]
fn basic_tpu_v7x_with_kernel_attribution() {
    let mut space = XSpace::default();
    let index = device(&mut space, "/device:TPU:0", Some(0), "TPU v7x");
    let plane = &mut space.planes[index];
    plane.named_line(100, "PALLAS");
    plane.event(100, "pallas_matmul_fwd", 0, 10000 * 1000, &[]);
    plane.named_line(0, "counters_0");
    counter(plane, 0, "CYCLES", v7x(V7X_CYCLES), 10000.0, 10000);
    counter(plane, 0, "MXU_BUSY", v7x_tc("MXU_BUSY_1"), 16000.0, 0);
    counter(plane, 0, "MXU_BF16", v7x_tc("MATMUL_VREG_BF16_MXU_0"), 8000.0, 0);
    let result = kernels(&space);
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 1);
    assert_eq!(result["devices"][0]["device_id"], 0);
    assert_eq!(result["devices"][0]["kernels"].as_array().unwrap().len(), 1);
    let kernel = &result["devices"][0]["kernels"][0];
    assert_eq!(kernel["kernel_name"], "pallas_matmul_fwd");
    assert!(kernel["mxu_utilization"].as_f64().unwrap() > 0.0);
    assert_eq!(kernel["mxu_cycles_breakdown"]["BF16"], 100.0);
    assert_eq!(kernel["mxu_is_anomaly"], false);
}

#[test]
fn unsupported_device_ignored_by_kernel_utilization() {
    let mut space = XSpace::default();
    device(&mut space, "/device:TPU:0", Some(0), "TPU v5e");
    let result = kernels(&space);
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 0);
}

#[test]
fn anomaly_and_nan_safety() {
    let result =
        kernels(&tpu("TPU v7x", "counters_anomalous_kernel", &[("CYCLES", v7x(V7X_CYCLES), 1000.0), ("MXU_BUSY", v7x_tc("MXU_BUSY_2"), 5000.0), ("VPU_NAN", v7x_tc("VPU_VALU_FADD_OPS_0"), f64::NAN)]));
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 1);
    assert_eq!(result["devices"][0]["kernels"].as_array().unwrap().len(), 1);
    let kernel = &result["devices"][0]["kernels"][0];
    assert_eq!(kernel["kernel_name"], "anomalous_kernel");
    assert!(kernel["mxu_utilization"].as_f64().unwrap() > 100.0);
    assert_eq!(kernel["mxu_is_anomaly"], true);
}

#[test]
fn multi_core_other_metrics_aggregation_tpu_v7x() {
    let result = kernels(&tpu(
        "TPU v7x",
        "counters_multicore_kernel",
        &[
            ("DIE0_CYCLES", v7x(V7X_CYCLES), 1000.0),
            ("DIE1_CYCLES", v7x("VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT"), 1000.0),
            ("VPU_FADD_0", v7x_tc("VPU_VALU_FADD_OPS_0"), 250.0),
        ],
    ));
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 1);
    assert_eq!(result["devices"][0]["kernels"].as_array().unwrap().len(), 1);
    let kernel = &result["devices"][0]["kernels"][0];
    assert_eq!(kernel["kernel_name"], "multicore_kernel");
    assert_eq!(kernel["other_metrics"]["VPU Utilization"], 3.13);
}

#[test]
fn device_id_fallback_from_plane_name() {
    let mut space = XSpace::default();
    let index = device(&mut space, "/device:TPU:3", None, "TPU v7x");
    let plane = &mut space.planes[index];
    plane.named_line(0, "counters_test_kernel");
    counter(plane, 0, "CYCLES", v7x(V7X_CYCLES), 1000.0, 0);
    let result = kernels(&space);
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 1);
    assert_eq!(result["devices"][0]["device_id"], 3);
    assert_eq!(result["devices"][0]["kernels"].as_array().unwrap().len(), 1);
    assert_eq!(result["devices"][0]["kernels"][0]["kernel_name"], "test_kernel");
}

#[test]
fn multi_core_hbm_bandwidth_aggregation_tpu_v7x() {
    let result = kernels(&tpu(
        "TPU v7x",
        "counters_hbm_kernel",
        &[
            ("DIE0_CYCLES", v7x(V7X_CYCLES), 10000.0),
            ("DIE0_MISC_CYCLES", v7x_tc("CYCLES"), 10000.0),
            ("DIE1_CYCLES", v7x("VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT"), 10000.0),
            ("DIE1_MISC_CYCLES", v7x("VF_CHIP_DIE1_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES"), 10000.0),
            ("HBM_WR", v7x("VF_CHIP_DIE0_HBM_0_SS_HBMC_0_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_WR_REQ_PS0"), 50000.0),
        ],
    ));
    assert_eq!(result["status"], "SUCCESS");
    assert_eq!(result["devices"].as_array().unwrap().len(), 1);
    let other_metrics = result["devices"][0]["kernels"][0]["other_metrics"].as_object().unwrap();
    assert!(other_metrics.contains_key("HBM Bandwidth Utilization"));
    assert!(!other_metrics.contains_key("HBM Rd+Wr - core 0"));
    assert!(!other_metrics.contains_key("HBM Rd+Wr - core 1"));
}
