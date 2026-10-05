use super::opstats_adapter::op_stats;
use crate::tools::opstats::OpStats;
use crate::xplane::steps::TPU;
use std::sync::Arc;

fn combine(all: &[&str]) -> Arc<OpStats> {
    OpStats::combine(&all.iter().map(|text| Some(Arc::new(op_stats(text)))).collect::<Vec<_>>()).unwrap()
}

#[test]
fn combine_run_environment_with_unknown_device() {
    let dst_op_stats = combine(&[r#"run_environment { device_type: "TPU" }"#, r#"run_environment { device_type: "Device" }"#]);
    assert_eq!(dst_op_stats.extra.device_type, "TPU");
}

#[test]
fn combine_perf_env_order_zero() {
    let op_stats_1 = "perf_env { peak_tera_flops_per_second: 100 }";
    let op_stats_2 = "perf_env { peak_tera_flops_per_second: 0 }";
    assert_eq!(combine(&[op_stats_1, op_stats_2]).perf.peak_tera_flops, 100.0);
    assert_eq!(combine(&[op_stats_2, op_stats_1]).perf.peak_tera_flops, 100.0);
}

#[test]
fn combine_run_environment_with_mismatch_hardware_type() {
    let dst_op_stats = combine(&["run_environment { hardware_type: CPU_ONLY }", "run_environment { hardware_type: TPU }"]);
    assert_eq!(dst_op_stats.extra.hardware, TPU);
    assert!(dst_op_stats.tpu);
}

#[test]
fn combine_run_environment_power_metrics_single_host() {
    let src_op_stats = Some(Arc::new(op_stats(r#"run_environment { device_type: "TPU" }"#)));
    let dst_op_stats = OpStats::combine(std::slice::from_ref(&src_op_stats)).unwrap();
    assert!(Arc::ptr_eq(&dst_op_stats, src_op_stats.as_ref().unwrap()));
}
