use super::opstats_adapter::op_stats;
use serde_json::Value;

const MAX_ERROR: f64 = 1E-10;
const PEAK_GIGAFLOPS_PER_SECOND: f64 = 1000.0;

fn normalized_gigaflops(metrics: &str) -> f64 {
    let text = format!(r#"device_op_metrics_db {{ total_time_ps: 100 metrics_db {{ name: "op" category: "convolution" self_time_ps: 100 {metrics} }} }} perf_env {{ peak_tera_flops_per_second: 1 }}"#);
    let profile: Value = serde_json::from_str(&crate::op_profile::json(&op_stats(&text), Some("category"))).unwrap();
    let op = &profile["byCategory"]["children"][0]["children"][0];
    assert_eq!(op["name"], "op");
    op["metrics"]["uncappedFlops"].as_f64().unwrap() * PEAK_GIGAFLOPS_PER_SECOND
}

#[test]
fn giga_flops_per_second_per_core_normalized_on_dvfs() {
    let rate = normalized_gigaflops("time_ps: 100 normalized_time_ps: 200 flops_v2: 1000 occurrences: 1 num_cores: 1");
    assert!((rate - 5000.0).abs() <= MAX_ERROR, "{rate}");
}

#[test]
fn giga_flops_per_second_per_core_normalized_on_dvfs_fallback() {
    let rate = normalized_gigaflops("time_ps: 100 normalized_time_ps: 0 flops_v2: 1000 occurrences: 1 num_cores: 1");
    assert!((rate - 10000.0).abs() <= MAX_ERROR, "{rate}");
}
