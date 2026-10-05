use super::cli_support::{ok, scratch};
use super::e2e_oracles::{MEDIUM_STEP_PS, TRAINING_STEP_PS, datasheet_violations, demo, oracle_duty_cycle, oracle_ridge_point, oracle_step_time_ms, session, training_trace};
use crate::cli::json::J;
use std::path::{Path, PathBuf};

const RELATIVE_TOLERANCE: f64 = 0.01;

fn training(name: &str, step_ps: i64) -> PathBuf {
    session(&scratch(name), "host.xplane.pb", &training_trace(step_ps))
}

fn query(command: &str, path: &Path, flags: &[&str]) -> J {
    let argv: Vec<&str> = [command, path.to_str().unwrap()].into_iter().chain(flags.iter().copied()).collect();
    let result = ok(&argv);
    assert!(!result.has("error"), "{result:?}");
    result
}

fn overview_step_time(path: &Path) -> f64 {
    query("get_overview", path, &[]).at("performance_summary").at("steptime_ms_average").text().replace("ms", "").trim().parse().unwrap()
}

fn number(value: &J) -> f64 {
    value.float().unwrap()
}

#[test]
fn test_n01_t1_steptime_fidelity() {
    let path = training("n01", TRAINING_STEP_PS);
    let oracle = oracle_step_time_ms(&path);
    assert!(oracle > 0.0);
    let reported = overview_step_time(&path);
    assert!(reported > 0.0);
    assert!((reported - oracle).abs() <= RELATIVE_TOLERANCE * oracle, "{reported} vs {oracle}");
}

#[test]
fn test_n02_t1_duty_cycle_fidelity() {
    let path = training("n02", TRAINING_STEP_PS);
    let oracle = oracle_duty_cycle(&path);
    assert!(oracle > 0.0 && oracle <= 1.0);
    let summary = query("get_overview", &path, &[]).at("performance_summary").clone();
    if let Some(reported) = summary.get("device_compute_duty_cycle_percent").or_else(|| summary.get("duty_cycle")) {
        let value = reported.text().replace('%', "").trim().parse::<f64>().unwrap();
        let value = if value > 1.0 { value / 100.0 } else { value };
        assert!((value - oracle).abs() <= 0.01);
    }
}

#[test]
fn test_n03_t2_steptime_fidelity() {
    let path = training("n03", MEDIUM_STEP_PS);
    let oracle = oracle_step_time_ms(&path);
    assert!(oracle > 0.0);
    let reported = overview_step_time(&path);
    assert!((reported - oracle).abs() <= RELATIVE_TOLERANCE * oracle, "{reported} vs {oracle}");
}

#[test]
fn test_n04_t1_step_trace_fidelity() {
    let path = training("n04", TRAINING_STEP_PS);
    let result = query("get_step_trace", &path, &[]);
    let summary = result.at("summary");
    assert!(summary.at("total_steps").int().unwrap() > 0);
    let average = number(summary.at("step_time_ms_average"));
    let oracle = oracle_step_time_ms(&path);
    assert!((average - oracle).abs() <= RELATIVE_TOLERANCE * oracle);
    assert!(number(summary.at("compute_time_ms_average")) > 0.0);
    assert!(number(summary.at("compute_percent")) > 95.0 && number(summary.at("compute_percent")) <= 100.0);
    assert_eq!(summary.at("primary_bottleneck").str(), Some("Compute"));
    assert!(number(summary.at("idle_time_ms_average")) >= 0.0);
    let parts: f64 = ["compute", "communication", "infeed", "outfeed", "idle"].iter().map(|part| number(summary.at(&format!("{part}_time_ms_average")))).sum();
    assert!((parts - average).abs() <= RELATIVE_TOLERANCE * average);
    for step in result.at("step_breakdown").items() {
        let total = number(step.at("step_time_ms"));
        assert!(number(step.at("compute_time_ms")) > 0.0 && number(step.at("compute_time_ms")) <= total);
        assert_eq!(step.at("bottleneck").str(), Some("Compute"));
        assert!(number(step.at("idle_time_ms")) >= 0.0);
        let parts: f64 = ["compute", "communication", "infeed", "outfeed", "idle"].iter().map(|part| number(step.at(&format!("{part}_time_ms")))).sum();
        assert!((parts - total).abs() <= RELATIVE_TOLERANCE * total);
    }
}

#[test]
fn test_n05_kpi_metrics_contract() {
    let path = training("n05", TRAINING_STEP_PS);
    let result = query("get_kpi_metrics", &path, &[]);
    let step: f64 = result.at("step_time_ms").text().replace("ms", "").trim().parse().unwrap();
    assert!(step > 0.0);
    let duty: f64 = result.at("duty_cycle_percent").text().replace('%', "").trim().parse().unwrap();
    assert!((0.0..=100.0).contains(&duty));
}

#[test]
fn test_n06_roofline_model_ridge_point() {
    let path = training("n06", TRAINING_STEP_PS);
    let info = query("get_roofline_model", &path, &[]).at("device_info").clone();
    let (flops, bandwidth, ridge) = (number(info.at("peak_flop_rate")), number(info.at("peak_hbm_bw")), number(info.at("hbm_ridge_point")));
    let expected = oracle_ridge_point(flops, bandwidth);
    assert!((ridge - expected).abs() <= RELATIVE_TOLERANCE * expected, "{ridge} vs {expected}");
}

#[test]
fn test_n08_top_hlo_ops_sorting() {
    let path = training("n08", TRAINING_STEP_PS);
    let result = query("get_top_hlo_ops", &path, &["--limit=5"]);
    let times: Vec<f64> = result.at("top_by_time").items().iter().map(|op| number(op.at("total_self_time_ms"))).collect();
    assert!(times.len() >= 2);
    assert!(times.windows(2).all(|pair| pair[0] >= pair[1]));
}

#[test]
fn test_n11_memory_profile_peak_bounds() {
    let path = demo(&scratch("n11"));
    let result = query("get_memory_profile", &path, &[]);
    let (capacity, peak) = (number(result.at("memory_capacity_gib")), number(result.at("peak_memory_usage_gib")));
    let details = result.at("peak_usage_details");
    let (stack, heap, free) = (number(details.at("stack_reservation_gib")), number(details.at("heap_allocation_gib")), number(details.at("free_memory_gib")));
    assert!(capacity > 0.0 && peak <= capacity);
    assert!(stack >= 0.0 && heap >= 0.0 && (stack + heap - peak).abs() <= 0.1);
    assert!(free >= 0.0 && (peak + free - capacity).abs() <= 0.1);
}

#[test]
fn test_n15_disjoint_interval_union_invariants() {
    let path = training("n15", TRAINING_STEP_PS);
    let violations = datasheet_violations(oracle_step_time_ms(&path), oracle_duty_cycle(&path));
    assert!(violations.is_empty(), "{violations:?}");
}
