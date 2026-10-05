use super::opstats_adapter::op_stats;
use serde_json::Value;

const GIBI_IN_GIGA: f64 = (1u64 << 30) as f64 / 1.0e9;

fn ridge_point(peak_gigaflops_per_second: f64, peak_gibibytes_per_second: f64) -> String {
    let text = format!(
        "run_environment {{ hardware_type: TPU }} perf_env {{ peak_tera_flops_per_second: {} peak_bws_giga_bytes_per_second: [{}] }}",
        peak_gigaflops_per_second / 1e3,
        peak_gibibytes_per_second * GIBI_IN_GIGA
    );
    let roofline: Value = serde_json::from_str(&crate::tools::roofline::json(&op_stats(&text))).unwrap();
    roofline[0]["p"]["hbm_ridge_point"].as_str().unwrap().to_string()
}

#[test]
fn ridge_point_normal_case() {
    assert_eq!(ridge_point(1000.0, 100.0), "9.31323");
}

#[test]
fn ridge_point_another_normal_case() {
    assert_eq!(ridge_point(500.0, 200.0), "2.32831");
}

#[test]
fn ridge_point_zero_memory_bandwidth() {
    assert_eq!(ridge_point(100.0, 0.0), "0");
}

#[test]
fn ridge_point_zero_flops() {
    assert_eq!(ridge_point(0.0, 100.0), "0");
}
