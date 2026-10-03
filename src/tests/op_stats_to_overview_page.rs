use super::opstats_adapter::op_stats;
use serde_json::Value;

fn analysis(text: &str) -> Value {
    let overview: Value = serde_json::from_str(&crate::overview_page::json(&op_stats(text), &[])).unwrap();
    overview[0]["p"].clone()
}

#[test]
fn tpu_duty_cycle() {
    let analysis = analysis(
        r#"run_environment { device_type: "TPU" }
           device_op_metrics_db { busy_time_ps: 70 idle_time_ps: 30 }"#,
    );
    assert_eq!(analysis["device_duty_cycle_percent"], "70.0%");
}

#[test]
fn high_confidence_tpu_duty_cycle() {
    let analysis = analysis(
        r#"run_environment { device_type: "TPU" }
           device_op_metrics_db { busy_time_ps: 70 idle_time_ps: 30 busy_time_high_confidence_ps: 60 idle_time_high_confidence_ps: 20 }"#,
    );
    assert_eq!(analysis["device_duty_cycle_percent"], "70.0%");
    assert_eq!(analysis["device_duty_cycle_high_confidence_percent"], "75.0%");
    assert_eq!(analysis["idle_time_high_confidence_ps"], "20");
    assert!(analysis.get("device_duty_cycle_high_confidence_percent").is_some());
    assert!(analysis.get("idle_time_high_confidence_ps").is_some());
}

#[test]
fn high_confidence_zero_time_test() {
    let analysis = analysis(
        r#"run_environment { device_type: "TPU" }
           device_op_metrics_db { busy_time_high_confidence_ps: 0 idle_time_high_confidence_ps: 0 }"#,
    );
    assert_eq!(analysis["device_duty_cycle_high_confidence_percent"], "0.0%");
}
