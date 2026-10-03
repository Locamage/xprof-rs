use super::cli_support::{Fake, args, json};
use crate::cli::json::J;
use crate::cli::steps::check_host_boundness;
use crate::cli::{Error, Kind};

const MATMUL: &str = r#"{"byCategory": {"children": [{"name": "matmul", "metrics": {"occurrences": 1, "rawTime": 3000.0}}]}}"#;
const BARRIER: &str = r#"{"traceEvents": [{"name": "barrier-cores", "dur": 50000.0}]}"#;
const HIGH_DUTY: &str = r#"[{"p": {"device_duty_cycle_percent": "98.3%"}}, {"p": {"steptime_ms_average": "100.0"}, "rows": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}, {"p": {"device_core_count": "8", "host_count": "1"}}]"#;

fn overview(duty: &str, cores: &str, hosts: &str) -> String {
    format!(
        r#"[{{"p": {{"device_duty_cycle_percent": "{duty}"}}}}, {{"p": {{"steptime_ms_average": "100.0"}}, "rows": [{{}}, {{}}, {{}}, {{}}, {{}}, {{}}, {{}}, {{}}, {{}}, {{}}]}}, {{"p": {{"device_core_count": "{cores}", "host_count": "{hosts}"}}}}]"#
    )
}

fn utilization(host: i64, [idleness, hbm, read, write]: [f64; 4]) -> String {
    let row = |name: &str, achieved: f64| format!("{host},0,0,0,{name},{achieved},100.0,percent\n");
    format!("Host,Device,Sample,Node,Name,Achieved,Peak,Unit\n{}{}{}{}", row("No MXU Busy", idleness), row("HBM Rd+Wr", hbm), row("ICI (Read)", read), row("ICI (Write)", write))
}

fn session(overview: String, hlo: &str, trace: &str, utilization: String, hosts: &[&str]) -> Fake {
    let (hlo, trace) = (hlo.to_string(), trace.to_string());
    Fake::tools(move |tool| match tool {
        "overview_page.json" => Some(overview.clone()),
        "hlo_op_profile.json" => Some(hlo.clone()),
        "trace_viewer.json" => Some(trace.clone()),
        "utilization_viewer.json" => Some(utilization.clone()),
        _ => None,
    })
    .with_hosts(hosts)
}

fn check(fake: &Fake, flags: &[(&str, J)]) -> J {
    json(check_host_boundness(fake, &args("test-session", flags)))
}

fn metric(result: &J, key: &str) -> f64 {
    result.at("metrics").at(key).float().unwrap()
}

#[test]
fn test_insufficient_data_when_zero_duration() {
    let data = r#"[{"p": {"device_duty_cycle_percent": "0.0%", "device_idle_time_percent": "100.0%"}}, {"p": {"steptime_ms_average": "0.0"}, "rows": []}, {"p": {"device_core_count": "8"}}]"#;
    let result = check(&Fake::fixed(data), &[]);
    assert_eq!(result.at("status").str(), Some("INSUFFICIENT_DATA"));
    assert!(result.at("reasons").items()[0].text().contains("lacks valid step timing duration telemetry"));
}

#[test]
fn test_unknown_by_missing_hlo_when_duty_cycle_high() {
    let result = check(&session(HIGH_DUTY.into(), "{}", "{}", utilization(0, [90.0, 10.0, 5.0, 5.0]), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("UNKNOWN"));
    assert_eq!(metric(&result, "tpu_duty_cycle_percent"), 98.3);
    assert!(result.at("reasons").items()[0].text().contains("TPU duty cycle is high"));
}

#[test]
fn test_host_bound_success_all_thresholds_met() {
    let result = check(&session(overview("15.0%", "8", "1"), MATMUL, BARRIER, utilization(0, [99.52, 2.98, 16.88, 16.88]), &["test-host"]), &[]);
    assert_eq!(result.at("status").str(), Some("HOST_BOUND"));
    assert_eq!(metric(&result, "idle_time_ratio_percent"), 100.0);
    assert_eq!(metric(&result, "equivalent_idle_chips"), 4.0);
    assert_eq!(metric(&result, "mxu_idleness_percent"), 99.52);
    assert_eq!(metric(&result, "hbm_bandwidth_utilization_percent"), 2.98);
    assert_eq!(metric(&result, "ici_read_utilization_percent"), 16.88);
    assert_eq!(metric(&result, "scaled_barrier_time_ms"), 500.0);
    assert!(result.at("reasons").items()[0].text().contains("Workload is host-bound"));
    assert!(result.at("recommendations").items()[0].text().contains("Hardware waste = 4.0 idle chips"));
}

#[test]
fn test_host_bound_despite_high_duty_cycle() {
    let result = check(&session(overview("90.0%", "8", "1"), MATMUL, BARRIER, utilization(0, [99.52, 2.98, 16.88, 16.88]), &["test-host"]), &[]);
    assert_eq!(result.at("status").str(), Some("HOST_BOUND"));
    assert_eq!(metric(&result, "tpu_duty_cycle_percent"), 90.0);
    assert_eq!(metric(&result, "idle_time_ratio_percent"), 100.0);
    assert_eq!(metric(&result, "equivalent_idle_chips"), 4.0);
}

#[test]
fn test_not_host_bound_by_low_idle_ratio() {
    let hlo = r#"{"byCategory": {"children": [{"name": "matmul", "metrics": {"occurrences": 1, "rawTime": 7360000000000.0}}]}}"#;
    let result = check(&session(overview("90.0%", "8", "1"), hlo, "{}", utilization(0, [10.0, 50.0, 10.0, 10.0]), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("NOT_HOST_BOUND"));
    assert_eq!(metric(&result, "tpu_duty_cycle_percent"), 90.0);
    assert_eq!(metric(&result, "idle_time_ratio_percent"), 8.7);
    assert!(result.at("reasons").items()[0].text().contains("TPU duty cycle (90.0%) is high and idle ratio is low."));
}

#[test]
fn test_not_host_bound_hbm_bottleneck() {
    let result = check(&session(overview("15.0%", "8", "1"), MATMUL, "{}", utilization(0, [90.0, 65.0, 10.0, 10.0]), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("NOT_HOST_BOUND"));
    assert!(result.at("reasons").items()[1].text().contains("HBM Bandwidth Utilization (65.0%) is >= 30.0%"));
}

#[test]
fn test_not_host_bound_ici_bottleneck() {
    let result = check(&session(overview("15.0%", "8", "1"), MATMUL, "{}", utilization(0, [90.0, 15.0, 45.0, 10.0]), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("NOT_HOST_BOUND"));
    assert!(result.at("reasons").items()[1].text().contains("ICI Utilization (Read: 45.0%, Write: 10.0%) is >= 30.0%"));
}

#[test]
fn test_multi_host_utilization_fallback() {
    let result = check(&session(overview("15.0%", "8", "2"), MATMUL, "{}", utilization(1, [95.0, 10.0, 10.0, 10.0]), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("HOST_BOUND"));
    assert_eq!(metric(&result, "mxu_idleness_percent"), 95.0);
}

#[test]
fn test_error_handling_when_overview_empty() {
    let error = check_host_boundness(&Fake::fixed(""), &args("sess_missing", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_bypass_cache_plumbing() {
    let fake = session(HIGH_DUTY.into(), "{}", "{}", utilization(0, [90.0, 10.0, 5.0, 5.0]), &[]);
    check(&fake, &[("bypass_cache", J::Bool(true))]);
    let calls = fake.calls.borrow();
    let bypass = ("bypass_cache".to_string(), "True".to_string());
    assert!(calls.contains(&("overview_page.json".to_string(), vec![("format".to_string(), "json".to_string()), bypass.clone()])));
    assert!(calls.iter().any(|(tool, params)| tool == "utilization_viewer.json" && params.contains(&bypass)));
}

#[test]
fn test_func_name_argument_support() {
    let result = check(&session(overview("15.0%", "8", "1"), MATMUL, "{}", utilization(0, [99.0, 5.0, 5.0, 5.0]), &[]), &[("func_name", J::from("train_step"))]);
    assert_eq!(result.at("status").str(), Some("HOST_BOUND"));
    assert!(result.at("recommendations").items()[2].text().contains("Use Lumini xprof_check_host_boundness with func_name"));
}

#[test]
fn test_traverse_and_sum_hlo_times_comprehensive() {
    let empty = check(&session(overview("15.0%", "1", "1"), r#"{"byCategory": {}}"#, "{}", String::new(), &[]), &[]);
    assert_eq!([metric(&empty, "scaled_compute_time_ms"), metric(&empty, "scaled_hbm_time_ms"), metric(&empty, "scaled_ici_time_ms")], [0.0; 3]);
    let tree = r#"{"byCategory": {"children": [{"name": "all-reduce", "metrics": {"occurrences": 1, "rawTime": 500000000000.0}}, {"name": "copy-start", "metrics": {"occurrences": 1, "rawTime": 300000000000.0}}, {"name": "custom-call", "xla": {"category": "convolution"}, "metrics": {"occurrences": 2, "rawTime": 1200000000000.0}}, {"name": "zero-occurrences-op", "metrics": {"occurrences": 0, "rawTime": 9999000000000.0}}]}}"#;
    let result = check(&session(overview("15.0%", "1", "1"), tree, "{}", String::new(), &[]), &[]);
    assert_eq!(metric(&result, "scaled_compute_time_ms"), 1200.0);
    assert_eq!(metric(&result, "scaled_hbm_time_ms"), 300.0);
    assert_eq!(metric(&result, "scaled_ici_time_ms"), 500.0);
}

#[test]
fn test_dict_overview_format_success() {
    let data = r#"{"performance_summary": {"device_duty_cycle_percent": "15.0%"}, "step_time": {"steptime_ms_average": "100.0"}, "run_environment": {"device_core_count": "8", "host_count": "1"}, "rows": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}"#;
    let result = check(&session(data.into(), MATMUL, "{}", utilization(0, [99.52, 2.98, 16.88, 16.88]), &["host-0"]), &[]);
    assert_eq!(result.at("status").str(), Some("HOST_BOUND"));
    assert_eq!(metric(&result, "tpu_duty_cycle_percent"), 15.0);
    assert_eq!(metric(&result, "mxu_idleness_percent"), 99.52);
}

#[test]
fn test_utilization_with_none_and_missing_values() {
    let result = check(&session(overview("15.0%", "0", "0"), MATMUL, "{}", String::new(), &[]), &[]);
    assert_eq!(result.at("status").str(), Some("NOT_HOST_BOUND"));
    assert_eq!(result.at("metrics").at("core_count"), &J::Int(1));
}

#[test]
fn test_get_utilization_metrics_short_circuits_on_error() {
    let fake = Fake::new(|tool, _| match tool {
        "overview_page.json" => Ok(Some(overview("15.0%", "8", "16").into_bytes())),
        "utilization_viewer.json" => Err(Error::new(Kind::Runtime, "RPC backend error")),
        _ => Ok(None),
    })
    .with_hosts(&[]);
    let result = check(&fake, &[]);
    assert_eq!(fake.fetched().iter().filter(|tool| *tool == "utilization_viewer.json").count(), 1);
    assert_eq!(metric(&result, "mxu_idleness_percent"), 0.0);
}

#[test]
fn test_get_utilization_metrics_short_circuits_on_session_no_data() {
    let fake = session(overview("15.0%", "8", "16"), MATMUL, "{}", String::new(), &[]);
    let result = check(&fake, &[]);
    assert_eq!(fake.fetched().iter().filter(|tool| *tool == "utilization_viewer.json").count(), 1);
    assert_eq!(metric(&result, "mxu_idleness_percent"), 0.0);
}

#[test]
fn test_get_utilization_metrics_falls_back_on_host_filter_miss() {
    let fake = session(overview("15.0%", "8", "16"), MATMUL, "{}", utilization(1, [85.0, 12.0, 4.0, 4.0]), &[]);
    let result = check(&fake, &[]);
    assert_eq!(fake.fetched().iter().filter(|tool| *tool == "utilization_viewer.json").count(), 2);
    assert_eq!(metric(&result, "mxu_idleness_percent"), 85.0);
    assert_eq!(metric(&result, "hbm_bandwidth_utilization_percent"), 12.0);
}
