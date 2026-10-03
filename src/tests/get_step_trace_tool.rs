use super::cli_support::{Fake, args, json};
use crate::cli::json::J;
use crate::cli::steps::get_step_trace;
use crate::cli::{Error, Kind};

const INPUT_PIPELINE: &str = r#"[{"cols": [{"id": "stepnum", "type": "string"}, {"id": "noninfeedTimeMs", "type": "number"}, {"id": "infeedTimeMs", "type": "number"}, {"id": "tooltip", "type": "string"}, {"id": "infeedPercentAverage", "type": "number"}], "rows": [{"c": [{"v": "1"}, {"v": 20.0}, {"v": 80.0}, {"v": "tooltip 1"}, {"v": 80.0}]}, {"c": [{"v": "2"}, {"v": 30.0}, {"v": 70.0}, {"v": "tooltip 2"}, {"v": 70.0}]}], "p": {"steptime_ms_average": "100.0", "steptime_ms_minimum": "100.0", "steptime_ms_maximum": "100.0", "steptime_ms_standard_deviation": "0.0", "infeed_percent_average": "75.0", "summary_conclusion": "Program is input-bound"}}]"#;
const OVERVIEW: &str =
    r#"[{"p": {"steptime_ms_average": "50.0", "steptime_ms_standard_deviation": "2.5", "tc_infeed_ms_average": "5.0", "tc_outfeed_ms_average": "1.0", "tc_idle_ms_average": "4.0"}}]"#;
const TWO_STEPS: &str = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 221, "podStatsPerCore": {"0": {"chipId": 0, "nodeId": 0, "hostName": "host1", "stepNum": 221, "totalDurationUs": 300000.0, "highFlopsComputeUs": 150000.0, "crsDurationUs": 50000.0, "sendDurationUs": 20000.0, "recvDurationUs": 30000.0, "hostInfeedDurationUs": 10000.0, "hostOutfeedDurationUs": 5000.0, "bottleneck": "Send and Recv"}, "1": {"chipId": 0, "nodeId": 1, "hostName": "host1", "stepNum": 221, "totalDurationUs": 300000.0, "highFlopsComputeUs": 150000.0, "crsDurationUs": 50000.0, "sendDurationUs": 20000.0, "recvDurationUs": 30000.0, "hostInfeedDurationUs": 10000.0, "hostOutfeedDurationUs": 5000.0, "bottleneck": "Send and Recv"}}}, {"stepNum": 222, "podStatsPerCore": {"0": {"chipId": 0, "nodeId": 0, "hostName": "host1", "stepNum": 222, "totalDurationUs": 400000.0, "highFlopsComputeUs": 200000.0, "crsDurationUs": 60000.0, "sendDurationUs": 30000.0, "recvDurationUs": 40000.0, "hostInfeedDurationUs": 20000.0, "hostOutfeedDurationUs": 10000.0, "bottleneck": "All-Reduce"}}}]}}"#;
const ONE_CORE: &str = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 1, "podStatsPerCore": {"0": {"totalDurationUs": 100000.0, "highFlopsComputeUs": 90000.0}}}]}}"#;

fn near(value: &J, expected: f64, places: i32) {
    let actual = value.float().unwrap_or_else(|| panic!("not a number: {value:?}"));
    assert!((actual - expected).abs() < 0.5 * 10f64.powi(-places), "{actual} != {expected}");
}

fn input_pipeline_fallback(tool: &str) -> Option<String> {
    (tool == "input_pipeline.json").then(|| INPUT_PIPELINE.to_string())
}

fn overview_fallback(tool: &str) -> Option<String> {
    (tool == "overview_page.json").then(|| OVERVIEW.to_string())
}

fn step_trace(fake: &Fake, flags: &[(&str, J)]) -> J {
    json(get_step_trace(fake, &args("test_session", flags)))
}

#[test]
fn test_get_step_trace_from_pod_viewer_success() {
    let result = step_trace(&Fake::fixed(TWO_STEPS), &[]);
    assert!(!result.has("error"));
    let summary = result.at("summary");
    assert_eq!(summary.at("total_steps"), &J::Int(2));
    assert_eq!(summary.at("is_aggregate"), &J::Bool(false));
    near(summary.at("step_time_ms_average"), 350.0, 2);
    near(summary.at("step_time_ms_min"), 300.0, 2);
    near(summary.at("step_time_ms_max"), 400.0, 2);
    near(summary.at("compute_time_ms_average"), 175.0, 2);
    near(summary.at("communication_time_ms_average"), 115.0, 2);
    near(summary.at("infeed_time_ms_average"), 15.0, 2);
    near(summary.at("outfeed_time_ms_average"), 7.5, 2);
    let steps = result.at("step_breakdown").items();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].at("step_num"), &J::Int(221));
    near(steps[0].at("step_time_ms"), 300.0, 2);
    near(steps[0].at("compute_time_ms"), 150.0, 2);
    near(steps[0].at("communication_time_ms"), 100.0, 2);
    let breakdown = steps[0].at("communication_breakdown_ms");
    assert_eq!(breakdown.at("all_reduce_ms"), &J::Float(50.0));
    assert_eq!(breakdown.at("send_ms"), &J::Float(20.0));
    assert_eq!(breakdown.at("recv_ms"), &J::Float(30.0));
    assert_eq!(steps[0].at("bottleneck").str(), Some("Send and Recv"));
}

#[test]
fn test_get_step_trace_step_num_filter() {
    let data = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 10, "podStatsPerCore": {"0": {"totalDurationUs": 100000.0, "highFlopsComputeUs": 80000.0, "crsDurationUs": 10000.0, "sendDurationUs": 5000.0, "recvDurationUs": 5000.0, "hostInfeedDurationUs": 0.0, "hostOutfeedDurationUs": 0.0}}}, {"stepNum": 11, "podStatsPerCore": {"0": {"totalDurationUs": 200000.0, "highFlopsComputeUs": 150000.0, "crsDurationUs": 20000.0, "sendDurationUs": 15000.0, "recvDurationUs": 15000.0, "hostInfeedDurationUs": 0.0, "hostOutfeedDurationUs": 0.0}}}]}}"#;
    let result = step_trace(&Fake::fixed(data), &[("step_num", J::Int(11))]);
    assert!(!result.has("error"));
    let steps = result.at("step_breakdown").items();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].at("step_num"), &J::Int(11));
    near(steps[0].at("step_time_ms"), 200.0, 2);
}

#[test]
fn test_get_step_trace_limit() {
    let entries: Vec<String> = (0..5).map(|step| format!(r#"{{"stepNum": {step}, "podStatsPerCore": {{"0": {{"totalDurationUs": 100000.0, "highFlopsComputeUs": 80000.0}}}}}}"#)).collect();
    let data = format!(r#"{{"podStatsSequence": {{"podStatsMap": [{}]}}}}"#, entries.join(", "));
    let result = step_trace(&Fake::fixed(&data), &[("limit", J::Int(2))]);
    assert_eq!(result.at("summary").at("total_steps"), &J::Int(5));
    assert_eq!(result.at("step_breakdown").items().len(), 2);
}

#[test]
fn test_get_step_trace_device_core_filter() {
    let data = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 1, "podStatsPerCore": {"0": {"totalDurationUs": 100000.0, "highFlopsComputeUs": 90000.0}, "1": {"totalDurationUs": 200000.0, "highFlopsComputeUs": 180000.0}}}]}}"#;
    let result = step_trace(&Fake::fixed(data), &[("device_core", J::Int(1))]);
    near(result.at("step_breakdown").items()[0].at("step_time_ms"), 200.0, 2);
}

#[test]
fn test_get_step_trace_device_core_not_found() {
    let result = step_trace(&Fake::fixed(ONE_CORE), &[("device_core", J::Int(999))]);
    assert_eq!(result.at("status").str(), Some("NO_DATA"));
}

#[test]
fn test_get_step_trace_device_core_not_found_fallback() {
    let fake = Fake::tools(|tool| if tool == "pod_viewer.json" { Some(ONE_CORE.into()) } else { input_pipeline_fallback(tool) });
    let result = step_trace(&fake, &[("device_core", J::Int(999))]);
    assert!(!result.has("status"));
    assert!(result.has("summary") && result.has("step_breakdown"));
    assert_eq!(result.at("summary").at("primary_bottleneck").str(), Some("Input / Infeed"));
}

#[test]
fn test_get_step_trace_include_summary_false() {
    let result = step_trace(&Fake::fixed(ONE_CORE), &[("include_summary", J::Bool(false))]);
    assert!(!result.has("summary"));
    assert!(result.has("step_breakdown"));
}

#[test]
fn test_get_step_trace_from_input_pipeline_fallback() {
    let result = step_trace(&Fake::tools(input_pipeline_fallback), &[]);
    assert!(!result.has("error"));
    let summary = result.at("summary");
    assert_eq!(summary.at("is_aggregate"), &J::Bool(false));
    assert_eq!(summary.at("primary_bottleneck").str(), Some("Input / Infeed"));
    assert_eq!(summary.at("conclusion").str(), Some("Program is input-bound"));
    let steps = result.at("step_breakdown").items();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].at("step_num"), &J::Int(1));
    near(steps[0].at("compute_time_ms"), 20.0, 2);
    near(steps[0].at("infeed_time_ms"), 80.0, 2);
}

#[test]
fn test_get_step_trace_from_overview_page_fallback() {
    let result = step_trace(&Fake::tools(overview_fallback), &[]);
    assert!(!result.has("error"));
    let summary = result.at("summary");
    assert_eq!(summary.get("total_steps"), Some(&J::Null));
    assert_eq!(summary.at("is_aggregate"), &J::Bool(true));
    assert!(summary.at("note").text().contains("step count not available"));
    near(summary.at("step_time_ms_average"), 50.0, 2);
    near(summary.at("compute_time_ms_average"), 40.0, 2);
}

#[test]
fn test_get_step_trace_from_overview_page_fallback_with_step_rows() {
    let data = r#"[{"cols": [{"id": "stepnum"}, {"id": "computeTimeMs"}], "rows": [{"c": [{"v": "1"}, {"v": 10.0}]}, {"c": [{"v": "2"}, {"v": 12.0}]}, {"c": [{"v": "3"}, {"v": 11.0}]}]}, {"p": {"steptime_ms_average": "50.0", "tc_infeed_ms_average": "5.0"}}]"#;
    let result = step_trace(&Fake::tools(move |tool| (tool == "overview_page.json").then(|| data.to_string())), &[]);
    assert!(!result.has("error"));
    let summary = result.at("summary");
    assert_eq!(summary.at("total_steps"), &J::Int(3));
    assert_eq!(summary.at("is_aggregate"), &J::Bool(true));
    assert!(!summary.at("note").text().contains("step count not available"));
}

#[test]
fn test_get_step_trace_from_overview_page_fallback_with_total_steps_prop() {
    let data = r#"[{"p": {"steptime_ms_average": "50.0", "total_steps": "42"}}]"#;
    let result = step_trace(&Fake::tools(move |tool| (tool == "overview_page.json").then(|| data.to_string())), &[]);
    assert_eq!(result.at("summary").at("total_steps"), &J::Int(42));
    assert_eq!(result.at("summary").at("is_aggregate"), &J::Bool(true));
}

#[test]
fn test_get_step_trace_no_data() {
    let result = step_trace(&Fake::fixed(""), &[]);
    assert_eq!(result.at("status").str(), Some("NO_DATA"));
    assert!(result.at("message").text().contains("No step trace data"));
}

#[test]
fn test_get_step_trace_exception_handled() {
    let result = step_trace(&Fake::new(|_, _| Err(Error::new(Kind::Runtime, "Backend unavailable"))), &[]);
    assert_eq!(result.at("status").str(), Some("NO_DATA"));
    assert!(result.at("message").text().contains("No step trace data"));
}

#[test]
fn test_get_step_trace_bypass_cache() {
    let fake = Fake::fixed("");
    let result = step_trace(&fake, &[("bypass_cache", J::Bool(true))]);
    assert_eq!(result.at("status").str(), Some("NO_DATA"));
    let expected = ("pod_viewer.json".to_string(), vec![("format".to_string(), "json".to_string()), ("bypass_cache".to_string(), "True".to_string())]);
    assert!(fake.calls.borrow().contains(&expected));
}

#[test]
fn test_get_step_trace_file_not_found() {
    let fake = Fake::new(|_, _| Err(Error::new(Kind::FileNotFound, "Trace path not found")));
    let error = get_step_trace(&fake, &args("non_existent_session", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_get_step_trace_from_pod_viewer_step_breakdown_map() {
    let data = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 7, "podStatsPerCore": {"0": {"chipId": 0, "nodeId": 0, "stepNum": 7, "totalDurationUs": 200000.0, "stepBreakdownUs": {"1": 150000.0, "2": 5000.0, "3": 25000.0, "6": 12000.0, "7": 8000.0}}}}]}}"#;
    let result = step_trace(&Fake::fixed(data), &[]);
    let step = &result.at("step_breakdown").items()[0];
    assert_eq!(step.at("step_num"), &J::Int(7));
    near(step.at("step_time_ms"), 200.0, 2);
    near(step.at("compute_time_ms"), 150.0, 2);
    near(step.at("compute_percent"), 75.0, 2);
    near(step.at("communication_time_ms"), 30.0, 2);
    near(step.at("infeed_time_ms"), 12.0, 2);
    near(step.at("outfeed_time_ms"), 8.0, 2);
    assert_eq!(step.at("bottleneck").str(), Some("Compute"));
}

#[test]
fn test_get_step_trace_pod_viewer_empty_breakdown_falls_back() {
    let pod = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 0, "podStatsPerCore": {"0": {"totalDurationUs": 234523.37, "bottleneck": "Output", "stepBreakdownUs": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}}}}]}}"#;
    let pipeline = r#"[{"cols": [{"id": "stepnum", "type": "string"}, {"id": "tcComputeTimeMs", "type": "number"}, {"id": "tcInfeedTimeMs", "type": "number"}, {"id": "tcOutfeedTimeMs", "type": "number"}, {"id": "tcIdleTimeMs", "type": "number"}, {"id": "hostTransferTimeMs", "type": "number"}, {"id": "infeedPercentAverage", "type": "number"}], "rows": [{"c": [{"v": "0"}, {"v": 231.9486925}, {"v": 0.0}, {"v": 0.0}, {"v": 2.5746775}, {"v": 0.0}, {"v": 0.0}]}]}]"#;
    let fake = Fake::tools(move |tool| match tool {
        "pod_viewer.json" | "pod_viewer" => Some(pod.into()),
        "input_pipeline_analyzer" => Some(pipeline.into()),
        _ => None,
    });
    let summary = step_trace(&fake, &[]).at("summary").clone();
    near(summary.at("step_time_ms_average"), 234.5234, 2);
    near(summary.at("compute_time_ms_average"), 231.9487, 2);
    assert!(summary.at("compute_percent").float().unwrap() > 95.0);
    assert_eq!(summary.at("primary_bottleneck").str(), Some("Compute"));
}

#[test]
fn test_get_step_trace_input_pipeline_unknown_tool_name_falls_back() {
    let fake = Fake::new(|tool, _| match tool {
        "pod_viewer.json" | "pod_viewer" => Ok(None),
        "input_pipeline_analyzer" => Err(Error::new(Kind::Value, format!("Unknown XProf tool name: '{tool}'"))),
        _ => Ok(overview_fallback(tool).map(String::into_bytes)),
    });
    let result = step_trace(&fake, &[]);
    assert!(!result.has("error"));
    assert_eq!(result.at("summary").at("is_aggregate"), &J::Bool(true));
    near(result.at("summary").at("step_time_ms_average"), 50.0, 2);
}

#[test]
fn test_get_step_trace_pod_viewer_reports_unattributed_time_as_idle() {
    let data = r#"{"podStatsSequence": {"podStatsMap": [{"stepNum": 3, "podStatsPerCore": {"0": {"chipId": 0, "nodeId": 0, "stepNum": 3, "totalDurationUs": 100000.0, "stepBreakdownUs": {"1": 80000.0, "6": 10000.0, "4": 4000.0}}}}]}}"#;
    let result = step_trace(&Fake::fixed(data), &[]);
    let step = &result.at("step_breakdown").items()[0];
    near(step.at("step_time_ms"), 100.0, 2);
    near(step.at("compute_time_ms"), 80.0, 2);
    near(step.at("idle_time_ms"), 10.0, 2);
    near(step.at("idle_percent"), 10.0, 2);
    let covered: f64 = ["compute_time_ms", "communication_time_ms", "infeed_time_ms", "outfeed_time_ms", "idle_time_ms"].iter().map(|key| step.at(key).float().unwrap()).sum();
    near(&J::Float(covered), step.at("step_time_ms").float().unwrap(), 2);
    near(result.at("summary").at("idle_time_ms_average"), 10.0, 2);
    near(result.at("summary").at("idle_percent"), 10.0, 2);
}

#[test]
fn test_get_step_trace_input_pipeline_reports_tc_idle_time() {
    let data = r#"[{}, {"cols": [{"id": "stepnum"}, {"id": "tcComputeTimeMs"}, {"id": "tcInfeedTimeMs"}, {"id": "tcOutfeedTimeMs"}, {"id": "tcIdleTimeMs"}], "rows": [{"c": [{"v": "0"}, {"v": 231.9486925}, {"v": 0.0}, {"v": 0.0}, {"v": 2.5746775}]}]}]"#;
    let fake = Fake::tools(move |tool| (!matches!(tool, "pod_viewer.json" | "pod_viewer")).then(|| data.to_string()));
    let step = step_trace(&fake, &[]).at("step_breakdown").items()[0].clone();
    near(step.at("step_time_ms"), 234.5234, 3);
    near(step.at("compute_time_ms"), 231.9487, 3);
    near(step.at("idle_time_ms"), 2.5747, 3);
    near(step.at("compute_percent"), 98.9, 1);
    near(step.at("idle_percent"), 1.1, 1);
    assert_eq!(step.at("bottleneck").str(), Some("Compute"));
}
