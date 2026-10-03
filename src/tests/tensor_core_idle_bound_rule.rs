use super::mock_tool_data_provider::{MockToolDataProvider, apply, generic_step_time_breakdown, overview_page, tpu_step_time_breakdown};

const RULE: &str = "TensorCoreIdleBoundRule";

#[test]
fn meets_conditions() {
    let provider = MockToolDataProvider { input_pipeline_analysis: tpu_step_time_breakdown(100.0, 11.0, 0.0), overview_page: overview_page(49.0, 49.0), ..Default::default() };
    let suggestion = apply(RULE, &provider).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, RULE);
    assert!(suggestion.suggestion_text.contains("High TensorCore idle time percentage of <b>11.0%</b>"));
}

#[test]
fn idle_time_too_low() {
    let provider = MockToolDataProvider { input_pipeline_analysis: tpu_step_time_breakdown(100.0, 9.0, 0.0), overview_page: overview_page(49.0, 49.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}

#[test]
fn no_tpu_step_time_breakdown_field() {
    let provider = MockToolDataProvider { input_pipeline_analysis: generic_step_time_breakdown(100.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}

#[test]
fn hbm_and_mxu_utilization_too_high() {
    let provider = MockToolDataProvider { input_pipeline_analysis: tpu_step_time_breakdown(100.0, 11.0, 0.0), overview_page: overview_page(51.0, 51.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}
