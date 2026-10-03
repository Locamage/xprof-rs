use super::mock_tool_data_provider::{MockToolDataProvider, apply, overview_page};

const RULE: &str = "MemoryBoundRule";

#[test]
fn meets_conditions() {
    let provider = MockToolDataProvider { overview_page: overview_page(49.0, 71.0), ..Default::default() };
    let suggestion = apply(RULE, &provider).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, RULE);
    assert!(suggestion.suggestion_text.contains("71.0%</b> and low MXU utilization of <b>49.0%"));
}

#[test]
fn hbm_utilization_too_low() {
    let provider = MockToolDataProvider { overview_page: overview_page(49.0, 69.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}

#[test]
fn mxu_utilization_too_high() {
    let provider = MockToolDataProvider { overview_page: overview_page(51.0, 71.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}
