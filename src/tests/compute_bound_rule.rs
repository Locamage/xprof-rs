use super::mock_tool_data_provider::{MockToolDataProvider, apply, overview_page};

const RULE: &str = "ComputeBoundRule";

#[test]
fn meets_conditions() {
    let provider = MockToolDataProvider { overview_page: overview_page(71.0, 49.0), ..Default::default() };
    let suggestion = apply(RULE, &provider).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, RULE);
    assert!(suggestion.suggestion_text.contains("71.0%</b> and low HBM Bandwidth utilization of <b>49.0%"));
}

#[test]
fn mxu_utilization_too_low() {
    let provider = MockToolDataProvider { overview_page: overview_page(69.0, 49.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}

#[test]
fn hbm_utilization_too_high() {
    let provider = MockToolDataProvider { overview_page: overview_page(71.0, 51.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}
