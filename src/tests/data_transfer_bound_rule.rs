use super::mock_tool_data_provider::{MockToolDataProvider, apply, input_pipeline_analysis};

const RULE: &str = "DataTransferBoundRule";

#[test]
fn meets_conditions() {
    let provider = MockToolDataProvider { input_pipeline_analysis: input_pipeline_analysis(20.0, 40.0, 10.0), ..Default::default() };
    let suggestion = apply(RULE, &provider).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, RULE);
    assert!(suggestion.suggestion_text.contains("16.0% of the total step time"));
}

#[test]
fn not_input_bound() {
    let provider = MockToolDataProvider { input_pipeline_analysis: input_pipeline_analysis(5.0, 40.0, 10.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}

#[test]
fn input_bound_but_not_data_transfer_bound() {
    let provider = MockToolDataProvider { input_pipeline_analysis: input_pipeline_analysis(20.0, 10.0, 90.0), ..Default::default() };
    assert!(apply(RULE, &provider).unwrap().is_none());
}
