use super::mock_tool_data_provider::{MockToolDataProvider, generate_suggestion, meets_conditions, step_sequence};
use crate::smart_suggestion::collective_percent;

const RULE: &str = "CollectiveBoundRule";

#[test]
fn mock_call_test() {
    let provider = MockToolDataProvider { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 5)])]), ..Default::default() };
    let result = collective_percent(&provider);
    assert!((result.unwrap() - 5.0).abs() <= 0.001);
}

#[test]
fn meets_conditions_above_threshold() {
    let provider = MockToolDataProvider { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 35)]), (&[(0, 100)], &[("all-reduce", 36)])]), ..Default::default() };
    assert!(meets_conditions(RULE, &provider));
}

#[test]
fn meets_conditions_below_threshold() {
    let provider = MockToolDataProvider { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 1)]), (&[(0, 100)], &[("all-reduce", 2)])]), ..Default::default() };
    assert!(!meets_conditions(RULE, &provider));
}

#[test]
fn meets_conditions_error() {
    assert!(!meets_conditions(RULE, &MockToolDataProvider::default()));
}

#[test]
fn generate_suggestion_error() {
    assert!(generate_suggestion(RULE, &MockToolDataProvider::default()).is_err());
}
