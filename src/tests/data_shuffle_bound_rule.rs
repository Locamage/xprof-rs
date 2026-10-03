use super::mock_tool_data_provider::{MockToolDataProvider, generate_suggestion, meets_conditions, step_sequence};

const RULE: &str = "DataShuffleBoundRule";

#[test]
fn meets_conditions_above_threshold() {
    let provider = MockToolDataProvider { op_stats: step_sequence(&[(&[(0, 100)], &[("shuffle", 24)]), (&[(0, 100)], &[("gather", 37)])]), ..Default::default() };
    assert!(meets_conditions(RULE, &provider));
}

#[test]
fn meets_conditions_below_threshold() {
    let provider = MockToolDataProvider { op_stats: step_sequence(&[(&[(0, 100)], &[("shuffle", 1)]), (&[(0, 100)], &[("gather", 2)])]), ..Default::default() };
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
