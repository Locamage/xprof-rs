use super::mock_tool_data_provider::{self as mock, MockToolDataProvider, apply, memory_profile, op_profile_with_async_done};

const RULE: &str = "SparseCoreOffloadRule";

fn provider(all_reduce_raw_time: f64, peak_bytes_in_use: f64) -> MockToolDataProvider {
    MockToolDataProvider { op_profile: op_profile_with_async_done(1000.0, "all-reduce.0", all_reduce_raw_time), memory_profile: memory_profile(peak_bytes_in_use, 200.0), ..Default::default() }
}

#[test]
fn meets_conditions() {
    assert!(mock::meets_conditions(RULE, &provider(110.0, 80.0)));
}

#[test]
fn async_done_too_low() {
    assert!(!mock::meets_conditions(RULE, &provider(90.0, 80.0)));
}

#[test]
fn memory_utilization_too_high() {
    assert!(!mock::meets_conditions(RULE, &provider(110.0, 120.0)));
}

#[test]
fn generate_suggestion() {
    let suggestion = apply(RULE, &provider(110.0, 80.0)).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, RULE);
    assert!(suggestion.suggestion_text.contains("11.0%</b> of time is spent on async-done operations with low memory utilization of <b>40.0%</b>"));
}
