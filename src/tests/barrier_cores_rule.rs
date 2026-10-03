use super::mock_tool_data_provider::{MockToolDataProvider, apply, event_time_fractions};

#[test]
fn meets_conditions() {
    let provider = MockToolDataProvider { event_time_fractions: event_time_fractions(&[("chip0", &[0.15, 0.25]), ("chip1", &[0.05, 0.35])], &[]), ..Default::default() };
    let suggestion = apply("BarrierCoresRule", &provider).unwrap().unwrap();
    assert_eq!(suggestion.rule_name, "BarrierCoresRule");
    assert!(suggestion.suggestion_text.contains("20.0% of each step time"));
}

#[test]
fn not_special_op_bound() {
    let provider = MockToolDataProvider { event_time_fractions: event_time_fractions(&[("chip0", &[0.01, 0.02]), ("chip1", &[0.05, 0.25])], &[]), ..Default::default() };
    assert!(apply("BarrierCoresRule", &provider).unwrap().is_none());
}

#[test]
fn error_fetching_percentile() {
    let provider = MockToolDataProvider::default();
    assert!(apply("BarrierCoresRule", &provider).unwrap().is_none());
}

#[test]
fn meets_conditions_with_stragglers() {
    let chip0 = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.10, 0.10];
    let hosts: [(&str, &[f32]); 4] = [("host0", &[0.50, 0.50]), ("host1", &[0.50, 0.50]), ("host2", &[0.50, 0.50]), ("host3", &[0.10, 0.10])];
    let provider = MockToolDataProvider { event_time_fractions: event_time_fractions(&[("chip0", &chip0)], &hosts), ..Default::default() };
    let suggestion = apply("BarrierCoresRule", &provider).unwrap().unwrap();
    assert!(suggestion.suggestion_text.contains("<li>Host <b>host3</b> average barrier-cores time fraction: <b>10.0%</b></li>"));
    assert!(suggestion.suggestion_text.contains("Investigate Stragglers"));
}
