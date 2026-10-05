use super::*;
use crate::tests::mock_tool_data_provider::{MockToolDataProvider as Mock, apply, event_time_fractions, input_pipeline_analysis, overview_page};

#[test]
fn input_bound_rule_reports_the_input_percent() {
    let data = Mock { input_pipeline_analysis: input_pipeline_analysis(20.0, 0.0, 0.0), ..Default::default() };
    assert!(apply("InputBoundRule", &data).unwrap().unwrap().suggestion_text.contains("<b>20.0% of step time</b>"));
}

#[test]
fn debug_print_rule_lists_hosts_over_the_threshold() {
    let data = Mock { event_time_fractions: event_time_fractions(&[], &[("a", &[0.06]), ("b", &[0.01])]), ..Default::default() };
    let text = apply("DebugPrintRule", &data).unwrap().unwrap().suggestion_text;
    assert!(text.contains("<b> up to 6.0% of step time</b>") && text.contains("<li>Host <b>a</b> average debug_print time fraction: <b>6.0%</b></li>") && !text.contains("<b>b</b>"));
}

#[test]
fn third_party_engine_runs_only_the_barrier_rule_and_renders_protobuf_json() {
    let data = Mock { event_time_fractions: event_time_fractions(&[("chip", &[0.5])], &[]), overview_page: overview_page(90.0, 1.0), ..Default::default() };
    let suggestions = run(&data, &THIRD_PARTY_RULES).unwrap();
    assert_eq!(suggestions.iter().map(|suggestion| suggestion.rule).collect::<Vec<_>>(), ["BarrierCoresRule"]);
    assert!(render(&suggestions).starts_with("{\"suggestions\":[{\"ruleName\":\"BarrierCoresRule\",\"suggestionText\":\"\\u003cp\\u003eYour program"));
    assert_eq!(render(&[]), "{\"suggestions\":[]}");
}
