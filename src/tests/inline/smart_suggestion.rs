use super::*;
use std::collections::BTreeMap;

#[derive(Default)]
struct Mock {
    overview: Option<Overview>,
    input: Option<InputPipeline>,
    fractions: Option<Fractions>,
    steps: Option<Vec<Step>>,
    profile: Option<Node>,
    memory: Option<Vec<(f64, f64)>>,
}

fn missing<T: Clone>(value: &Option<T>) -> Data<T> {
    value.clone().ok_or_else(|| "Test Error".to_string())
}

impl ToolData for Mock {
    fn overview(&self) -> Data<Overview> {
        missing(&self.overview)
    }

    fn input_pipeline(&self) -> Data<InputPipeline> {
        missing(&self.input)
    }

    fn event_fractions(&self, _: &str) -> Data<Fractions> {
        missing(&self.fractions)
    }

    fn steps(&self) -> Data<Vec<Step>> {
        missing(&self.steps)
    }

    fn op_profile(&self) -> Data<Node> {
        missing(&self.profile)
    }

    fn memory(&self) -> Data<Vec<(f64, f64)>> {
        missing(&self.memory)
    }
}

fn apply(name: &str, data: &Mock) -> Data<Option<String>> {
    let rule = RULES.iter().find(|rule| rule.name == name).unwrap();
    if (rule.meets)(data) { (rule.generate)(data).map(Some) } else { Ok(None) }
}

fn chips(values: &[(&str, &[f32])]) -> BTreeMap<String, Vec<f32>> {
    values.iter().map(|(name, fractions)| (name.to_string(), fractions.to_vec())).collect()
}

fn input(input_percent: f64, enqueue_us: f64, demanded_file_read_us: f64) -> Mock {
    Mock { input: Some(InputPipeline { input_percent, enqueue_us, demanded_file_read_us, ..Default::default() }), ..Default::default() }
}

#[test]
fn input_bound_rule_reports_the_input_percent() {
    assert!(apply("InputBoundRule", &input(20.0, 0.0, 0.0)).unwrap().unwrap().contains("<b>20.0% of step time</b>"));
}

#[test]
fn debug_print_rule_lists_hosts_over_the_threshold() {
    let data = Mock { fractions: Some(Fractions { hosts: chips(&[("a", &[0.06]), ("b", &[0.01])]), ..Default::default() }), ..Default::default() };
    let text = apply("DebugPrintRule", &data).unwrap().unwrap();
    assert!(text.contains("<b> up to 6.0% of step time</b>") && text.contains("<li>Host <b>a</b> average debug_print time fraction: <b>6.0%</b></li>") && !text.contains("<b>b</b>"));
}

#[test]
fn third_party_engine_runs_only_the_barrier_rule_and_renders_protobuf_json() {
    let data = Mock { fractions: Some(Fractions { chips: chips(&[("chip", &[0.5])]), ..Default::default() }), overview: Some(Overview { mxu_percent: 90.0, hbm_percent: 1.0 }), ..Default::default() };
    let suggestions = run(&data, &THIRD_PARTY_RULES).unwrap();
    assert_eq!(suggestions.iter().map(|suggestion| suggestion.rule).collect::<Vec<_>>(), ["BarrierCoresRule"]);
    assert!(render(&suggestions).starts_with("{\"suggestions\":[{\"ruleName\":\"BarrierCoresRule\",\"suggestionText\":\"\\u003cp\\u003eYour program"));
    assert_eq!(render(&[]), "{\"suggestions\":[]}");
}
