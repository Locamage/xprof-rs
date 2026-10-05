use crate::tools::event_fractions::Fractions;
use crate::tools::smart_suggestion::{Data, InputPipeline, Node, Overview, RULES, Rule, Step, ToolData};
use std::collections::BTreeMap;

pub type StepInfo<'a> = (&'a [(u32, u64)], &'a [(&'a str, u64)]);

pub const TEST_ERROR: &str = "Test Error";

#[derive(Default)]
pub struct MockToolDataProvider {
    pub overview_page: Option<Overview>,
    pub input_pipeline_analysis: Option<InputPipeline>,
    pub event_time_fractions: Option<Fractions>,
    pub op_stats: Option<Vec<Step>>,
    pub op_profile: Option<Node>,
    pub memory_profile: Option<Vec<(f64, f64)>>,
}

fn returned<T: Clone>(value: &Option<T>) -> Data<T> {
    value.clone().ok_or_else(|| TEST_ERROR.to_string())
}

impl ToolData for MockToolDataProvider {
    fn overview(&self) -> Data<Overview> {
        returned(&self.overview_page)
    }

    fn input_pipeline(&self) -> Data<InputPipeline> {
        returned(&self.input_pipeline_analysis)
    }

    fn event_fractions(&self, _: &str) -> Data<Fractions> {
        returned(&self.event_time_fractions)
    }

    fn steps(&self) -> Data<Vec<Step>> {
        returned(&self.op_stats)
    }

    fn op_profile(&self) -> Data<Node> {
        returned(&self.op_profile)
    }

    fn memory(&self) -> Data<Vec<(f64, f64)>> {
        returned(&self.memory_profile)
    }
}

pub struct SmartSuggestion {
    pub rule_name: &'static str,
    pub suggestion_text: String,
}

pub fn rule(name: &str) -> &'static Rule {
    RULES.iter().find(|rule| rule.name == name).unwrap()
}

pub fn meets_conditions(name: &str, provider: &MockToolDataProvider) -> bool {
    (rule(name).meets)(provider)
}

pub fn generate_suggestion(name: &str, provider: &MockToolDataProvider) -> Data<String> {
    (rule(name).generate)(provider)
}

pub fn apply(name: &str, provider: &MockToolDataProvider) -> Data<Option<SmartSuggestion>> {
    let rule = rule(name);
    if !(rule.meets)(provider) {
        return Ok(None);
    }
    Ok(Some(SmartSuggestion { rule_name: rule.name, suggestion_text: (rule.generate)(provider)? }))
}

pub fn overview_page(mxu_utilization_percent: f64, memory_bw_utilization_relative_to_hw_limit_percent: f64) -> Option<Overview> {
    Some(Overview { mxu_percent: mxu_utilization_percent, hbm_percent: memory_bw_utilization_relative_to_hw_limit_percent })
}

pub fn input_pipeline_analysis(input_percent: f64, enqueue_us: f64, demanded_file_read_us: f64) -> Option<InputPipeline> {
    Some(InputPipeline { input_percent, enqueue_us, demanded_file_read_us, ..Default::default() })
}

pub fn tpu_step_time_breakdown(step_time_average: f64, tc_idle_ms_average: f64, sc_step_time_ms_average: f64) -> Option<InputPipeline> {
    Some(InputPipeline { step_time_ms: step_time_average, tpu: Some((tc_idle_ms_average, sc_step_time_ms_average)), ..Default::default() })
}

pub fn generic_step_time_breakdown(step_time_average: f64) -> Option<InputPipeline> {
    Some(InputPipeline { step_time_ms: step_time_average, tpu: None, ..Default::default() })
}

pub fn step_sequence(steps: &[StepInfo]) -> Option<Vec<Step>> {
    Some(
        steps.iter().map(|(cores, categories)| Step { cores: cores.to_vec(), categories: categories.iter().map(|(category, self_time_ps)| (category.to_string(), *self_time_ps)).collect() }).collect(),
    )
}

pub fn event_time_fractions(chips: &[(&str, &[f32])], hosts: &[(&str, &[f32])]) -> Option<Fractions> {
    let map = |entries: &[(&str, &[f32])]| entries.iter().map(|(name, fractions)| (name.to_string(), fractions.to_vec())).collect::<BTreeMap<_, _>>();
    Some(Fractions { chips: map(chips), hosts: map(hosts) })
}

pub fn op_profile_with_async_done(root_raw_time: f64, name: &str, raw_time: f64) -> Option<Node> {
    let leaf = Node { name: name.into(), raw_time, ..Default::default() };
    let async_done = Node { name: "async-done".into(), children: vec![leaf], ..Default::default() };
    Some(Node { raw_time: root_raw_time, children: vec![Node { children: vec![async_done], ..Default::default() }], ..Default::default() })
}

pub fn memory_profile(peak_bytes_in_use: f64, memory_capacity: f64) -> Option<Vec<(f64, f64)>> {
    Some(vec![(peak_bytes_in_use, memory_capacity)])
}
