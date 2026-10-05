use super::mock_tool_data_provider::{
    MockToolDataProvider as P, apply, event_time_fractions, generate_suggestion, generic_step_time_breakdown, input_pipeline_analysis, meets_conditions, memory_profile, op_profile_with_async_done,
    overview_page, step_sequence, tpu_step_time_breakdown,
};

#[derive(Clone, Copy)]
enum Expect {
    Text(&'static [&'static str]),
    Nothing,
    Meets(bool),
    GenerateErr,
}
use Expect::*;

fn check(rule: &str, provider: &P, expect: Expect) {
    match expect {
        Text(texts) => {
            let suggestion = apply(rule, provider).unwrap().unwrap();
            assert_eq!(suggestion.rule_name, rule);
            for text in texts {
                assert!(suggestion.suggestion_text.contains(text), "{rule}: {text}");
            }
        }
        Nothing => assert!(apply(rule, provider).unwrap().is_none(), "{rule}"),
        Meets(meets) => assert_eq!(meets_conditions(rule, provider), meets, "{rule}"),
        GenerateErr => assert!(generate_suggestion(rule, provider).is_err(), "{rule}"),
    }
}

macro_rules! rules { ($($module:ident = $rule:literal { $($name:ident: $provider:expr => $expect:expr,)* } $({ $($extra:item)* })?)*) => { $(mod $module { use super::*; $(#[test] fn $name() { check($rule, &$provider, $expect) })* $($($extra)*)? })* } }

fn offload(all_reduce_raw_time: f64, peak_bytes_in_use: f64) -> P {
    P { op_profile: op_profile_with_async_done(1000.0, "all-reduce.0", all_reduce_raw_time), memory_profile: memory_profile(peak_bytes_in_use, 200.0), ..Default::default() }
}

fn tpu(tc_idle: f64, sc_step: f64, utilization: f64) -> P {
    P { input_pipeline_analysis: tpu_step_time_breakdown(100.0, tc_idle, sc_step), overview_page: overview_page(utilization, utilization), ..Default::default() }
}

rules! {
    barrier_cores_rule = "BarrierCoresRule" {
        meets_conditions: P { event_time_fractions: event_time_fractions(&[("chip0", &[0.15, 0.25]), ("chip1", &[0.05, 0.35])], &[]), ..Default::default() } => Text(&["20.0% of each step time"]),
        not_special_op_bound: P { event_time_fractions: event_time_fractions(&[("chip0", &[0.01, 0.02]), ("chip1", &[0.05, 0.25])], &[]), ..Default::default() } => Nothing,
        error_fetching_percentile: P::default() => Nothing,
        meets_conditions_with_stragglers: P {
            event_time_fractions: event_time_fractions(&[("chip0", &[0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.10, 0.10])], &[("host0", &[0.50, 0.50]), ("host1", &[0.50, 0.50]), ("host2", &[0.50, 0.50]), ("host3", &[0.10, 0.10])]),
            ..Default::default()
        } => Text(&["<li>Host <b>host3</b> average barrier-cores time fraction: <b>10.0%</b></li>", "Investigate Stragglers"]),
    }
    collective_bound_rule = "CollectiveBoundRule" {
        meets_conditions_above_threshold: P { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 35)]), (&[(0, 100)], &[("all-reduce", 36)])]), ..Default::default() } => Meets(true),
        meets_conditions_below_threshold: P { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 1)]), (&[(0, 100)], &[("all-reduce", 2)])]), ..Default::default() } => Meets(false),
        meets_conditions_error: P::default() => Meets(false),
        generate_suggestion_error: P::default() => GenerateErr,
    } {
        #[test]
        fn mock_call_test() {
            let provider = P { op_stats: step_sequence(&[(&[(0, 100)], &[("all-reduce", 5)])]), ..Default::default() };
            assert!((crate::tools::smart_suggestion::collective_percent(&provider).unwrap() - 5.0).abs() <= 0.001);
        }
    }
    compute_bound_rule = "ComputeBoundRule" {
        meets_conditions: P { overview_page: overview_page(71.0, 49.0), ..Default::default() } => Text(&["71.0%</b> and low HBM Bandwidth utilization of <b>49.0%"]),
        mxu_utilization_too_low: P { overview_page: overview_page(69.0, 49.0), ..Default::default() } => Nothing,
        hbm_utilization_too_high: P { overview_page: overview_page(71.0, 51.0), ..Default::default() } => Nothing,
    }
    data_shuffle_bound_rule = "DataShuffleBoundRule" {
        meets_conditions_above_threshold: P { op_stats: step_sequence(&[(&[(0, 100)], &[("shuffle", 24)]), (&[(0, 100)], &[("gather", 37)])]), ..Default::default() } => Meets(true),
        meets_conditions_below_threshold: P { op_stats: step_sequence(&[(&[(0, 100)], &[("shuffle", 1)]), (&[(0, 100)], &[("gather", 2)])]), ..Default::default() } => Meets(false),
        meets_conditions_error: P::default() => Meets(false),
        generate_suggestion_error: P::default() => GenerateErr,
    }
    data_transfer_bound_rule = "DataTransferBoundRule" {
        meets_conditions: P { input_pipeline_analysis: input_pipeline_analysis(20.0, 40.0, 10.0), ..Default::default() } => Text(&["16.0% of the total step time"]),
        not_input_bound: P { input_pipeline_analysis: input_pipeline_analysis(5.0, 40.0, 10.0), ..Default::default() } => Nothing,
        input_bound_but_not_data_transfer_bound: P { input_pipeline_analysis: input_pipeline_analysis(20.0, 10.0, 90.0), ..Default::default() } => Nothing,
    }
    host_processing_bound_rule = "HostProcessingBoundRule" {
        meets_conditions: P { input_pipeline_analysis: input_pipeline_analysis(20.0, 10.0, 90.0), ..Default::default() } => Text(&["18.0% of the total step time"]),
        not_input_bound: P { input_pipeline_analysis: input_pipeline_analysis(5.0, 10.0, 90.0), ..Default::default() } => Nothing,
        input_bound_but_not_host_processing_bound: P { input_pipeline_analysis: input_pipeline_analysis(20.0, 90.0, 10.0), ..Default::default() } => Nothing,
    }
    memory_bound_rule = "MemoryBoundRule" {
        meets_conditions: P { overview_page: overview_page(49.0, 71.0), ..Default::default() } => Text(&["71.0%</b> and low MXU utilization of <b>49.0%"]),
        hbm_utilization_too_low: P { overview_page: overview_page(49.0, 69.0), ..Default::default() } => Nothing,
        mxu_utilization_too_high: P { overview_page: overview_page(51.0, 71.0), ..Default::default() } => Nothing,
    }
    sparse_core_bound_rule = "SparseCoreBoundRule" {
        meets_conditions: tpu(0.0, 11.0, 49.0) => Text(&["11.0% of the total step time </b> is spent on SparseCore"]),
        idle_time_too_low: tpu(0.0, 9.0, 49.0) => Nothing,
        no_tpu_step_time_breakdown_field: P { input_pipeline_analysis: generic_step_time_breakdown(100.0), ..Default::default() } => Nothing,
        hbm_and_mxu_utilization_too_high: tpu(0.0, 11.0, 51.0) => Nothing,
    }
    sparse_core_offload_rule = "SparseCoreOffloadRule" {
        meets_conditions: offload(110.0, 80.0) => Meets(true),
        async_done_too_low: offload(90.0, 80.0) => Meets(false),
        memory_utilization_too_high: offload(110.0, 120.0) => Meets(false),
        generate_suggestion: offload(110.0, 80.0) => Text(&["11.0%</b> of time is spent on async-done operations with low memory utilization of <b>40.0%</b>"]),
    }
    tensor_core_idle_bound_rule = "TensorCoreIdleBoundRule" {
        meets_conditions: tpu(11.0, 0.0, 49.0) => Text(&["High TensorCore idle time percentage of <b>11.0%</b>"]),
        idle_time_too_low: tpu(9.0, 0.0, 49.0) => Nothing,
        no_tpu_step_time_breakdown_field: P { input_pipeline_analysis: generic_step_time_breakdown(100.0), ..Default::default() } => Nothing,
        hbm_and_mxu_utilization_too_high: tpu(11.0, 0.0, 51.0) => Nothing,
    }
}
