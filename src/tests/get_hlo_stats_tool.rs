use super::cli_support::{Fake, args, json};
use crate::cli::json::J;
use crate::cli::ops::get_hlo_stats;
use crate::cli::{Error, Kind};

const DATABASE: &str = r#"{"cols": [{"id": "rank"}, {"id": "program_id"}, {"id": "hlo_category"}, {"id": "hlo_expression"}, {"id": "tf_op_name"}, {"id": "occurrences"}, {"id": "total_time_in_us"}, {"id": "total_self_time_in_us"}, {"id": "total_self_time_as_fraction"}, {"id": "measured_flop_rate"}, {"id": "flops_v2"}, {"id": "flops"}, {"id": "measured_memory_bw"}, {"id": "bound_by"}, {"id": "source_file"}, {"id": "source_line"}], "rows": [
{"c": [{"v": 1}, {"v": "1111111111111111111"}, {"v": "Convolution"}, {"v": "%convolution.1 = ... calls=conv_sub"}, {"v": "conv2d"}, {"v": 10}, {"v": 1200.0}, {"v": 1000.0}, {"v": 0.5}, {"v": 0.8}, {"v": 500.0}, {"v": 0}, {"v": 50.0}, {"v": "Compute"}, {"v": "model.py"}, {"v": 42}]},
{"c": [{"v": 2}, {"v": "2222222222222222222"}, {"v": "Fusion"}, {"v": "%fusion.2 = ... calls=fusion_sub"}, {"v": "gelu"}, {"v": 5}, {"v": 900.0}, {"v": 800.0}, {"v": 0.4}, {"v": 0.2}, {"v": 100.0}, {"v": 0}, {"v": 80.0}, {"v": "HBM"}, {"v": ""}, {"v": 0}]},
{"c": [{"v": 3}, {"v": "3333333333333333333"}, {"v": "Tuple"}, {"v": "%tuple.3 = tuple()"}, {"v": ""}, {"v": 20}, {"v": 200.0}, {"v": 200.0}, {"v": 0.1}, {"v": 0.0}, {"v": null}, {"v": 10}, {"v": 10.0}, {"v": "Unknown"}, {"v": ""}, {"v": 0}]}]}"#;
const SINGLE: &str = r#"{"cols": [{"id": "rank"}, {"id": "program_id"}, {"id": "hlo_category"}, {"id": "hlo_expression"}, {"id": "total_self_time_in_us"}], "rows": [{"c": [{"v": 1}, {"v": 100}, {"v": "Custom"}, {"v": "EXPRESSION"}, {"v": 500.0}]}]}"#;
const DATATABLE: &str = r#"{"cols": [{"id": "rank", "label": "Rank", "type": "number"}, {"id": "program_id", "label": "Program ID", "type": "number"}, {"id": "hlo_category", "label": "Category", "type": "string"}, {"id": "hlo_expression", "label": "HLO Expression", "type": "string"}, {"id": "tf_op_name", "label": "TF Op Name", "type": "string"}, {"id": "occurrences", "label": "Occurrences", "type": "number"}, {"id": "total_time_in_us", "label": "Total Time (us)", "type": "number"}, {"id": "total_self_time_in_us", "label": "Total Self Time (us)", "type": "number"}, {"id": "total_self_time_as_fraction", "label": "Fraction", "type": "number"}, {"id": "measured_flop_rate", "label": "FLOP Rate", "type": "number"}, {"id": "flops", "label": "FLOPs", "type": "number"}, {"id": "measured_memory_bw", "label": "Memory BW", "type": "number"}, {"id": "bound_by", "label": "Bound By", "type": "string"}, {"id": "source_file", "label": "Source File", "type": "string"}, {"id": "source_line", "label": "Source Line", "type": "number"}], "rows": [
{"c": [{"v": 1}, {"v": 11111}, {"v": "Convolution"}, {"v": "%convolution.1 = ..."}, {"v": "conv2d"}, {"v": 10}, {"v": 1200.0}, {"v": 1000.0}, {"v": 0.5}, {"v": 0.8}, {"v": 500.0}, {"v": 50.0}, {"v": "Compute"}, {"v": "model.py"}, {"v": 42}]},
{"c": [{"v": 2}, {"v": 22222}, {"v": "Fusion"}, {"v": "%fusion.2 = ..."}, {"v": "gelu"}, {"v": 5}, {"v": 900.0}, {"v": 800.0}, {"v": 0.4}, {"v": 0.2}, {"v": 100.0}, {"v": 80.0}, {"v": "HBM"}, {"v": ""}, {"v": 0}]}]}"#;
const SERVER: &str = r##"{"cols": [{"id": "rank", "label": "Rank", "type": "number"}, {"id": "program_id", "label": "Program id", "type": "string"}, {"id": "category", "label": "HLO op category", "type": "string"}, {"id": "hlo_op_name", "label": "HLO op name", "type": "string"}, {"id": "hlo_op_expression", "label": "HLO op text", "type": "string"}, {"id": "tf_op_name", "label": "Framework op name", "type": "string"}, {"id": "occurrences", "label": "#Occurrences", "type": "number"}, {"id": "total_time", "label": "Total time (us)", "type": "number"}, {"id": "avg_time", "label": "Avg. time (us)", "type": "number"}, {"id": "total_self_time", "label": "Total self time (us)", "type": "number"}, {"id": "avg_self_time", "label": "Avg. self time (us)", "type": "number"}, {"id": "total_self_time_percent", "label": "Total self time (%)", "type": "number"}, {"id": "cumulative_total_self_time_percent", "label": "Cumulative total self time (%)", "type": "number"}, {"id": "dma_stall_percent", "label": "%time stalled by DMA", "type": "number"}, {"id": "model_flop_rate", "label": "Model GFLOP/s", "type": "number"}, {"id": "normalized_flop_rate", "label": "Normalized GFLOP/s", "type": "number"}, {"id": "measured_memory_bw", "label": "Measured memory BW (GiB/s)", "type": "number"}, {"id": "hbm_bw", "label": "HBM BW (GiB/s)", "type": "number"}, {"id": "cmem_read_bw", "label": "CMEM Read BW (GiB/s)", "type": "number"}, {"id": "cmem_write_bw", "label": "CMEM Write BW (GiB/s)", "type": "number"}, {"id": "operational_intensity", "label": "Operational intensity (FLOPS/Byte)", "type": "number"}, {"id": "bound_by", "label": "Bound by", "type": "string"}, {"id": "hlo_rematerialization", "label": "Rematerialization", "type": "string"}, {"id": "outside_compilation", "label": "Outside Compilation", "type": "string"}, {"id": "autotuned", "label": "Autotuned", "type": "string"}, {"id": "source_info", "label": "Source Info", "type": "string"}, {"id": "core_type", "label": "TPU core type", "type": "string"}, {"id": "parent_op_name", "label": "Parent op name", "type": "string"}, {"id": "vdd_energy", "label": "VDD Energy (J)", "type": "number"}], "rows": [{"c": [{"v": 1}, {"v": "15845321592809624413"}, {"v": "convolution fusion"}, {"v": "fusion.12341"}, {"v": "%fusion.12341 = (f32[32,128,128], f32[32,128,128]) fusion(%arg0, %arg1), kind=kCustom, calls=%fused_computation.12341"}, {"v": "conv2d"}, {"v": 3}, {"v": 1318200.0}, {"v": 439400.0}, {"v": 974132.5}, {"v": 324710.8}, {"v": 99.77}, {"v": 99.77}, {"v": 0.0}, {"v": 1200.5}, {"v": 1150.0}, {"v": 650.2}, {"v": 650.2}, {"v": 0.0}, {"v": 0.0}, {"v": 15.8}, {"v": "Compute"}, {"v": "No"}, {"v": "No"}, {"v": "No"}, {"v": "transformer.py:5841"}, {"v": "TensorCore"}, {"v": "root"}, {"v": 0.0}]}]}"##;
const LONG_EXPRESSION: &str = "custom_op_instruction_without_standard_hlo_format_that_is_very_long";

fn records(fake: &Fake, pairs: &[(&str, J)]) -> Vec<J> {
    json(get_hlo_stats(fake, &args("session_123", pairs))).items().to_vec()
}

fn names(records: &[J]) -> Vec<String> {
    records.iter().map(|record| record.at("op_name").text()).collect()
}

#[test]
fn test_get_hlo_stats_json_default() {
    let records = records(&Fake::fixed(DATABASE), &[]);
    assert_eq!(names(&records), ["convolution.1", "fusion.2", "tuple.3"]);
    assert_eq!(records[0].at("self_time_percent"), &J::Float(50.0));
    assert_eq!(records[0].at("source_file").str(), Some("model.py"));
    assert_eq!(records[0].at("source_line"), &J::Int(42));
}

#[test]
fn test_get_hlo_stats_bypass_cache_forwarded() {
    let fake = Fake::fixed(DATABASE);
    get_hlo_stats(&fake, &args("session_123", &[("bypass_cache", J::Bool(true))])).unwrap();
    let calls = fake.calls.borrow();
    let (tool, params) = calls.last().unwrap();
    assert_eq!(tool, "hlo_stats.json");
    let expected: Vec<(String, String)> = [("format", "json"), ("tqx", "out:pb"), ("bypass_cache", "True")].iter().map(|(key, value)| (key.to_string(), value.to_string())).collect();
    assert_eq!(params, &expected);
}

#[test]
fn test_op_name_fallback() {
    let records = records(&Fake::fixed(&SINGLE.replace("EXPRESSION", LONG_EXPRESSION)), &[]);
    assert_eq!(names(&records), [LONG_EXPRESSION]);
}

#[test]
fn test_op_name_regex_no_redos_on_multiple_percents() {
    let records = records(&Fake::fixed(&SINGLE.replace("EXPRESSION", "%foo %bar %baz %qux = f32[10] add(%foo, %bar)")), &[]);
    assert_eq!(names(&records), ["qux"]);
}

#[test]
fn test_sorting() {
    for (sort_by, expected) in [
        ("self_time", ["convolution.1", "fusion.2", "tuple.3"]),
        ("total_time", ["convolution.1", "fusion.2", "tuple.3"]),
        ("occurrences", ["tuple.3", "convolution.1", "fusion.2"]),
        ("flops", ["convolution.1", "fusion.2", "tuple.3"]),
        ("bandwidth", ["fusion.2", "convolution.1", "tuple.3"]),
    ] {
        assert_eq!(names(&records(&Fake::fixed(DATABASE), &[("sort_by", J::from(sort_by))])), expected, "{sort_by}");
    }
}

#[test]
fn test_category_filter() {
    assert_eq!(names(&records(&Fake::fixed(DATABASE), &[("category_filter", J::from("Convolution"))])), ["convolution.1"]);
}

#[test]
fn test_limit() {
    assert_eq!(records(&Fake::fixed(DATABASE), &[("limit", J::Int(2))]).len(), 2);
}

#[test]
fn test_error_fetch_failure() {
    let fake = Fake::new(|_, _| Err(Error::new(Kind::Runtime, "Connection Refused")));
    let error = get_hlo_stats(&fake, &args("session_123", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Runtime);
    assert!(error.message.contains("Connection Refused"));
}

#[test]
fn test_error_empty_result() {
    let error = get_hlo_stats(&Fake::new(|_, _| Ok(None)), &args("session_123", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Runtime);
}

#[test]
fn test_get_hlo_stats_datatable_json_success() {
    let records = records(&Fake::fixed(DATATABLE), &[]);
    assert_eq!(names(&records), ["convolution.1", "fusion.2"]);
    assert_eq!(records[0].at("self_time_percent"), &J::Float(50.0));
    assert_eq!(records[0].at("flops"), &J::Float(500.0));
}

#[test]
fn test_get_hlo_stats_server_datatable_schema_exact() {
    let records = records(&Fake::fixed(SERVER), &[]);
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.at("rank"), &J::Int(1));
    assert_eq!(record.at("program_id"), &J::Int(15845321592809624413));
    assert_eq!(record.at("category").str(), Some("convolution fusion"));
    assert_eq!(record.at("op_name").str(), Some("fusion.12341"));
    assert_eq!(record.at("tf_op_name").str(), Some("conv2d"));
    assert_eq!(record.at("occurrences"), &J::Int(3));
    assert_eq!(record.at("total_time_us"), &J::Float(1318200.0));
    assert_eq!(record.at("total_self_time_us"), &J::Float(974132.5));
    assert_eq!(record.at("self_time_percent"), &J::Float(99.77));
    assert_eq!(record.at("measured_flop_rate"), &J::Float(1200.5));
    assert_eq!(record.at("measured_memory_bw_gbs"), &J::Float(650.2));
    assert_eq!(record.at("bound_by").str(), Some("Compute"));
    assert_eq!(record.at("source_file").str(), Some("transformer.py"));
    assert_eq!(record.at("source_line"), &J::Int(5841));
}

#[test]
fn test_error_empty_records() {
    let fake = Fake::fixed(r#"{"cols": [{"id": "rank"}], "rows": []}"#);
    let error = get_hlo_stats(&fake, &args("session_123", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert!(error.message.contains("No HLO stats records found"));
}
