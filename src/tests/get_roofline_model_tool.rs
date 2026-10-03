use super::cli_support::{Fake, args, json};
use crate::cli::Kind;
use crate::cli::json::J;
use crate::cli::overview::get_roofline_model;

const COLUMNS: &str = r#""cols": [{"id": "step", "type": "string"}, {"id": "rank", "type": "number"}, {"id": "category", "type": "string"}, {"id": "operation", "type": "string"}, {"id": "occurrences", "type": "number"}, {"id": "total_time", "type": "number"}, {"id": "total_self_time", "type": "number"}, {"id": "total_self_time_percent", "type": "number"}, {"id": "measured_flop_rate", "type": "number"}, {"id": "model_flop_rate", "type": "number"}, {"id": "measured_memory_bw", "type": "number"}, {"id": "hbm_bw", "type": "number"}, {"id": "operational_intensity", "type": "number"}, {"id": "bound_by", "type": "string"}, {"id": "roofline_efficiency", "type": "number"}, {"id": "compute_efficiency", "type": "number"}, {"id": "max_mem_bw_utilization", "type": "number"}, {"id": "hlo_module_id", "type": "string"}, {"id": "source_info", "type": "string"}], "p": {"device_type": "TPU v6 Lite", "peak_flop_rate": "946700", "peak_hbm_bw": "1525.5", "hbm_ridge_point": "577.96"}"#;

fn table(rows: &str) -> String {
    format!("[{{{COLUMNS}, \"rows\": [{rows}]}}]")
}

#[test]
fn test_get_roofline_model_success() {
    let data = table(
        r#"{"c": [{"v": "Total"}, {"v": 0.0}, {"v": "Program"}, {"v": "Program"}, {"v": 1.0}, {"v": 100000.0}, {"v": 0.0}, {"v": 0.0}, {"v": 7181.72}, {"v": 6440.89}, {"v": 258.31}, {"v": 205.79}, {"v": 25.89}, {"v": "HBM"}, {"v": 0.1349}, {"v": 0.0076}, {"v": 0.1349}, {"v": "0"}, {"v": ""}]}, {"c": [{"v": "Total"}, {"v": 1.0}, {"v": "all-reduce"}, {"v": "psum.118"}, {"v": 10.0}, {"v": 50000.0}, {"v": 50000.0}, {"v": 0.50}, {"v": 100.0}, {"v": 90.0}, {"v": 200.0}, {"v": 180.0}, {"v": 0.25}, {"v": "HBM"}, {"v": 0.1631}, {"v": 0.0001}, {"v": 0.1631}, {"v": "12345"}, {"v": "<div title='file.py:10'>file.py:10</div>"}]}"#,
    );
    let parsed = json(get_roofline_model(&Fake::fixed(&data), &args("test_session", &[("top_n", J::Int(5))])));
    let program = parsed.at("program");
    assert_eq!(program.at("bound_by").str(), Some("HBM"));
    assert_eq!(program.at("roofline_efficiency_percent").str(), Some("13.49%"));
    assert_eq!(program.at("compute_efficiency_percent").str(), Some("0.76%"));
    assert_eq!(program.at("max_mem_bw_utilization_percent").str(), Some("13.49%"));
    assert_eq!(program.at("operational_intensity_flop_per_byte"), &J::Float(25.89));
    assert_eq!(program.at("measured_flop_rate_gflops"), &J::Float(7181.72));
    assert_eq!(program.at("hbm_bw_gibs"), &J::Float(205.79));
    assert_eq!(parsed.at("device_info").at("device_type").str(), Some("TPU v6 Lite"));
    assert_eq!(parsed.at("device_info").at("peak_flop_rate"), &J::Float(946700.0));
    let operations = parsed.at("top_operations").items();
    assert_eq!(operations.len(), 1);
    let operation = &operations[0];
    assert_eq!(operation.at("name").str(), Some("psum.118"));
    assert_eq!(operation.at("category").str(), Some("all-reduce"));
    assert_eq!(operation.at("total_self_time_ms"), &J::Float(50.0));
    assert_eq!(operation.at("bound_by").str(), Some("HBM"));
    assert_eq!(operation.at("roofline_efficiency_percent").str(), Some("16.31%"));
    assert_eq!(operation.at("source_info").str(), Some("file.py:10"));
}

#[test]
fn test_get_roofline_model_custom_call_opaque() {
    let data = table(
        r#"{"c": [{"v": "Total"}, {"v": 0.0}, {"v": "Program"}, {"v": "Program"}, {"v": 1.0}, {"v": 100000.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": "Unknown"}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": "0"}, {"v": ""}]}, {"c": [{"v": "Total"}, {"v": 1.0}, {"v": "custom-call"}, {"v": "custom-call.1"}, {"v": 1.0}, {"v": 50000.0}, {"v": 50000.0}, {"v": 1.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": "Unknown"}, {"v": 0.0}, {"v": 0.0}, {"v": 0.0}, {"v": "12345"}, {"v": ""}]}"#,
    );
    let parsed = json(get_roofline_model(&Fake::fixed(&data), &args("test_session", &[("top_n", J::Int(5))])));
    assert_eq!(parsed.at("program").at("bound_by").str(), Some("CustomCall (opaque)"));
    let operations = parsed.at("top_operations").items();
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].at("bound_by").str(), Some("CustomCall (opaque)"));
    assert!(parsed.at("guidance").text().contains("Pallas kernels"));
}

#[test]
fn test_get_roofline_model_no_data() {
    let parsed = json(get_roofline_model(&Fake::tools(|_| None), &args("test_session", &[])));
    assert_eq!(parsed.at("status").str(), Some("NO_DATA"));
}

#[test]
fn test_get_roofline_model_invalid_data() {
    assert_eq!(get_roofline_model(&Fake::fixed("{}"), &args("test_session", &[])).unwrap_err().kind, Kind::Value);
}

#[test]
fn test_get_roofline_model_bypass_cache() {
    let fake = Fake::tools(|_| None);
    get_roofline_model(&fake, &args("test_session", &[("bypass_cache", J::Bool(true))])).unwrap();
    assert!(fake.calls.borrow().iter().any(|(tool, params)| tool == "roofline_model.json" && *params == [("bypass_cache".to_string(), "True".to_string())]));
}
