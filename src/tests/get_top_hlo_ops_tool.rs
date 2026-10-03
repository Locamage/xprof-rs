use super::cli_support::{Fake, args, json};
use crate::cli::Kind;
use crate::cli::json::J;
use crate::cli::ops::get_top_hlo_ops;

const PROFILE: &str = r#"{"byCategory": {"name": "root", "metrics": {"rawTime": 1}, "children": [
{"name": "Op1", "xla": {"category": "Fusion", "sourceInfo": {"fileName": "/path/to/model.py", "lineNumber": 42, "stackFrame": "main()\n"}}, "metrics": {"rawTime": 60000000000, "occurrences": 1, "rawFlops": 100, "rawBytesAccessedArray": [200]}},
{"name": "Op2", "xla": {"category": "Convolution"}, "metrics": {"rawTime": 30000000000, "occurrences": 1, "rawFlops": 1000, "rawBytesAccessedArray": [50]}},
{"name": "Op3", "xla": {"category": "Tuple"}, "metrics": {"rawTime": 10000000000, "occurrences": 1, "rawFlops": 10, "rawBytesAccessedArray": [1000]}}]}}"#;

fn top(fake: &Fake, pairs: &[(&str, J)]) -> J {
    json(get_top_hlo_ops(fake, &args("test_session", pairs)))
}

fn names(result: &J, key: &str) -> Vec<String> {
    result.at(key).items().iter().map(|op| op.at("name").text()).collect()
}

#[test]
fn test_get_top_hlo_ops_by_time() {
    let fake = Fake::fixed(PROFILE);
    let result = top(&fake, &[("limit", J::Int(2))]);
    assert!(!result.has("error"));
    assert_eq!(names(&result, "top_by_time"), ["root/Op1", "root/Op2"]);
    let calls = fake.calls.borrow();
    assert_eq!(calls.last().unwrap(), &("op_profile".to_string(), vec![("format".to_string(), "pb".to_string())]));
}

#[test]
fn test_get_top_hlo_ops_by_flops() {
    let result = top(&Fake::fixed(PROFILE), &[("limit", J::Int(2))]);
    assert_eq!(names(&result, "top_by_flops"), ["root/Op2", "root/Op1"]);
}

#[test]
fn test_get_top_hlo_ops_by_bytes() {
    let result = top(&Fake::fixed(PROFILE), &[("limit", J::Int(2))]);
    assert_eq!(names(&result, "top_by_bytes_accessed"), ["root/Op3", "root/Op1"]);
}

#[test]
fn test_get_top_hlo_ops_no_ops() {
    let result = top(&Fake::fixed(r#"{"byCategory": {"metrics": {"rawTime": 0}}}"#), &[]);
    assert_eq!(result.at("top_by_time"), &J::List(Vec::new()));
    assert_eq!(result.at("total_matched"), &J::Int(0));
}

#[test]
fn test_get_top_hlo_ops_source_info() {
    let result = top(&Fake::fixed(PROFILE), &[("limit", J::Int(1))]);
    let op = &result.at("top_by_time").items()[0];
    assert_eq!(op.at("source_file").str(), Some("/path/to/model.py"));
    assert_eq!(op.at("source_line"), &J::Int(42));
    assert_eq!(op.at("stack_frame").str(), Some("main()\n"));
}

#[test]
fn test_get_top_hlo_ops_category_filter() {
    let result = top(&Fake::fixed(PROFILE), &[("limit", J::Int(10)), ("category_filter", J::from("Convolution"))]);
    assert_eq!(names(&result, "top_by_time"), ["root/Op2"]);
}

#[test]
fn test_get_top_hlo_ops_custom_call_provenance() {
    let profile = r#"{"byCategory": {"metrics": {"rawTime": 1000000000}, "children": [{"name": "IDLE", "metrics": {"rawTime": 1000000000, "rawFlops": 0}, "xla": {"category": "custom-call", "provenance": "pallas_call"}}]}}"#;
    let result = top(&Fake::fixed(profile), &[("limit", J::Int(1))]);
    assert_eq!(names(&result, "top_by_time"), ["IDLE [pallas_call]"]);
    assert!(result.at("guidance").text().contains("Pallas kernels"));
}

#[test]
fn test_missing_hlo_returns_clean_diagnostic() {
    let error = get_top_hlo_ops(&Fake::new(|_, _| Ok(None)), &args("session_without_hlo", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert!(error.message.contains("No HLO op_profile found in trace"));
}
