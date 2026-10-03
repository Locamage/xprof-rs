use super::cli_support::{Fake, args, json, parse, text};
use crate::cli::json::J;
use crate::cli::ops::{get_hlo_op_profile, get_profile_summary};
use crate::cli::overview::{get_device_information, get_hosts};
use crate::cli::{Error, Kind};

const PROFILE: &str = r#"{"byCategory": {"name": "by_category", "metrics": {"rawTime": 100000000000, "occurrences": 15, "rawFlops": 1500}, "children": [
{"name": "MatMul", "category": {}, "metrics": {"rawTime": 60000000000, "occurrences": 10, "rawFlops": 1000, "rawBytesAccessedArray": [100, 200]}},
{"name": "Fusion", "xla": {"category": "FusionCategory"}, "metrics": {"rawTime": 40000000000, "occurrences": 5, "rawFlops": 500}}]}}"#;
const NESTED: &str = r#"{"byCategory": {"name": "by_category", "metrics": {"rawTime": 100000000000, "occurrences": 15, "rawFlops": 1500}, "children": [
{"name": "MatMul", "category": {}, "metrics": {"rawTime": 60000000000, "occurrences": 10, "rawFlops": 1000}},
{"name": "FusionParent", "xla": {"category": "FusionCategory"}, "metrics": {"rawTime": 40000000000, "occurrences": 5, "rawFlops": 500}, "children": [
{"name": "child_add", "xla": {"category": "FusionCategory"}, "metrics": {"rawTime": 15000000000, "occurrences": 5, "rawFlops": 200}},
{"name": "child_mul", "xla": {"category": "FusionCategory"}, "metrics": {"rawTime": 25000000000, "occurrences": 5, "rawFlops": 300}}]}]}}"#;
const FUSION: &str = r#"{"byCategory": {"name": "by_category", "metrics": {"rawTime": 100000000000, "occurrences": 15, "rawFlops": 1500}, "children": [
{"name": "MatMul", "category": {}, "metrics": {"rawTime": 60000000000, "occurrences": 10, "rawFlops": 1000}},
{"name": "fusion.668", "xla": {"category": "convolution fusion"}, "metrics": {"rawTime": 40000000000, "occurrences": 5, "rawFlops": 500}, "children": [
{"name": "sub_add", "xla": {"category": "convolution fusion"}, "metrics": {"rawTime": 0, "occurrences": 5, "rawFlops": 0}},
{"name": "sub_mul", "xla": {"category": "convolution fusion"}, "metrics": {"rawTime": 0, "occurrences": 5, "rawFlops": 0}}]}]}}"#;
const SUMMARY_PROFILE: &str = r#"{"byCategory": {"name": "root", "metrics": {"rawTime": 1e14}}}"#;

fn profile(data: &str, pairs: &[(&str, J)]) -> J {
    json(get_hlo_op_profile(&Fake::fixed(data), &args("session_op", pairs)))
}

fn categories(result: &J) -> Vec<String> {
    result.at("category_summary").items().iter().map(|entry| entry.at("category").text()).collect()
}

fn fraction_total(result: &J) -> f64 {
    result.at("category_summary").items().iter().map(|entry| entry.at("fraction_of_total_time").float().unwrap()).sum()
}

fn flat_names(data: &str) -> Vec<String> {
    profile(data, &[("view", J::from("flat"))]).items().iter().map(|op| op.at("name").text()).collect()
}

fn failure(data: &str, pairs: &[(&str, J)]) -> Kind {
    get_hlo_op_profile(&Fake::fixed(data), &args("session_op", pairs)).unwrap_err().kind
}

fn summary_params(bypass: bool) -> Vec<(String, String)> {
    let fake = Fake::fixed(SUMMARY_PROFILE);
    let result = text(get_profile_summary(&fake, &args("session_summary", &[("bypass_cache", J::Bool(bypass))])));
    assert!(result.contains("Profile Summary"));
    assert!(result.contains("Total Time:"));
    let calls = fake.calls.borrow();
    assert_eq!(calls.last().unwrap().0, "op_profile");
    calls.last().unwrap().1.clone()
}

#[test]
fn test_get_hlo_op_profile_grouped_default() {
    let result = profile(PROFILE, &[]);
    assert!(result.has("category_summary") && result.has("grouped_operations") && result.has("navigation_hints"));
    let names = categories(&result);
    assert!(names.contains(&"Category: MatMul".to_string()) && names.contains(&"FusionCategory".to_string()));
    assert_eq!(result.at("total_profile_time_ms"), &J::Float(100.0));
    let hints = result.at("navigation_hints");
    assert!(hints.has("drill_down_category") && hints.has("available_categories"));
    assert!(hints.at("inspect_op_neighborhood").text().contains("--instruction_name="));
    assert!(!hints.at("inspect_op_neighborhood").text().contains("--op_name="));
}

#[test]
fn test_get_hlo_op_profile_category_view() {
    let result = profile(PROFILE, &[("view", J::from("category"))]);
    assert_eq!(result.at("total_profile_time_ms"), &J::Float(100.0));
    assert!(!result.has("grouped_operations"));
    assert_eq!(categories(&result), ["Category: MatMul", "FusionCategory"]);
    assert_eq!(result.at("category_summary").items()[0].at("total_self_time_ms"), &J::Float(60.0));
    assert!((fraction_total(&result) - 1.0).abs() < 5e-5);
}

#[test]
fn test_get_hlo_op_profile_nested_fusions_category_rollup_unity() {
    assert!((fraction_total(&profile(NESTED, &[("view", J::from("category"))])) - 1.0).abs() < 5e-5);
    let names = flat_names(NESTED);
    for expected in ["by_category/MatMul", "by_category/FusionParent/child_add", "by_category/FusionParent/child_mul"] {
        assert!(names.contains(&expected.to_string()), "{expected}");
    }
    assert!(!names.contains(&"by_category/FusionParent".to_string()));
}

#[test]
fn test_get_hlo_op_profile_fusion_with_zero_time_children_preserved() {
    let result = profile(FUSION, &[("view", J::from("category"))]);
    let fusion = result.at("category_summary").items().iter().find(|entry| entry.at("category").str() == Some("convolution fusion")).unwrap().clone();
    assert_eq!(fusion.at("total_self_time_ms"), &J::Float(40.0));
    assert!((fraction_total(&result) - 1.0).abs() < 5e-5);
    let names = flat_names(FUSION);
    assert!(names.contains(&"by_category/MatMul".to_string()) && names.contains(&"by_category/fusion.668".to_string()));
    assert!(!names.iter().any(|name| name.starts_with("by_category/fusion.668/")));
}

#[test]
fn test_get_hlo_op_profile_category_filter() {
    let result = profile(PROFILE, &[("category", J::from("Fusion"))]);
    assert_eq!(result.at("category").str(), Some("FusionCategory"));
    assert_eq!(result.at("total_self_time_ms"), &J::Float(40.0));
    assert_eq!(result.at("operations").items().len(), 1);
    assert_eq!(result.at("operations").items()[0].at("name").str(), Some("by_category/Fusion"));
    let hint = result.at("navigation_hints").at("inspect_top_op_ast").text();
    assert!(hint.contains("--instruction_name=") && !hint.contains("--op_name="));
}

#[test]
fn test_get_hlo_op_profile_category_not_found() {
    assert_eq!(failure(PROFILE, &[("category", J::from("NonExistent"))]), Kind::FileNotFound);
}

#[test]
fn test_get_hlo_op_profile_flat_view() {
    let result = profile(PROFILE, &[("view", J::from("flat"))]);
    let ops = result.items();
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].at("name").str(), Some("by_category/MatMul"));
    assert_eq!(ops[0].at("total_self_time_ms"), &J::Float(60.0));
    assert_eq!(ops[1].at("name").str(), Some("by_category/Fusion"));
    assert_eq!(ops[1].at("total_self_time_ms"), &J::Float(40.0));
}

#[test]
fn test_get_hlo_op_profile_tree_view() {
    let result = profile(PROFILE, &[("view", J::from("tree")), ("depth", J::Int(2))]);
    assert!(result.has("tree") && result.has("navigation_hints"));
    assert_eq!(result.at("current_path").str(), Some("by_category"));
    assert_eq!(result.at("tree").at("children").items().len(), 2);
}

#[test]
fn test_get_hlo_op_profile_tree_path_not_found() {
    assert_eq!(failure(PROFILE, &[("view", J::from("tree")), ("path", J::from("invalid/path"))]), Kind::FileNotFound);
}

#[test]
fn test_get_hlo_op_profile_invalid_view() {
    assert_eq!(failure(PROFILE, &[("view", J::from("unsupported_view"))]), Kind::Value);
}

#[test]
fn test_get_hlo_op_profile_invalid_sort_by() {
    assert_eq!(failure(PROFILE, &[("sort_by", J::from("unsupported_sort"))]), Kind::Value);
}

#[test]
fn test_get_profile_summary_success() {
    let expected: Vec<(String, String)> = vec![("format".into(), "json".into()), ("bypass_cache".into(), "False".into())];
    assert_eq!(summary_params(false), expected);
}

#[test]
fn test_get_profile_summary_bypass_cache() {
    let expected: Vec<(String, String)> = vec![("format".into(), "json".into()), ("bypass_cache".into(), "True".into())];
    assert_eq!(summary_params(true), expected);
}

#[test]
fn test_get_hosts() {
    let result = json(get_hosts(&Fake::fixed("").with_hosts(&["host1", "host2"]), &args("session_hosts", &[])));
    assert_eq!(result, parse(r#"{"hosts": [{"hostname": "host1"}, {"hostname": "host2"}]}"#));
}

#[test]
fn test_get_hosts_error() {
    let fake = Fake::fixed("").failing_hosts(Error::new(Kind::Runtime, "RPC Fail"));
    assert_eq!(get_hosts(&fake, &args("session_hosts", &[])).unwrap_err().kind, Kind::Runtime);
}

#[test]
fn test_get_hosts_empty() {
    assert_eq!(get_hosts(&Fake::fixed("").with_hosts(&[]), &args("session_hosts", &[])).unwrap_err().kind, Kind::FileNotFound);
}

#[test]
fn test_get_device_information_success() {
    let roofline = r#"[{"p": {"device_type": "TPU v5p", "peak_flop_rate": "1234.5", "peak_hbm_bw": "678", "peak_vmem_bw": "910", "ridge_point": "not_a_number"}}]"#;
    let result = json(get_device_information(&Fake::fixed(roofline), &args("session_device", &[])));
    assert_eq!(result.at("device_type").str(), Some("TPU v5p"));
    assert_eq!(result.at("peak_flop_rate"), &J::Float(1234.5));
    assert_eq!(result.at("peak_hbm_bw_gibs"), &J::Float(678.0));
    assert_eq!(result.at("peak_vmem_bw_gibs"), &J::Float(910.0));
    assert!(!result.has("peak_hbm_bw") && !result.has("peak_vmem_bw"));
    assert_eq!(result.at("ridge_point").str(), Some("not_a_number"));
    assert_eq!(result.at("units"), &parse(r#"{"peak_hbm_bw_gibs": "GiB/s", "peak_vmem_bw_gibs": "GiB/s"}"#));
}

#[test]
fn test_get_device_information_error() {
    let fake = Fake::new(|_, _| Err(Error::new(Kind::Runtime, "RPC Fail")));
    assert_eq!(get_device_information(&fake, &args("session_device", &[])).unwrap_err().kind, Kind::Runtime);
}

#[test]
fn test_get_device_information_empty() {
    assert_eq!(get_device_information(&Fake::fixed(""), &args("session_device", &[])).unwrap_err().kind, Kind::FileNotFound);
}

#[test]
fn test_get_profile_summary_missing_data() {
    assert_eq!(get_profile_summary(&Fake::new(|_, _| Ok(None)), &args("session_missing", &[])).unwrap_err().kind, Kind::FileNotFound);
}

#[test]
fn test_get_hlo_op_profile_missing_data() {
    assert_eq!(get_hlo_op_profile(&Fake::new(|_, _| Ok(None)), &args("session_missing", &[])).unwrap_err().kind, Kind::FileNotFound);
}
