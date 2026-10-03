use super::cli_support::{Fake, args, json, parse, text};
use crate::cli::Kind;
use crate::cli::hlo::get_peak_allocations;
use crate::cli::json::J;
use std::cell::Cell;

const NOTE: &str = "# Peak Memory Allocations by Module\n\n> [!NOTE]\n> **Aggregation Logic:**\n> - Buffers with similar names (e.g., `name.1`, `name.2`) and identical sizes are aggregated into `name.*`.\n> - Buffers smaller than the threshold are aggregated into 'Others'.\n\n";
const MODULE_ONE: &str = "## Module: `module1`\nTotal HBM: 100.00 MiB\n\n| Instruction | Size (MiB) |\n| :--- | ---: |\n| `op1` | 2.00 |\n";
const OP1_2MIB: &str = r#"{"totalBufferAllocationMib": 100.0, "bufferAssignment": {"logicalBuffers": [{"size": "2097152", "definedAt": {"instructionName": "op1"}}]}}"#;
const OP1_1MIB: &str = r#"{"totalBufferAllocationMib": 100.0, "bufferAssignment": {"logicalBuffers": [{"size": "1048576", "definedAt": {"instructionName": "op1"}}]}}"#;
const OP2_2MIB: &str = r#"{"totalBufferAllocationMib": 200.0, "bufferAssignment": {"logicalBuffers": [{"size": "2097152", "definedAt": {"instructionName": "op2"}}]}}"#;

type Calls = Vec<(String, Vec<(String, String)>)>;

fn sequence(responses: &[&'static str]) -> Fake {
    let (responses, next) = (responses.to_vec(), Cell::new(0));
    Fake::new(move |_, _| {
        let index = next.get();
        next.set(index + 1);
        Ok(responses.get(index).map(|response| response.as_bytes().to_vec()))
    })
}

fn calls(modules: &[&str]) -> Calls {
    let format = ("format".to_string(), "json".to_string());
    std::iter::once(("memory_viewer.json".to_string(), vec![format.clone()]))
        .chain(modules.iter().map(|module| ("memory_viewer.json".to_string(), vec![format.clone(), ("module_name".to_string(), module.to_string())])))
        .collect()
}

fn no_summary() -> (&'static str, J) {
    ("include_summary", J::Bool(false))
}

#[test]
fn test_get_peak_allocations_success_variants() {
    let cases: [(&[&'static str], &str, &[&str]); 2] = [
        (
            &["module1,module2", OP1_1MIB, OP2_2MIB],
            r#"[{"module_name": "module2", "total_hbm_mib": 200.0, "top_buffers": [{"instruction": "op2", "size_mib": 2.0}]}, {"module_name": "module1", "total_hbm_mib": 100.0, "top_buffers": [{"instruction": "op1", "size_mib": 1.0}]}]"#,
            &["module1", "module2"],
        ),
        (
            &["module1", r#"{"totalBufferAllocationMib": 100.0, "maxHeap": [{"logicalBufferSizeMib": 1.0, "instructionName": "op1"}]}"#],
            r#"[{"module_name": "module1", "total_hbm_mib": 100.0, "top_buffers": [{"instruction": "op1", "size_mib": 1.0}]}]"#,
            &["module1"],
        ),
    ];
    for (responses, expected, modules) in cases {
        let fake = sequence(responses);
        assert_eq!(json(get_peak_allocations(&fake, &args("session_123", &[no_summary()]))), parse(expected));
        assert_eq!(*fake.calls.borrow(), calls(modules));
    }
}

#[test]
fn test_get_peak_allocations_empty_data_variants() {
    for response in ["", "  "] {
        let error = get_peak_allocations(&Fake::fixed(response), &args("session_123", &[])).unwrap_err();
        assert_eq!(error.kind, Kind::Value, "{response:?}");
    }
}

#[test]
fn test_get_peak_allocations_error_markdown() {
    let error = get_peak_allocations(&Fake::fixed(""), &args("session_123", &[("output_format", J::from("markdown"))])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}

#[test]
fn test_get_peak_allocations_aggregation() {
    let fake = sequence(&[
        "module1",
        r#"{"totalBufferAllocationMib": 300.0, "bufferAssignment": {"logicalBuffers": [{"size": "67108864", "definedAt": {"instructionName": "param.1"}},{"size": "67108864", "definedAt": {"instructionName": "param.2"}},{"size": "1048576", "definedAt": {"instructionName": "op1"}}]}}"#,
    ]);
    let expected = r#"[{"module_name": "module1", "total_hbm_mib": 300.0, "top_buffers": [{"instruction": "param.* (2 occurrences of size 64 MiB)", "size_mib": 128.0}, {"instruction": "op1", "size_mib": 1.0}]}]"#;
    assert_eq!(json(get_peak_allocations(&fake, &args("session_123", &[no_summary()]))), parse(expected));
    assert_eq!(*fake.calls.borrow(), calls(&["module1"]));
}

#[test]
fn test_get_peak_allocations_limit_variants() {
    for (limit, expected_len) in [(2, 2), (0, 3)] {
        let fake = sequence(&["module1,module2,module3", r#"{"totalBufferAllocationMib": 100.0}"#, r#"{"totalBufferAllocationMib": 300.0}"#, r#"{"totalBufferAllocationMib": 200.0}"#]);
        let result = json(get_peak_allocations(&fake, &args("session_123", &[("limit", J::Int(limit)), no_summary()])));
        let names: Vec<String> = result.items().iter().map(|module| module.at("module_name").text()).collect();
        assert_eq!(names, ["module2", "module3", "module1"][..expected_len]);
        assert_eq!(*fake.calls.borrow(), calls(&["module1", "module2", "module3"]));
    }
}

#[test]
fn test_get_peak_allocations_size_threshold() {
    let fake = sequence(&[
        "module1",
        r#"{"totalBufferAllocationMib": 100.0, "bufferAssignment": {"logicalBuffers": [{"size": "2097152", "definedAt": {"instructionName": "large_op"}},{"size": "524288", "definedAt": {"instructionName": "small_op1"}},{"size": "262144", "definedAt": {"instructionName": "small_op2"}}]}}"#,
    ]);
    let expected = r#"[{"module_name": "module1", "total_hbm_mib": 100.0, "top_buffers": [{"instruction": "large_op", "size_mib": 2.0}, {"instruction": "Others (< 1.0 MiB)", "size_mib": 0.75}]}]"#;
    assert_eq!(json(get_peak_allocations(&fake, &args("session_123", &[("min_size_mib", J::Float(1.0)), no_summary()]))), parse(expected));
    assert_eq!(*fake.calls.borrow(), calls(&["module1"]));
}

#[test]
fn test_get_peak_allocations_markdown() {
    let fake = sequence(&["module1", OP1_2MIB]);
    let result = text(get_peak_allocations(&fake, &args("session_123", &[("output_format", J::from("markdown")), no_summary()])));
    assert_eq!(result, format!("{NOTE}{MODULE_ONE}"));
    assert_eq!(*fake.calls.borrow(), calls(&["module1"]));
}

#[test]
fn test_get_peak_allocations_no_aggregation() {
    let fake = sequence(&[
        "module1",
        r#"{"totalBufferAllocationMib": 100.0, "bufferAssignment": {"logicalBuffers": [{"size": "1048576", "definedAt": {"instructionName": "param.1"}},{"size": "1048576", "definedAt": {"instructionName": "param.2"}},{"size": "524288", "definedAt": {"instructionName": "small_op"}}]}}"#,
    ]);
    let expected = r#"[{"module_name": "module1", "total_hbm_mib": 100.0, "top_buffers": [{"instruction": "param.1", "size_mib": 1.0}, {"instruction": "param.2", "size_mib": 1.0}, {"instruction": "small_op", "size_mib": 0.5}]}]"#;
    assert_eq!(json(get_peak_allocations(&fake, &args("session_123", &[("aggregate_instructions", J::Bool(false)), no_summary()]))), parse(expected));
}

#[test]
fn test_get_peak_allocations_summary_json() {
    let fake = sequence(&["module1", OP1_2MIB]);
    let expected = r#"{"summary": {"total_modules": 1, "top_modules": [{"module_name": "module1", "total_hbm_mib": 100.0}]}, "modules": [{"module_name": "module1", "total_hbm_mib": 100.0, "top_buffers": [{"instruction": "op1", "size_mib": 2.0}]}]}"#;
    assert_eq!(json(get_peak_allocations(&fake, &args("session_123", &[("include_summary", J::Bool(true))]))), parse(expected));
    assert_eq!(*fake.calls.borrow(), calls(&["module1"]));
}

#[test]
fn test_get_peak_allocations_summary_markdown() {
    let fake = sequence(&["module1", OP1_2MIB]);
    let result = text(get_peak_allocations(&fake, &args("session_123", &[("output_format", J::from("markdown")), ("include_summary", J::Bool(true))])));
    let summary = "## Session Summary\n- Total Modules: 1\n\n| Module | Total HBM (MiB) |\n| :--- | ---: |\n| `module1` | 100.00 |\n\n";
    assert_eq!(result, format!("{NOTE}{summary}{MODULE_ONE}"));
    assert_eq!(*fake.calls.borrow(), calls(&["module1"]));
}

#[test]
fn test_missing_memory_returns_clean_diagnostic() {
    let error = get_peak_allocations(&Fake::fixed(""), &args("session_without_memory", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
    assert!(error.message.contains("No memory viewer data returned"));
}

#[test]
fn test_get_module_names_success() {
    let fake = sequence(&["module1,module2", OP1_1MIB, OP2_2MIB]);
    let result = json(get_peak_allocations(&fake, &args("session_123", &[no_summary()])));
    assert_eq!(result.items().len(), 2);
    assert_eq!(*fake.calls.borrow(), calls(&["module1", "module2"]));
}

#[test]
fn test_get_module_names_failure_variants() {
    for (response, expected) in [("", "No memory viewer data returned"), (",", "No HLO modules found")] {
        let error = get_peak_allocations(&Fake::fixed(response), &args("session_123", &[])).unwrap_err();
        assert_eq!(error.kind, Kind::Value);
        assert!(error.message.contains(expected), "{}", error.message);
    }
}

#[test]
fn test_parse_and_aggregate_buffers_logical_buffers() {
    let module = r#"{"bufferAssignment": {"logicalBuffers": [{"size": "1048576", "definedAt": {"instructionName": "param.1"}}, {"size": "1048576", "definedAt": {"instructionName": "param.2"}}, {"size": "2097152", "definedAt": {"instructionName": "op1"}}]}}"#;
    let result = json(get_peak_allocations(&sequence(&["module1", module]), &args("session_123", &[no_summary()])));
    let expected = r#"[{"instruction": "param.* (2 occurrences of size 1 MiB)", "size_mib": 2.0}, {"instruction": "op1", "size_mib": 2.0}]"#;
    assert_eq!(result.items()[0].at("top_buffers"), &parse(expected));
}

#[test]
fn test_parse_and_aggregate_buffers_max_heap_fallback() {
    let module = r#"{"maxHeap": [{"logicalBufferSizeMib": 1.0, "instructionName": "param.1"}, {"logicalBufferSizeMib": 1.0, "instructionName": "param.2"}, {"logicalBufferSizeMib": 2.0, "instructionName": "op1"}]}"#;
    let result = json(get_peak_allocations(&sequence(&["module1", module]), &args("session_123", &[no_summary()])));
    let expected = r#"[{"instruction": "param.* (2 occurrences of size 1 MiB)", "size_mib": 2.0}, {"instruction": "op1", "size_mib": 2.0}]"#;
    assert_eq!(result.items()[0].at("top_buffers"), &parse(expected));
}

#[test]
fn test_parse_and_aggregate_buffers_no_aggregation() {
    let module = r#"{"bufferAssignment": {"logicalBuffers": [{"size": "1048576", "definedAt": {"instructionName": "param.1"}}, {"size": "1048576", "definedAt": {"instructionName": "param.2"}}, {"size": "2097152", "definedAt": {"instructionName": "op1"}}, {"size": "524288", "definedAt": {"instructionName": "small_op"}}]}}"#;
    let result = json(get_peak_allocations(&sequence(&["module1", module]), &args("session_123", &[("aggregate_instructions", J::Bool(false)), no_summary()])));
    let expected =
        r#"[{"instruction": "op1", "size_mib": 2.0}, {"instruction": "param.1", "size_mib": 1.0}, {"instruction": "param.2", "size_mib": 1.0}, {"instruction": "small_op", "size_mib": 0.5}]"#;
    assert_eq!(result.items()[0].at("top_buffers"), &parse(expected));
}

#[test]
fn test_fetch_modules_data() {
    let fake = sequence(&["module1,module2", OP1_1MIB, OP2_2MIB]);
    let result = json(get_peak_allocations(&fake, &args("session_123", &[no_summary()])));
    let modules: Vec<(String, J, J)> = result.items().iter().map(|module| (module.at("module_name").text(), module.at("total_hbm_mib").clone(), module.at("top_buffers").clone())).collect();
    assert_eq!(
        modules,
        [
            ("module2".to_string(), J::Float(200.0), parse(r#"[{"instruction": "op2", "size_mib": 2.0}]"#)),
            ("module1".to_string(), J::Float(100.0), parse(r#"[{"instruction": "op1", "size_mib": 1.0}]"#)),
        ]
    );
    assert_eq!(fake.calls.borrow()[1..], calls(&["module1", "module2"])[1..]);
}
