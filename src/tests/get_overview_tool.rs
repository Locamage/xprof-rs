use super::cli_support::{Fake, args, json};
use crate::cli::json::J;
use crate::cli::overview::get_overview;
use crate::cli::{Error, Kind};

const OVERVIEW: &str = r#"[{"p": {"stat_memory_bw": "800 GiB/s", "stat_step_time": "15ms", "run_environment": "TPU v4", "idle_percent%": "15.5", "compute_percent%": "84.5", "invalid_percent%": "not_a_number", "device_type": "TPU v4"}}, {"cols": [{"id": "host_id", "type": "string"}, {"id": "bns_address", "type": "string"}, {"id": "command_line", "type": "string"}], "rows": [{"c": [{"v": "test_host"}, {"v": "/bns/test/address"}, {"v": "test command --streamz_default_root_labels=xmanager:int:12345"}]}]}]"#;
const COMMAND: &str = "test command --streamz_default_root_labels=xmanager:int:12345";

#[test]
fn test_get_overview_success() {
    let fake = Fake::fixed(OVERVIEW);
    let result = json(get_overview(&fake, &args("session_123", &[("include_command", J::Bool(false))])));
    let (summary, run) = (result.at("performance_summary"), result.at("run_environment"));
    assert_eq!(summary.at("stat_memory_bw").str(), Some("800 GiB/s"));
    assert_eq!(summary.at("stat_step_time").str(), Some("15ms"));
    assert_eq!(run.at("run_environment").str(), Some("TPU v4"));
    assert_eq!(run.at("device_type").str(), Some("TPU v4"));
    assert_eq!(run.at("hostname").str(), Some("test_host"));
    assert_eq!(run.at("bns").str(), Some("/bns/test/address"));
    assert_eq!(run.at("xid").str(), Some("12345"));
    assert!(!run.has("run_command"));
    let result = json(get_overview(&fake, &args("session_123", &[("include_command", J::Bool(true))])));
    let run = result.at("run_environment");
    assert_eq!(run.at("run_command").str(), Some(COMMAND));
    assert_eq!(run.at("hostname").str(), Some("test_host"));
    assert_eq!(run.at("bns").str(), Some("/bns/test/address"));
    assert_eq!(run.at("xid").str(), Some("12345"));
}

#[test]
fn test_get_overview_error() {
    let error = get_overview(&Fake::new(|_, _| Err(Error::new(Kind::Runtime, "RPC Fail"))), &args("session_123", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Runtime);
    assert!(error.message.contains("RPC Fail"));
}

#[test]
fn test_get_overview_roofline_fallback() {
    let fake = Fake::tools(|tool| {
        match tool {
        "overview_page" | "overview_page.json" => {
            Some(r#"[{"p": {"flop_rate_utilization_relative_to_roofline": "0.0%", "memory_bw_utilization_relative_to_hw_limit": "0.0%", "device_type": "TPU v6 Lite"}}]"#.into())
        }
        "roofline_model" | "roofline_model.json" => Some(
            r#"[{"cols": [{"id": "category", "type": "string"}, {"id": "roofline_efficiency", "type": "number"}, {"id": "compute_efficiency", "type": "number"}, {"id": "max_mem_bw_utilization", "type": "number"}, {"id": "bound_by", "type": "string"}, {"id": "operational_intensity", "type": "number"}], "rows": [{"c": [{"v": "Program"}, {"v": 0.1349}, {"v": 0.0076}, {"v": 0.1349}, {"v": "HBM"}, {"v": 25.89}]}]}]"#.into(),
        ),
        _ => None,
    }
    });
    let result = json(get_overview(&fake, &args("session_123", &[])));
    let summary = result.at("performance_summary");
    assert_eq!(summary.at("flop_rate_utilization_relative_to_roofline").str(), Some("13.49%"));
    assert_eq!(summary.at("roofline_efficiency_percent").str(), Some("13.49%"));
    assert_eq!(summary.at("compute_efficiency_percent").str(), Some("0.76%"));
    assert_eq!(summary.at("memory_bw_utilization_relative_to_hw_limit").str(), Some("13.49%"));
    assert_eq!(summary.at("bound_by").str(), Some("HBM"));
    assert_eq!(summary.at("operational_intensity_flop_per_byte"), &J::Float(25.89));
}
