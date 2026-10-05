use super::cli_support::{run, scratch};
use super::xspace::XSpace;
use crate::cli::json::J;
use crate::cli::{COMMANDS, literal, preprocess};
use prost::Message;

const SESSION: &str = "session_123";
const MISSING: &str = "Path not found: session_123";
const REPORT: &str = "https://github.com/openxla/xprof/issues";
const TOOL_MODULES: [&str; 17] = [
    "check_host_boundness",
    "get_hlo_stats",
    "get_kernel_stats",
    "get_kpi_metrics",
    "get_llo_analysis",
    "get_llo_debug_string",
    "get_memory_profile",
    "get_overview",
    "get_peak_allocations",
    "get_roofline_model",
    "get_step_trace",
    "get_top_hlo_ops",
    "get_utilization_viewer",
    "verify_numerical_parity",
    "get_graph_viewer",
    "get_kernel_utilization",
    "upload_trace",
];

fn error(argv: &[&str], code: i32, reason: &str) -> J {
    let (status, out, err) = run(argv);
    assert_eq!(status, code, "{argv:?}: {out}");
    let payload = J::parse(&out).unwrap_or_else(|| panic!("{argv:?}: {out}"));
    assert_eq!(payload.at("status").str(), Some("ERROR"));
    assert_eq!(payload.at("reason").str(), Some(reason));
    assert_eq!(payload.has("traceback"), code == 1);
    assert!(err.contains(reason), "{err}");
    payload
}

fn reached(argv: &[&str], prefix: &str) {
    let (status, out, _) = run(argv);
    assert_eq!(status, 0, "{argv:?}: {out}");
    assert_eq!(out, format!("{prefix}FileNotFoundError('{MISSING}')\n"));
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(std::string::ToString::to_string).collect()
}

fn missing(argv: &[&str]) {
    assert_eq!(error(argv, 3, "PATH_ERROR").at("error").str(), Some(MISSING));
}

macro_rules! cases { ($($name:ident: $check:expr;)*) => { $(#[test] fn $name() { $check })* } }

cases! {
    test_get_hlo_module_content: reached(&["get_hlo_module_content", SESSION, "--fmt=text", "--max_lines=2000"], "Error fetching HLO module content: ");
    test_get_hlo_neighborhood: reached(&["get_hlo_neighborhood", SESSION, "instr_name", "2"], "Error analyzing neighborhood: ");
    test_get_hlo_neighborhood_with_op_name: reached(&["get_hlo_neighborhood", SESSION, "--op_name=instr_name"], "Error analyzing neighborhood: ");
    test_get_hlo_text: reached(&["get_hlo_text", SESSION, "", "module_name", "op_name"], "Error analyzing neighborhood: ");
    test_list_hlo_modules: reached(&["list_hlo_modules", SESSION], "Error listing HLO modules: ");
    test_get_hlo_op_profile: missing(&["get_hlo_op_profile", SESSION, "15"]);
    test_list_xplane_events: missing(&["list_xplane_events", SESSION, "--plane_regex=.*", "--event_regex=.*", "--max_events=100", "--offset=0"]);
    test_aggregate_xplane_events: missing(&["aggregate_xplane_events", SESSION, ".*", ".*"]);
    test_get_xspace_proto: missing(&["get_xspace_proto", SESSION]);
    test_get_overview: missing(&["get_overview", SESSION]);
    test_get_profile_summary: missing(&["get_profile_summary", SESSION]);
    test_get_hosts: missing(&["get_hosts", SESSION]);
    test_get_roofline_model: missing(&["get_roofline_model", SESSION]);
    test_get_kpi_metrics: missing(&["get_kpi_metrics", SESSION]);
}

#[test]
fn test_get_kernel_utilization() {
    let payload = error(&["get_kernel_utilization", SESSION, "--kernel_name=matmul"], 1, "INTERNAL_ERROR");
    assert!(payload.at("error").text().starts_with("Error fetching kernel_utilization.json for session 'session_123'"));
}

#[test]
fn test_upload_trace() {
    let payload = error(&["upload_trace", "/path/to/trace.xplane.pb"], 4, "INVALID_VALUE");
    assert_eq!(payload.at("error").str(), Some("Logdir not set in client. Provide logdir."));
}

#[test]
fn test_main() {
    assert!(crate::cli::execute(&[]).is_none());
    assert!(crate::cli::execute(&strings(&["--logdir", "/tmp"])).is_none());
}

#[test]
fn test_all_tool_modules_registered_in_cli_main() {
    for tool in TOOL_MODULES {
        assert!(COMMANDS.iter().any(|(name, _, _)| *name == tool), "Tool '{tool}' is missing registration in cli_main()!");
    }
}

#[test]
fn test_wrap_with_logdir_preserves_valid_signature_in_oss() {
    for (name, spec, _) in COMMANDS {
        let (_, keyword) = spec.split_once('|').unwrap();
        assert!(keyword.split_whitespace().any(|parameter| parameter == "logdir"), "{name}");
        assert!(spec.split_whitespace().any(|parameter| parameter.trim_end_matches('!') == "bypass_cache"), "{name}");
    }
}

#[test]
fn test_main_fire_usage_error_exit_2() {
    let payload = error(&["get_hlo_op_profile", SESSION, "--top_n=abc"], 2, "USAGE_ERROR");
    assert!(!payload.has("traceback"));
    let dir = scratch("cli-usage");
    std::fs::write(dir.join("host0.xplane.pb"), XSpace::default().encode_to_vec()).unwrap();
    let (status, out, err) = run(&["get_hosts", &dir.to_string_lossy(), "--unknown=1"]);
    assert_eq!((status, out.as_str()), (2, ""));
    assert!(err.starts_with("ERROR: Could not consume arg: --unknown=1"), "{err}");
}

#[test]
fn test_main_file_not_found_exit_3() {
    error(&["get_overview", "non_existent_dir"], 3, "PATH_ERROR");
}

#[test]
fn test_main_permission_error_exit_3() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("cli-permission");
    let trace = dir.join("trace.xplane.pb");
    std::fs::write(&trace, b"").unwrap();
    let locked = dir.join("locked");
    std::fs::create_dir_all(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
    let logdir = format!("--logdir={}", locked.join("logdir").display());
    let outcome = run(&["upload_trace", &trace.to_string_lossy(), &logdir]);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(outcome.0, 3, "{}", outcome.1);
    assert_eq!(J::parse(&outcome.1).unwrap().at("reason").str(), Some("PATH_ERROR"));
    assert!(outcome.2.contains("PATH_ERROR"));
}

#[test]
fn test_main_os_error_disk_full_exit_3() {
    let dir = scratch("cli-oserror");
    let trace = dir.join("trace.xplane.pb");
    std::fs::write(&trace, b"").unwrap();
    std::fs::write(dir.join("not_a_dir"), b"").unwrap();
    error(&["upload_trace", &trace.to_string_lossy(), &format!("--logdir={}", dir.join("not_a_dir").display())], 3, "PATH_ERROR");
}

#[test]
fn test_main_is_a_directory_exit_3() {
    let dir = scratch("cli-directory");
    let payload = error(&["get_kernel_utilization", &dir.to_string_lossy()], 3, "PATH_ERROR");
    assert!(payload.at("error").text().ends_with("(DATA_ABSENT)."));
}

#[test]
fn test_main_value_error_exit_4() {
    error(&["get_hlo_op_profile", "corrupt_dir", "--view=bogus"], 4, "INVALID_VALUE");
}

#[test]
fn test_main_internal_error_exit_1() {
    let (status, out, err) = run(&["get_kernel_utilization", SESSION]);
    assert_eq!(status, 1);
    let payload = J::parse(&out).unwrap();
    assert_eq!(payload.at("reason").str(), Some("INTERNAL_ERROR"));
    assert!(payload.at("error").text().contains(REPORT));
    assert!(payload.at("traceback").text().contains("RuntimeError: Error fetching kernel_utilization.json"));
    assert!(err.contains("INTERNAL_ERROR"));
    assert!(err.contains("RuntimeError: Error fetching kernel_utilization.json"), "{err}");
}

#[test]
fn test_main_upload_trace_value_error_exit_4() {
    let dir = scratch("cli-upload-value");
    std::fs::write(dir.join("trace.txt"), b"").unwrap();
    error(&["upload_trace", &dir.join("trace.txt").to_string_lossy(), &format!("--logdir={}", dir.display())], 4, "INVALID_VALUE");
}

#[test]
fn test_main_upload_trace_file_not_found_exit_3() {
    let dir = scratch("cli-upload-missing");
    error(&["upload_trace", &dir.join("missing.xplane.pb").to_string_lossy(), &format!("--logdir={}", dir.display())], 3, "PATH_ERROR");
}

#[test]
fn test_empty_session_id_raises_value_error() {
    error(&["get_hosts", ""], 4, "INVALID_VALUE");
    error(&["get_hosts", "--session_id="], 4, "INVALID_VALUE");
}

#[test]
fn test_preprocess_argv_quotes_timestamp_tokens() {
    let raw = strings(&["get_kernel_stats", "2026_08_24_06_33_12", "--logdir=/tmp/trace", "--session_id=2026_08_28_05_52_00"]);
    assert_eq!(preprocess(&raw), raw);
    assert_eq!(literal("2026_08_24_06_33_12"), J::from("2026_08_24_06_33_12"));
    assert_eq!(literal("2026_08_28_05_52_00"), J::from("2026_08_28_05_52_00"));
}

#[test]
fn test_preprocess_argv_preserves_numeric_and_standard_flags() {
    let raw = strings(&["list_xplane_events", "sess1", "--limit=10", "-k=5"]);
    assert_eq!(preprocess(&raw), raw);
    assert_eq!(literal("10"), J::Int(10));
}

#[test]
fn test_d25_cli_argument_aliases() {
    let cases: [(&[&str], &[&str]); 4] = [
        (&["get_overview", "--session_dir=/tmp/trace"], &["get_overview", "/tmp/trace"]),
        (&["get_top_hlo_ops", "--session_path=/tmp/trace", "--limit=10"], &["get_top_hlo_ops", "/tmp/trace", "--limit=10"]),
        (&["get_overview", "--source", "/tmp/trace"], &["get_overview", "/tmp/trace"]),
        (&["--session_dir=/tmp/trace", "get_overview"], &["get_overview", "/tmp/trace"]),
    ];
    for (raw, expected) in cases {
        assert_eq!(preprocess(&strings(raw)), strings(expected));
    }
}

#[test]
fn test_d25_cli_argument_aliases_reach_fire() {
    let missing = scratch("cli-alias").join("trace");
    let payload = error(&["get_overview", &format!("--session_dir={}", missing.display())], 3, "PATH_ERROR");
    assert_eq!(payload.at("error").text(), format!("Trace path '{}' does not exist.", missing.display()));
}

#[test]
fn test_d25_cli_argument_aliases_console_script_argv_none() {
    let missing = scratch("cli-alias-first").join("trace");
    let payload = error(&[&format!("--session_dir={}", missing.display()), "get_overview"], 3, "PATH_ERROR");
    assert_eq!(payload.at("error").text(), format!("Trace path '{}' does not exist.", missing.display()));
}

#[test]
fn test_wrap_with_logdir_coerces_int_to_str() {
    let logdir = scratch("cli-coerce");
    let run_dir = logdir.join("plugins/profile/2026_08_24_06_33_12");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("host0.xplane.pb"), XSpace::default().encode_to_vec()).unwrap();
    let (status, out, _) = run(&["get_hosts", "20260824063312", &format!("--logdir={}", logdir.display())]);
    assert_eq!(status, 0, "{out}");
    assert_eq!(J::parse(&out).unwrap().at("hosts").items()[0].at("hostname").str(), Some("host0"));
    assert_eq!(error(&["get_hosts", "20260824063312"], 3, "PATH_ERROR").at("error").str(), Some("Path not found: 20260824063312"));
}
