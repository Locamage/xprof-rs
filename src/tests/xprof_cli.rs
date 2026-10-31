use super::cli_support::{ok, run, scratch};
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
    assert_eq!(
        error(argv, 3, "PATH_ERROR").at("error").str(),
        Some(MISSING)
    );
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
    let payload = error(
        &["get_kernel_utilization", SESSION, "--kernel_name=matmul"],
        1,
        "INTERNAL_ERROR",
    );
    assert!(
        payload
            .at("error")
            .text()
            .starts_with("Error fetching kernel_utilization.json for session 'session_123'")
    );
}

#[test]
fn test_upload_trace() {
    let payload = error(
        &["upload_trace", "/path/to/trace.xplane.pb"],
        4,
        "INVALID_VALUE",
    );
    assert_eq!(
        payload.at("error").str(),
        Some("Logdir not set in client. Provide logdir.")
    );
}

#[test]
fn test_main() {
    assert!(crate::cli::execute(&[]).is_none());
    assert!(crate::cli::execute(&strings(&["--logdir", "/tmp"])).is_none());
}

#[test]
fn test_all_tool_modules_registered_in_cli_main() {
    for tool in TOOL_MODULES {
        assert!(
            COMMANDS.iter().any(|(name, _, _)| *name == tool),
            "Tool '{tool}' is missing registration in cli_main()!"
        );
    }
}

#[test]
fn test_wrap_with_logdir_preserves_valid_signature_in_oss() {
    for (name, spec, _) in COMMANDS {
        let (_, keyword) = spec.split_once('|').unwrap();
        assert!(
            keyword
                .split_whitespace()
                .any(|parameter| parameter == "logdir"),
            "{name}"
        );
        assert!(
            spec.split_whitespace()
                .any(|parameter| parameter.trim_end_matches('!') == "bypass_cache"),
            "{name}"
        );
    }
}

#[test]
fn test_main_fire_usage_error_exit_2() {
    let payload = error(
        &["get_graph_viewer", SESSION, "--op_profile_limit=abc"],
        2,
        "USAGE_ERROR",
    );
    assert!(!payload.has("traceback"));
    error(
        &["get_hlo_op_profile", SESSION, "--top_n=abc"],
        3,
        "PATH_ERROR",
    );
    let dir = scratch("cli-usage");
    std::fs::write(
        dir.join("host0.xplane.pb"),
        XSpace::default().encode_to_vec(),
    )
    .unwrap();
    let (status, out, err) = run(&["get_hosts", &dir.to_string_lossy(), "--unknown=1"]);
    assert_eq!((status, out.as_str()), (2, ""));
    assert!(
        err.starts_with("ERROR: Could not consume arg: --unknown=1"),
        "{err}"
    );
}

#[test]
fn test_run_dir_of_bare_file_name_is_current_dir() {
    use crate::cli::client::{Client, Local};
    assert_eq!(
        Local::default().run_dir("Cargo.toml").unwrap(),
        std::path::Path::new(".")
    );
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
    assert_eq!(
        J::parse(&outcome.1).unwrap().at("reason").str(),
        Some("PATH_ERROR")
    );
    assert!(outcome.2.contains("PATH_ERROR"));
}

#[test]
fn test_main_os_error_disk_full_exit_3() {
    let dir = scratch("cli-oserror");
    let trace = dir.join("trace.xplane.pb");
    std::fs::write(&trace, b"").unwrap();
    std::fs::write(dir.join("not_a_dir"), b"").unwrap();
    error(
        &[
            "upload_trace",
            &trace.to_string_lossy(),
            &format!("--logdir={}", dir.join("not_a_dir").display()),
        ],
        3,
        "PATH_ERROR",
    );
}

#[test]
fn test_main_is_a_directory_exit_3() {
    let dir = scratch("cli-directory");
    let payload = error(
        &["get_kernel_utilization", &dir.to_string_lossy()],
        3,
        "PATH_ERROR",
    );
    assert!(payload.at("error").text().ends_with("(DATA_ABSENT)."));
}

#[test]
fn test_main_value_error_exit_4() {
    error(
        &["get_hlo_op_profile", "corrupt_dir", "--view=bogus"],
        4,
        "INVALID_VALUE",
    );
}

#[test]
fn test_main_internal_error_exit_1() {
    let (status, out, err) = run(&["get_kernel_utilization", SESSION]);
    assert_eq!(status, 1);
    let payload = J::parse(&out).unwrap();
    assert_eq!(payload.at("reason").str(), Some("INTERNAL_ERROR"));
    assert!(payload.at("error").text().contains(REPORT));
    assert!(
        payload
            .at("traceback")
            .text()
            .contains("RuntimeError: Error fetching kernel_utilization.json")
    );
    assert!(err.contains("INTERNAL_ERROR"));
    assert!(
        err.contains("RuntimeError: Error fetching kernel_utilization.json"),
        "{err}"
    );
}

#[test]
fn test_main_upload_trace_value_error_exit_4() {
    let dir = scratch("cli-upload-value");
    std::fs::write(dir.join("trace.txt"), b"").unwrap();
    error(
        &[
            "upload_trace",
            &dir.join("trace.txt").to_string_lossy(),
            &format!("--logdir={}", dir.display()),
        ],
        4,
        "INVALID_VALUE",
    );
}

#[test]
fn test_main_upload_trace_file_not_found_exit_3() {
    let dir = scratch("cli-upload-missing");
    error(
        &[
            "upload_trace",
            &dir.join("missing.xplane.pb").to_string_lossy(),
            &format!("--logdir={}", dir.display()),
        ],
        3,
        "PATH_ERROR",
    );
}

#[test]
fn test_empty_session_id_raises_value_error() {
    error(&["get_hosts", ""], 4, "INVALID_VALUE");
    error(&["get_hosts", "--session_id="], 4, "INVALID_VALUE");
}

#[test]
fn test_preprocess_argv_quotes_timestamp_tokens() {
    let raw = strings(&[
        "get_kernel_stats",
        "2026_08_24_06_33_12",
        "--logdir=/tmp/trace",
        "--session_id=2026_08_28_05_52_00",
    ]);
    assert_eq!(preprocess(&raw), raw);
    assert_eq!(
        literal("2026_08_24_06_33_12"),
        J::from("2026_08_24_06_33_12")
    );
    assert_eq!(
        literal("2026_08_28_05_52_00"),
        J::from("2026_08_28_05_52_00")
    );
}

#[test]
fn test_preprocess_argv_preserves_numeric_and_standard_flags() {
    let raw = strings(&["list_xplane_events", "sess1", "--limit=10", "-k=5"]);
    assert_eq!(preprocess(&raw), raw);
    assert_eq!(literal("10"), J::Int(10));
}

#[test]
fn test_literal_keeps_tuples_sets_and_dicts_apart() {
    let cases = [
        ("(1,)", "(1,)"),
        ("1,2", "(1, 2)"),
        ("(1,[2])", "(1, [2])"),
        ("{1,2,1}", "{1, 2}"),
        ("{a:1,b:x,a:2}", "{'a': 2, 'b': 'x'}"),
        ("{}", "{}"),
        ("[a,b]", "['a', 'b']"),
    ];
    for (text, shown) in cases {
        assert_eq!(literal(text).text(), shown, "{text}");
    }
    assert_eq!(literal("{a:1}").compact(), "{\"a\": 1}");
    assert_eq!(literal("{1,2}").compact(), "[1, 2]");
    assert!(!literal("{}").truthy() && literal("(1,)").truthy());
    assert!(
        J::Bool(true).equals(&J::Int(1))
            && J::Int(1).equals(&J::Float(1.0))
            && !J::from("1").equals(&J::Int(1))
    );
}

#[test]
fn test_long_lists_write_as_one_list() {
    let items: Vec<J> = (0..10_000)
        .map(|index| {
            if index % 3 == 0 {
                J::from(format!("é{index}"))
            } else {
                J::Float(f64::from(index) / 7.0)
            }
        })
        .collect();
    let indented: Vec<String> = items
        .iter()
        .map(|item| format!("\n  {}", item.dumps()))
        .collect();
    let compact: Vec<String> = items.iter().map(J::compact).collect();
    let list = J::List(items);
    assert_eq!(list.dumps(), format!("[{}\n]", indented.join(",")));
    assert_eq!(list.compact(), format!("[{}]", compact.join(", ")));
}

#[test]
fn test_packed_options_follow_the_native_converter() {
    use crate::cli::packed;
    assert_eq!(
        packed("type", &J::Bool(true)).unwrap().as_deref(),
        Some("true")
    );
    assert_eq!(packed("type", &J::Int(3)).unwrap().as_deref(), Some("3"));
    assert_eq!(packed("type", &literal("[1]")).unwrap(), None);
    assert_eq!(packed("type", &J::Null).unwrap(), None);
    assert_eq!(
        packed("type", &J::Float(1.5)).unwrap_err().message,
        "Unsupported option type for key 'type': <class 'float'>"
    );
    assert_eq!(
        packed("tag", &literal("{1,2}")).unwrap_err().message,
        "Unsupported option type for key 'tag': <class 'set'>"
    );
}

#[test]
fn test_d25_cli_argument_aliases() {
    let cases: [(&[&str], &[&str]); 4] = [
        (
            &["get_overview", "--session_dir=/tmp/trace"],
            &["get_overview", "/tmp/trace"],
        ),
        (
            &["get_top_hlo_ops", "--session_path=/tmp/trace", "--limit=10"],
            &["get_top_hlo_ops", "/tmp/trace", "--limit=10"],
        ),
        (
            &["get_overview", "--source", "/tmp/trace"],
            &["get_overview", "/tmp/trace"],
        ),
        (
            &["--session_dir=/tmp/trace", "get_overview"],
            &["get_overview", "/tmp/trace"],
        ),
    ];
    for (raw, expected) in cases {
        assert_eq!(preprocess(&strings(raw)), strings(expected));
    }
}

#[test]
fn test_d25_cli_argument_aliases_reach_fire() {
    let missing = scratch("cli-alias").join("trace");
    let payload = error(
        &[
            "get_overview",
            &format!("--session_dir={}", missing.display()),
        ],
        3,
        "PATH_ERROR",
    );
    assert_eq!(
        payload.at("error").text(),
        format!("Trace path '{}' does not exist.", missing.display())
    );
}

#[test]
fn test_d25_cli_argument_aliases_console_script_argv_none() {
    let missing = scratch("cli-alias-first").join("trace");
    let payload = error(
        &[
            &format!("--session_dir={}", missing.display()),
            "get_overview",
        ],
        3,
        "PATH_ERROR",
    );
    assert_eq!(
        payload.at("error").text(),
        format!("Trace path '{}' does not exist.", missing.display())
    );
}

#[test]
fn test_wrap_with_logdir_coerces_int_to_str() {
    let logdir = scratch("cli-coerce");
    let run_dir = logdir.join("plugins/profile/2026_08_24_06_33_12");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(
        run_dir.join("host0.xplane.pb"),
        XSpace::default().encode_to_vec(),
    )
    .unwrap();
    let (status, out, _) = run(&[
        "get_hosts",
        "20260824063312",
        &format!("--logdir={}", logdir.display()),
    ]);
    assert_eq!(status, 0, "{out}");
    assert_eq!(
        J::parse(&out).unwrap().at("hosts").items()[0]
            .at("hostname")
            .str(),
        Some("host0")
    );
    assert_eq!(
        error(&["get_hosts", "20260824063312"], 3, "PATH_ERROR")
            .at("error")
            .str(),
        Some("Path not found: 20260824063312")
    );
}

#[test]
fn test_multi_host_logdir_session_remote_and_help() {
    let demo = std::fs::read(super::e2e_oracles::DEMO).unwrap();
    let logdir = scratch("multi-host");
    for (session, host) in [("a", "h0"), ("a", "h1"), ("b", "h0")] {
        let dir = logdir.join("plugins/profile").join(session);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{host}.xplane.pb")), &demo).unwrap();
    }
    let xspace_dir = logdir.join("plugins/profile/xspace");
    std::fs::create_dir_all(&xspace_dir).unwrap();
    for host in ["h0", "h1"] {
        std::fs::write(xspace_dir.join(format!("{host}.xspace.pb")), &demo).unwrap();
    }
    let spill_session = format!("xspace-spill-{}", std::process::id());
    let spill_dir = logdir.join("plugins/profile").join(&spill_session);
    std::fs::create_dir_all(&spill_dir).unwrap();
    for host in ["h0", "h1"] {
        std::fs::File::create(spill_dir.join(format!("{host}.xspace.pb")))
            .unwrap()
            .set_len((10 * 1024 * 1024 + 1) as u64)
            .unwrap();
    }
    let single_host = logdir.join("plugins/profile/a-single");
    std::fs::create_dir_all(&single_host).unwrap();
    std::fs::copy(
        logdir.join("plugins/profile/a/h0.xplane.pb"),
        single_host.join("h0.xplane.pb"),
    )
    .unwrap();
    let logdir = logdir.to_string_lossy();
    let a = format!("{logdir}/plugins/profile/a");
    let b = format!("{logdir}/plugins/profile/b");
    let b_trace = format!("{b}/h0.xplane.pb");
    let (code, memory_out, _) = run(&[
        "get_memory_profile",
        "a",
        &format!("--logdir={logdir}"),
        "--bypass_cache=True",
    ]);
    assert_eq!(code, 0, "{memory_out}");
    let (single_code, single_memory_out, _) = run(&[
        "get_memory_profile",
        "a-single",
        &format!("--logdir={logdir}"),
        "--bypass_cache=True",
    ]);
    assert_eq!(single_code, 0, "{single_memory_out}");
    assert_eq!(memory_out, single_memory_out);
    let memory = J::parse(&memory_out).unwrap();
    let (code, kpi_out, _) = run(&["get_kpi_metrics", "a", &format!("--logdir={logdir}")]);
    assert_eq!(code, 0, "{kpi_out}");
    let (single_code, single_kpi_out, _) =
        run(&["get_kpi_metrics", "a-single", &format!("--logdir={logdir}")]);
    assert_eq!(single_code, 0, "{single_kpi_out}");
    let kpi = J::parse(&kpi_out).unwrap();
    let single_kpi = J::parse(&single_kpi_out).unwrap();
    if memory
        .at("peak_memory_usage_gib")
        .float()
        .is_some_and(|peak| peak > 0.0)
    {
        assert!(kpi.at("peak_hbm_gib").float().is_some(), "{kpi_out}");
    } else {
        assert_eq!(kpi.at("peak_hbm_gib"), single_kpi.at("peak_hbm_gib"));
    }
    let (code, out, _) = run(&["get_xspace_proto", "a", &format!("--logdir={logdir}")]);
    assert_ne!(code, 0, "{out}");
    let error = J::parse(&out).unwrap().at("error").text();
    assert!(
        error.contains("Session 'a' has 2 hosts: h0, h1. Pass --host=<name>."),
        "{out}"
    );
    let output = scratch("multi-host-xspace").join("h0.xplane.pb");
    let (code, out, _) = run(&[
        "get_xspace_proto",
        "a",
        "--host=h0",
        &format!("--output_path={}", output.display()),
        &format!("--logdir={logdir}"),
    ]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(
        std::fs::read(output).unwrap(),
        std::fs::read(format!("{a}/h0.xplane.pb")).unwrap()
    );
    let (code, out, _) = run(&["get_hosts", "xspace", &format!("--logdir={logdir}")]);
    assert_eq!(code, 0, "{out}");
    let hosts: Vec<String> = J::parse(&out)
        .unwrap()
        .at("hosts")
        .items()
        .iter()
        .map(|host| host.at("hostname").text())
        .collect();
    assert_eq!(hosts, ["h0", "h1"]);
    let output = scratch("multi-host-xspace-suffix").join("h1.xspace.pb");
    let (code, out, _) = run(&[
        "get_xspace_proto",
        "xspace",
        "--host=h1",
        &format!("--output_path={}", output.display()),
        &format!("--logdir={logdir}"),
    ]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(std::fs::read(output).unwrap(), demo);
    let (code, out, _) = run(&[
        "get_xspace_proto",
        &spill_session,
        "--host=h1",
        &format!("--logdir={logdir}"),
    ]);
    assert_eq!(code, 0, "{out}");
    let spill_file = J::parse(&out).unwrap().at("file_path").text();
    assert_eq!(spill_file, format!("/tmp/xspace_{spill_session}.h1.pb"));
    std::fs::remove_file(spill_file).unwrap();
    for command in ["compute_utilization", "get_kernel_utilization"] {
        let (code, out, _) = run(&[command, "b", &format!("--logdir={logdir}")]);
        assert_eq!(code, 0, "{command}: {out}");
        assert_eq!(out, "{\n  \"devices\": [],\n  \"status\": \"SUCCESS\"\n}\n");
    }
    for command in [
        "get_overview",
        "get_kpi_metrics",
        "get_top_hlo_ops",
        "check_host_boundness",
    ] {
        let (code, out, _) = run(&[command, &a]);
        assert_eq!(code, 0, "{out}");
        assert!(
            !out.contains("No overview data") && !out.contains("No HLO op_profile"),
            "{out}"
        );
    }
    let overview = run(&["get_overview", &a]).1;
    assert_eq!(run(&["get_overview", &logdir, "a"]).1, overview);
    assert_ne!(run(&["get_overview", &logdir, "b"]).1, overview);
    let (expected_code, expected_out, _) =
        run(&["get_overview", &logdir, "--include_command=True"]);
    let (code, out, _) = run(&["get_overview", &logdir, "True"]);
    assert_eq!((code, out), (expected_code, expected_out));
    assert_eq!(run(&["get_overview", &format!("file://{a}")]).1, overview);
    assert_eq!(
        run(&["get_overview", &format!("file://{logdir}"), "a"]).1,
        overview
    );
    for command in ["get_avg_step_time", "get_kernel_stats"] {
        for source in [&b, &b_trace] {
            let (expected_code, expected_out, _) = run(&[command, source]);
            assert_eq!(expected_code, 0, "{command} {source}: {expected_out}");
            let remote = format!("file://{source}");
            let (code, out, _) = run(&[command, &remote]);
            assert_eq!(
                (code, out),
                (expected_code, expected_out),
                "{command} {remote}"
            );
        }
        let (expected_code, expected_out, _) = run(&[command, &a]);
        assert_eq!(expected_code, 0, "{command} {a}: {expected_out}");
        let remote_logdir = format!("--logdir=file://{logdir}");
        let (code, out, _) = run(&[command, "a", &remote_logdir]);
        assert_eq!(code, 0, "{command}: {out}");
        assert_eq!(
            (code, out),
            (expected_code, expected_out),
            "{command} {remote_logdir}"
        );
    }
    let help = run(&["aggregate_xplane_events", "--help"]).1;
    assert!(
        help.contains("POSITIONAL ARGUMENTS\n    SESSION_ID\n")
            && help.contains("    --plane_regex=PLANE_REGEX\n"),
        "{help}"
    );
}

#[test]
fn get_tool_data_matches_server_responses_for_cpu_and_gpu_traces() {
    use crate::cli::client::Client;
    use std::io::Read;
    let gpu = include_bytes!("../../tests/data/gpu/xplane/g1_basic/gpuhost.xplane.pb.gz");
    let mut gpu_trace = Vec::new();
    flate2::read::GzDecoder::new(&gpu[..])
        .read_to_end(&mut gpu_trace)
        .unwrap();
    let fixtures = [
        (
            "demo",
            include_bytes!("../../tests/data/demo.xplane.pb").to_vec(),
        ),
        ("gpu", gpu_trace),
    ];
    for (name, trace) in fixtures {
        let logdir = scratch(&format!("tool-data-{name}"));
        let session = logdir.join("plugins/profile/s");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("host.xplane.pb"), &trace).unwrap();
        let server_state = crate::state(&crate::Settings {
            logdir: logdir.clone(),
            ..Default::default()
        });
        for tool in [
            "overview_page",
            "kernel_stats",
            "framework_op_stats",
            "op_profile",
            "hlo_stats",
            "memory_profile",
            "input_pipeline_analyzer",
            "trace_viewer@",
        ] {
            let mut query = url::form_urlencoded::Serializer::new(String::new());
            query.append_pair("run", "s").append_pair("tag", tool);
            let uri = format!("/data?{}", query.finish());
            let expected = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(super::legacy::fetch(&server_state, &uri));
            assert_eq!(
                expected.0,
                axum::http::StatusCode::OK,
                "{name} {tool}: {}",
                expected.2
            );
            let output = logdir.join(format!("{tool}.out"));
            let argv = [
                "get_tool_data".to_string(),
                "s".to_string(),
                format!("--tool={tool}"),
                format!("--output_path={}", output.display()),
                format!("--logdir={}", logdir.display()),
            ];
            let (code, result, error) = crate::cli::execute(&argv).unwrap();
            assert_eq!(
                code,
                0,
                "{name} {tool}: {} {error}",
                String::from_utf8_lossy(&result)
            );
            let actual = std::fs::read(output).unwrap();
            assert_eq!(actual, expected.2.as_bytes(), "{name} {tool}");
            if tool == "trace_viewer@" {
                assert!(String::from_utf8_lossy(&actual).contains("\"returnedEventsSize\""));
            }
        }
    }
    let expected = crate::cli::client::Local::default()
        .fetch("unknown", "s", &[])
        .unwrap_err();
    let argv = [
        "get_tool_data".to_string(),
        "s".to_string(),
        "--tool=unknown".to_string(),
    ];
    let (code, out, _) = crate::cli::execute(&argv).unwrap();
    assert_ne!(code, 0);
    assert_eq!(
        J::parse(&String::from_utf8(out).unwrap())
            .unwrap()
            .at("error")
            .text(),
        expected.message
    );
}

#[test]
fn test_remote_store_root_forms_use_latest_session_without_walking_other_runs() {
    let demo = std::fs::read(super::e2e_oracles::DEMO).unwrap();
    let root = scratch("remote-store-root");
    for (run_name, session, hosts) in [
        ("runA", "s1", &["h0", "h1"][..]),
        ("runA", "s2", &["h0"][..]),
        ("runB", "s1", &["h0"][..]),
    ] {
        let dir = root.join(run_name).join("plugins/profile").join(session);
        std::fs::create_dir_all(&dir).unwrap();
        for host in hosts {
            std::fs::write(dir.join(format!("{host}.xplane.pb")), &demo).unwrap();
        }
    }
    let run_a = root.join("runA");
    let session = run_a.join("plugins/profile/s1");
    let root_url = format!("file://{}", root.display());
    let run_url = format!("file://{}", run_a.display());
    let local_run = run_a.display().to_string();
    let local_session = session.display().to_string();
    let cases = [
        (
            vec![
                "get_overview".into(),
                "runA".into(),
                format!("--logdir={root_url}"),
            ],
            vec!["get_overview".into(), local_run.clone()],
        ),
        (
            vec![
                "get_overview".into(),
                "runA/plugins/profile/s1".into(),
                format!("--logdir={root_url}"),
            ],
            vec!["get_overview".into(), local_session.clone()],
        ),
        (
            vec![
                "get_overview".into(),
                format!("{run_url}/plugins/profile/s1"),
            ],
            vec!["get_overview".into(), local_session],
        ),
        (
            vec!["get_overview".into(), run_url.clone(), "s1".into()],
            vec!["get_overview".into(), local_run.clone(), "s1".into()],
        ),
        (
            vec![
                "get_overview".into(),
                "s1".into(),
                format!("--logdir={run_url}"),
            ],
            vec![
                "get_overview".into(),
                "s1".into(),
                format!("--logdir={local_run}"),
            ],
        ),
        (
            vec!["get_overview".into(), run_url],
            vec!["get_overview".into(), local_run],
        ),
    ];
    for (remote, local) in cases {
        let execute = |args: &[String]| {
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            let (code, out, err) = run(&args);
            assert_eq!(code, 0, "{args:?}: {out} {err}");
            out
        };
        assert_eq!(execute(&remote), execute(&local), "{remote:?}");
    }
    let remote = crate::server::remote::Remote::open(&root_url).unwrap();
    assert!(
        !remote.mirror.join("runB").exists(),
        "whole store was mirrored at {}",
        remote.mirror.display()
    );
}

#[test]
fn test_remote_spellings_share_one_cached_copy() {
    use std::os::unix::fs::MetadataExt;
    let demo = std::fs::read(super::e2e_oracles::DEMO).unwrap();
    let root = scratch("remote-cache-identity");
    let session = root.join("runA/plugins/profile/s1");
    std::fs::create_dir_all(&session).unwrap();
    std::fs::write(session.join("h0.xplane.pb"), &demo).unwrap();
    let root_url = format!("file://{}", root.display());
    let run_url = format!("{root_url}/runA");
    let remote = crate::server::remote::Remote::open(&root_url).unwrap();
    let cached = remote.mirror.join("runA/plugins/profile/s1/h0.xplane.pb");
    std::fs::remove_dir_all(&remote.mirror).unwrap();
    let snapshot = || {
        let metadata = std::fs::metadata(&cached).unwrap();
        (metadata.ino(), metadata.mtime(), metadata.mtime_nsec())
    };
    let mut first_snapshot = None;
    for args in [
        vec![format!("{run_url}/plugins/profile/s1")],
        vec![run_url.clone(), "s1".into()],
        vec!["s1".into(), format!("--logdir={run_url}")],
    ] {
        let mut command = vec!["get_overview".to_string()];
        command.extend(args);
        let args: Vec<&str> = command.iter().map(String::as_str).collect();
        let (code, out, err) = run(&args);
        assert_eq!(code, 0, "{args:?}: {out} {err}");
        let current = snapshot();
        if let Some(first) = first_snapshot {
            assert_eq!(current, first, "{args:?}");
        } else {
            first_snapshot = Some(current);
        }
    }
    let server_remote = crate::server::remote::Remote::open(&root_url).unwrap();
    assert_eq!(
        server_remote
            .mirror
            .join("runA/plugins/profile/s1/h0.xplane.pb"),
        cached
    );
}

#[test]
fn test_events_outside_the_capture_are_flagged() {
    let demo = super::e2e_oracles::DEMO;
    // The capture of the demo lasts this long. Its last host events end after it.
    let capture_ps = (1_790_796_284_353_597_317i64 - 1_790_796_283_134_782_597) as f64 * 1000.0;
    let listed = ok(&["list_xplane_events", demo, "--max_events=100000"]);
    let events = listed.at("events").items();
    let (start, duration) = (
        |event: &J| event.at("offset_ps").float().unwrap(),
        |event: &J| event.at("duration_ps").float().unwrap(),
    );
    let inside = events
        .iter()
        .filter(|event| start(event) >= 0.0 && start(event) + duration(event) <= capture_ps)
        .count();
    assert!(inside < events.len());
    let mut listed_counts = std::collections::BTreeMap::new();
    for event in events {
        *listed_counts
            .entry(event.at("event").text())
            .or_insert(0usize) += 1;
    }
    let aggregate = ok(&["aggregate_xplane_events", demo]);
    let aggregates = aggregate.at("aggregates").items();
    let aggregate_counts: std::collections::BTreeMap<_, _> = aggregates
        .iter()
        .map(|aggregate| {
            (
                aggregate.at("event").text(),
                aggregate.at("count").int().unwrap() as usize,
            )
        })
        .collect();
    assert_eq!(aggregate_counts, listed_counts);
    let cut: usize = aggregates
        .iter()
        .map(|aggregate| aggregate.at("truncated_by_capture").int().unwrap_or(0) as usize)
        .sum();
    assert_eq!(cut, events.len() - inside);
    assert!(cut > 0);
    assert_eq!(
        run(&["get_avg_step_time", demo]).1,
        "{\n  \"avg_step_time_ms\": 1.7486,\n  \"step_count\": 40\n}\n"
    );
}

#[test]
fn test_capture_boundaries_flag_exact_step_and_event_aggregates() {
    const STEP: &str = "jit_step(11448588407652663271)";
    const GRPC: &str = "GrpcSystemConfigTrace";
    let demo = super::e2e_oracles::DEMO;
    let original = XSpace::decode(&std::fs::read(demo).unwrap()[..]).unwrap();
    let capture_time = |space: &XSpace, name: &str| {
        let plane = space
            .planes
            .iter()
            .find(|plane| plane.name == "Task Environment")
            .unwrap();
        let id = plane
            .stat_metadata
            .values()
            .find(|metadata| metadata.name == name)
            .unwrap()
            .id;
        let stat = plane
            .stats
            .iter()
            .find(|stat| stat.metadata_id == id)
            .unwrap();
        stat.uint64_value
            .map(|value| value as i64)
            .or(stat.int64_value)
            .unwrap()
    };
    let start = capture_time(&original, "profile_start_time");
    let (mut first_start_ps, mut last_end_ps) = (i128::MAX, i128::MIN);
    for plane in &original.planes {
        if !plane.name.starts_with("/device:") {
            continue;
        }
        for line in plane
            .lines
            .iter()
            .filter(|line| line.name.eq_ignore_ascii_case("XLA Modules"))
        {
            for event in &line.events {
                if plane
                    .event_metadata
                    .get(&event.metadata_id)
                    .is_some_and(|metadata| metadata.name == STEP)
                {
                    let offset = event.offset_ps.unwrap_or_default() as i128;
                    let base = line.timestamp_ns as i128 * 1000
                        + if line.timestamp_ns <= start / 2 {
                            start as i128 * 1000
                        } else {
                            0
                        };
                    first_start_ps = first_start_ps.min(base + offset);
                    last_end_ps = last_end_ps.max(base + offset + event.duration_ps as i128);
                }
            }
        }
    }
    let start_after_first = i64::try_from(first_start_ps.div_euclid(1000) + 1).unwrap();
    let stop_before_last = i64::try_from((last_end_ps - 1).div_euclid(1000)).unwrap();
    let baseline_step = ok(&["get_avg_step_time", demo]);
    let baseline_aggregate = ok(&["aggregate_xplane_events", demo]);
    let base_step = baseline_aggregate
        .at("aggregates")
        .items()
        .iter()
        .find(|aggregate| aggregate.at("event").text() == STEP)
        .unwrap();
    let base_grpc = baseline_aggregate
        .at("aggregates")
        .items()
        .iter()
        .find(|aggregate| aggregate.at("event").text() == GRPC)
        .unwrap();
    assert_eq!(baseline_step.at("step_count").int(), Some(40));
    assert!(baseline_step.get("truncated_by_capture").is_none());
    assert_eq!(base_step.at("count").int(), Some(40));
    assert!(base_step.get("truncated_by_capture").is_none());
    assert!(base_grpc.get("truncated_by_capture").is_none());

    let mut start_space = original.clone();
    start_space
        .planes
        .iter_mut()
        .find(|plane| plane.name == "Task Environment")
        .unwrap()
        .add_stat("profile_start_time", start_after_first);
    let start_path = scratch("capture-window-start").join("demo.xplane.pb");
    std::fs::write(&start_path, start_space.encode_to_vec()).unwrap();
    let start_path = start_path.to_string_lossy();
    let start_step = ok(&["get_avg_step_time", &start_path]);
    let start_aggregate = ok(&["aggregate_xplane_events", &start_path]);
    let start_jit = start_aggregate
        .at("aggregates")
        .items()
        .iter()
        .find(|aggregate| aggregate.at("event").text() == STEP)
        .unwrap();
    let start_grpc = start_aggregate
        .at("aggregates")
        .items()
        .iter()
        .find(|aggregate| aggregate.at("event").text() == GRPC)
        .unwrap();
    assert_eq!(start_step.at("step_count").int(), Some(40));
    assert!(start_step.get("truncated_by_capture").is_none());
    assert_eq!(start_jit.at("count").int(), Some(40));
    assert!(start_jit.get("truncated_by_capture").is_none());
    assert_eq!(start_grpc.at("count").int(), base_grpc.at("count").int());
    assert_eq!(start_grpc.at("truncated_by_capture").int(), Some(1));

    let mut stop_space = original;
    stop_space
        .planes
        .iter_mut()
        .find(|plane| plane.name == "Task Environment")
        .unwrap()
        .add_stat("profile_stop_time", stop_before_last);
    let stop_path = scratch("capture-window-stop").join("demo.xplane.pb");
    std::fs::write(&stop_path, stop_space.encode_to_vec()).unwrap();
    let stop_path = stop_path.to_string_lossy();
    let stop_step = ok(&["get_avg_step_time", &stop_path]);
    let stop_aggregate = ok(&["aggregate_xplane_events", &stop_path]);
    let stop_jit = stop_aggregate
        .at("aggregates")
        .items()
        .iter()
        .find(|aggregate| aggregate.at("event").text() == STEP)
        .unwrap();
    assert_eq!(stop_step.at("step_count").int(), Some(40));
    assert_eq!(stop_step.at("truncated_by_capture").int(), Some(1));
    assert_eq!(stop_jit.at("count").int(), Some(40));
    assert_eq!(stop_jit.at("truncated_by_capture").int(), Some(1));
}
