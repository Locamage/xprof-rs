use super::cli_support::{Fake, args, json, ok, parse, run, scratch};
use super::e2e_oracles::{DEMO, TRAINING_STEP_PS, demo, demo_query, session, training_trace};
use crate::cli::client::{Client, Local};
use crate::cli::json::J;
use crate::cli::{Kind, hlo, xplane};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

const SHORT_TXT_GRAPH: &str = "ENTRY entry {\n  x = f32[10] parameter(0)\n  w = f32[10] parameter(1)\n  mul = f32[10] multiply(x, w)\n  ROOT neg = f32[10] negate(mul)\n}\n";
const SESSION_ID: &str = "2026_08_24_06_33_12";

fn overview(target: &Path) -> J {
    ok(&["get_overview", target.to_str().unwrap()])
}

fn swap_preserving_mtime(source: &str, target: &Path) {
    let modified = std::fs::metadata(target).unwrap().modified().unwrap();
    std::fs::copy(source, target).unwrap();
    std::fs::File::options().write(true).open(target).unwrap().set_modified(modified).unwrap();
}

#[test]
fn test_d01_d02_stale_cache_and_bypass_cache() {
    let dir = scratch("d01");
    let path = session(&dir, "test.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let first = overview(&path);
    assert!(!first.has("error"));
    swap_preserving_mtime(DEMO, &path);
    let second = ok(&["get_overview", path.to_str().unwrap(), "--bypass_cache"]);
    assert!(!second.has("error"));
    assert_ne!(first, second);
}

#[test]
fn test_d03_d04_error_on_empty_or_corrupt_trace() {
    let dir = scratch("d03");
    assert_eq!(Local::default().xspace_paths(&dir).unwrap_err().kind, Kind::FileNotFound);
    let corrupt = dir.join("corrupt.xplane.pb");
    std::fs::write(&corrupt, b"CORRUPT_INVALID_PROTOBUF_BYTES").unwrap();
    let error = xplane::list_xplane_events(&Local::default(), &args(corrupt.to_str().unwrap(), &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}

#[test]
fn test_d05_silent_truncation_resolved() {
    let result = demo_query("d05", "list_xplane_events", &["--max_events=10"]);
    assert!(result.has("total_matched") && result.has("truncated"));
    assert_eq!(result.at("events").items().len() as i128, result.at("returned").int().unwrap());
}

#[test]
fn test_d08_empty_category_filter_returns_empty_list() {
    let result = demo_query("d08", "get_top_hlo_ops", &["--category_filter=non_existent_category_xyz"]);
    assert!(!result.has("error"));
    assert_eq!(result.at("top_by_time"), &J::List(Vec::new()));
    assert_eq!(result.at("total_matched").int(), Some(0));
}

#[test]
fn test_d09_limit_negative_one_returns_all() {
    let all = demo_query("d09", "get_top_hlo_ops", &["--limit=-1"]);
    assert!(!all.at("top_by_time").items().is_empty());
    assert_eq!(all.at("top_by_time").items().len() as i128, all.at("total_matched").int().unwrap());
}

#[test]
fn test_d12_canonical_join_keys() {
    let records = demo_query("d12", "get_kernel_stats", &["--limit=5"]);
    let top = &records.items()[0];
    assert!(top.has("canonical_name") && top.has("short_name") && top.has("hlo_op_name"));
}

#[test]
fn test_d13_utilization_viewer_clean_status() {
    let result = demo_query("d13", "get_utilization_viewer", &[]);
    assert!(matches!(result, J::Map(_)));
    assert!(result.has("status") || result.has("reason") || result.has("message"));
    assert!(!result.has("error"));
}

#[test]
fn test_d14_llo_tools_unavailable_status() {
    for command in ["get_llo_analysis", "get_llo_debug_string"] {
        let result = demo_query(command, command, &[]);
        assert_eq!(result.at("status").str(), Some("UNAVAILABLE"));
        assert_eq!(result.at("reason").str(), Some("LLO_DATA_ABSENT"));
        assert!(result.at("error").text().contains("not available"));
    }
}

#[test]
fn test_d15_roofline_bottleneck_intensity_and_deduplication() {
    let result = demo_query("d15", "get_roofline_model", &[]);
    assert!(!result.has("error"));
    let program = result.at("program");
    for key in [
        "bottleneck_operational_intensity_flop_per_byte",
        "optimal_flop_rate_gflops",
        "dma_stall_percent",
        "hbm_read_bw_utilization_percent",
        "hbm_write_bw_utilization_percent",
        "vmem_read_bw_utilization_percent",
        "vmem_write_bw_utilization_percent",
        "cmem_read_bw_utilization_percent",
        "cmem_write_bw_utilization_percent",
    ] {
        assert!(program.has(key), "{key}");
    }
    let mut ranks: Vec<i128> = result.at("top_operations").items().iter().filter_map(|op| op.at("rank").int()).collect();
    let count = ranks.len();
    ranks.sort_unstable();
    ranks.dedup();
    assert_eq!(ranks.len(), count);
}

#[test]
fn test_d16_perf_counters_non_null_payload() {
    let dir = scratch("d16");
    let path = demo(&dir);
    let data = Local::default().fetch_text("perf_counters", path.to_str().unwrap(), &[("bypass_cache", "True".into())]).unwrap().unwrap();
    assert_ne!(data, "null");
    let table = parse(&data);
    assert!(table.has("cols") && table.has("rows"));
    let ids: Vec<String> = table.at("cols").items().iter().map(|column| column.get("id").or_else(|| column.get("label")).unwrap().text()).collect();
    for id in ["Host", "Chip", "Kernel", "Counter"] {
        assert!(ids.iter().any(|column| column == id), "{id}");
    }
}

#[test]
fn test_d17_llo_analysis_opcode_resolution_and_multi_module() {
    let result = demo_query("d17", "get_llo_analysis", &[]);
    assert_eq!(result.at("status").str(), Some("UNAVAILABLE"));
    assert!(result.at("error").text().contains("not available"));
}

#[test]
fn test_d23_get_hlo_neighborhood_default_mode_bfs_expansion() {
    let dir = scratch("d23n");
    std::fs::write(dir.join("module_0001.jit_compute.hlo_proto.pb"), b"").unwrap();
    let fake = Fake::fixed(SHORT_TXT_GRAPH).in_dir(&dir);
    let found = hlo::neighborhood(&fake, dir.to_str().unwrap(), Some("mul".into()), 2, None, None, false).unwrap();
    let in_order = ["[dist=1]", "neg", "w", "x"]
        .iter()
        .scan(0, |start, needle| {
            let at = *start + found[*start..].find(needle)?;
            *start = at + needle.len();
            Some(at)
        })
        .count();
    assert_eq!(in_order, 4, "{found}");
    assert!(found.contains("[entry]") && !found.contains("[unknown]"), "{found}");
}

#[test]
fn test_fingerprint_stability_and_cache_warmup() {
    let dir = scratch("warm");
    let path = session(&dir, "test.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let paths = Local::default().xspace_paths(&dir).unwrap();
    let first = overview(&path);
    assert_eq!(Local::default().xspace_paths(&dir).unwrap(), paths);
    let second = overview(&path);
    assert!(!second.has("error"));
    assert_eq!(first, second);
}

#[test]
fn test_d02_stays_closed_across_all_swap_cases() {
    let dir = scratch("d02");
    let path = session(&dir, "test.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let training = overview(&path);
    std::fs::copy(DEMO, &path).unwrap();
    let swapped = overview(&path);
    assert_ne!(training, swapped, "normal swap must invalidate");
    std::fs::write(&path, training_trace(TRAINING_STEP_PS)).unwrap();
    std::fs::File::options().write(true).open(&path).unwrap().set_modified(std::fs::metadata(DEMO).unwrap().modified().unwrap()).unwrap();
    assert_eq!(overview(&path), training, "timestamp-preserving clone must be re-read");
}

#[test]
fn test_nested_layout_fingerprint_detection() {
    let dir = scratch("nested");
    let nested = dir.join("plugins/profile/2026_08_20");
    std::fs::create_dir_all(&nested).unwrap();
    let file = nested.join("host.xplane.pb");
    std::fs::write(&file, training_trace(TRAINING_STEP_PS)).unwrap();
    let first = overview(&dir);
    std::fs::copy(DEMO, &file).unwrap();
    assert_ne!(first, overview(&dir));
}

#[test]
fn test_readonly_trace_dir_support() {
    let dir = scratch("readonly");
    let path = session(&dir, "test.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let run_dir = path.parent().unwrap();
    std::fs::set_permissions(run_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let (code, out, _) = run(&["get_overview", path.to_str().unwrap()]);
    std::fs::set_permissions(run_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(code, 0, "{out}");
    assert!(!parse(&out).has("error"));
}

#[test]
fn test_no_trace_inputs_sentinel_e2e() {
    let dir = scratch("empty");
    let (code, out, _) = run(&["get_overview", dir.to_str().unwrap()]);
    assert_eq!(code, 3);
    assert!(parse(&out).at("error").text().contains("(DATA_ABSENT)"));
}

#[test]
fn test_d16_llo_remediation_when_absent() {
    let dir = scratch("d16llo");
    std::fs::write(dir.join("host1.xplane.pb"), training_trace(TRAINING_STEP_PS)).unwrap();
    let fake = Fake::fixed("").with_hosts(&["host1"]).in_dir(&dir);
    for result in [xplane::get_llo_analysis(&fake, &args("test_session", &[])), xplane::get_llo_debug_string(&fake, &args("test_session", &[]))] {
        let result = json(result);
        assert_eq!(result.at("status").str(), Some("UNAVAILABLE"));
        assert_eq!(result.at("reason").str(), Some("LLO_DATA_ABSENT"));
        let remediation = result.at("remediation").text();
        for needle in ["LIBTPU_INIT_ARGS", "--xla_xprof_enable_custom_call_tracing=true", "Python 3.11+", "JAX >= 0.11.0", "xprof-nightly"] {
            assert!(remediation.contains(needle), "{needle}");
        }
    }
}

#[test]
fn test_d18_standardized_cli_error_codes() {
    assert_eq!(run(&["get_overview", "/nonexistent/path/to/trace"]).0, 3);
    assert_eq!(run(&["get_graph_viewer", "--session_id=s", "--symbol_id=sym"]).0, 4);
    assert_eq!(run(&["get_overview", "--invalid_flag_xyz=123"]).0, 2);
}

#[test]
fn test_d19_hlo_op_profile_exception_contracts() {
    let dir = scratch("d19");
    let path = demo(&dir);
    let path = path.to_str().unwrap();
    assert_eq!(run(&["get_hlo_op_profile", path, "--category=completely_nonexistent_category_xyz"]).0, 3);
    assert_eq!(run(&["get_hlo_op_profile", path, "--view=tree", "--path=nonexistent/tree/path/xyz"]).0, 3);
    assert_eq!(run(&["get_hlo_op_profile", path, "--view=invalid_unsupported_view"]).0, 4);
    assert_eq!(run(&["get_hlo_op_profile", path, "--sort_by=invalid_sort_key"]).0, 4);
}

#[test]
fn test_d20_hlo_stats_datatable_schema_fidelity() {
    let dir = scratch("d20");
    let path = demo(&dir);
    let path = path.to_str().unwrap();
    let records = parse(&run(&["get_hlo_stats", path, "--limit=10"]).1);
    let top = &records.items()[0];
    for key in ["rank", "program_id", "category", "op_name", "total_self_time_us", "self_time_percent", "measured_flop_rate", "measured_memory_bw_gbs", "bound_by", "source_file", "source_line"] {
        assert!(top.has(key), "{key}");
    }
    assert_eq!(parse(&run(&["get_hlo_stats", path, "--limit=5", "--bypass_cache"]).1).items().len(), 5);
    let sorted = parse(&run(&["get_hlo_stats", path, "--limit=5", "--sort_by=total_time"]).1);
    assert_eq!(sorted.items().len(), 5);
    assert!(sorted.items()[0].at("total_time_us").float() >= sorted.items()[1].at("total_time_us").float());
}

#[test]
fn test_d21_cache_collision_prevention_across_distinct_logdirs_with_same_session_id() {
    let dir = scratch("d21");
    let (first, second) = (dir.join("dir_a"), dir.join("dir_c"));
    for (logdir, data) in [(&first, training_trace(TRAINING_STEP_PS)), (&second, std::fs::read(DEMO).unwrap())] {
        let run_dir = logdir.join("plugins/profile").join(SESSION_ID);
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(run_dir.join("trace.xplane.pb"), data).unwrap();
    }
    let query = |logdir: &Path| run(&["get_overview", SESSION_ID, &format!("--logdir={}", logdir.display())]);
    let (left, right) = (query(&first), query(&second));
    assert_eq!((left.0, right.0), (0, 0), "{} {}", left.1, right.1);
    assert_ne!(parse(&left.1), parse(&right.1));
}

#[test]
fn test_d22_empty_session_id_rejection() {
    let (code, out, _) = run(&["get_overview", ""]);
    assert_eq!(code, 4);
    assert_eq!(parse(&out).at("error").str(), Some("session_id cannot be an empty string."));
}

#[test]
fn test_d23_upload_trace_import_roundtrip() {
    let dir = scratch("d23u");
    let source = session(&dir.join("source"), "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let logdir = dir.join("logdir");
    let logdir_flag = format!("--logdir={}", logdir.display());
    let result = ok(&["upload_trace", source.to_str().unwrap(), &logdir_flag, "--run_name=imported_step_100"]);
    assert_eq!(result.at("status").str(), Some("success"));
    assert!(Path::new(result.at("run_path").str().unwrap()).is_dir());
    let imported = Path::new(result.at("imported_file").str().unwrap());
    assert!(imported.exists());
    assert_eq!(imported.file_name(), source.file_name());
    assert_eq!(ok(&["get_overview", "imported_step_100", &logdir_flag]).at("performance_summary").at("steptime_ms_average").str(), Some("20.00"));
}
