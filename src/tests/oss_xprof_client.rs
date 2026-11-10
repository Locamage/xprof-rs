use super::cli_support::scratch;
use crate::cli::Kind;
use crate::cli::client::{Client, Local};
use std::path::PathBuf;

fn listing(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|entry| entry.unwrap().path()).collect();
    found.sort();
    found
}

#[test]
fn test_set_logdir_none() {
    assert!(Local { logdir: None, ..Default::default() }.logdir().is_none());
}

#[test]
fn test_set_logdir_str() {
    assert!(Local { logdir: Some(PathBuf::from("/tmp/test")), ..Default::default() }.logdir().is_some());
}

#[test]
fn test_get_run_dir_with_direct_dir() {
    let dir = scratch("client-direct");
    assert_eq!(Local::default().run_dir(&dir.to_string_lossy()).unwrap(), dir);
}

#[test]
fn test_get_xspace_paths_single_file() {
    let file = scratch("client-single").join("test.xplane.pb");
    std::fs::write(&file, b"").unwrap();
    assert_eq!(Local::default().xspace_paths(&file).unwrap(), vec![file]);
}

#[test]
fn test_fetch_unknown_tool_raises_value_error() {
    let error = Local::default().fetch("non_existent_tool", "test_session", &[]).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
    assert!(error.message.contains("Unknown XProf tool name"));
}

#[test]
fn test_compute_fingerprint_uses_exact_xspace_paths() {
    let run = scratch("client-exact").join("run1");
    let nested = run.join("plugins/profile/2026_08_20");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(run.join("host.xplane.pb"), b"trace_bytes_1").unwrap();
    std::fs::write(nested.join("worker.xspace.pb"), b"trace_bytes_2").unwrap();
    let client = Local::default();
    let before = client.xspace_paths(&run).unwrap();
    assert_eq!(before.len(), 2);
    std::fs::write(run.join("ALL_HOSTS.op_stats_v2.pb"), b"generated_op_stats").unwrap();
    std::fs::write(run.join("summary.json"), r#"{"status": "ok"}"#).unwrap();
    std::fs::write(run.join(".xprof_trace_fingerprint"), "stale").unwrap();
    assert_eq!(client.xspace_paths(&run).unwrap(), before);
}

#[test]
fn test_compute_fingerprint_no_trace_inputs_sentinel() {
    let empty = scratch("client-empty").join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    assert_eq!(Local::default().xspace_paths(&empty).unwrap_err().kind, Kind::FileNotFound);
    std::fs::write(empty.join("notes.txt"), "not a trace").unwrap();
    assert_eq!(Local::default().xspace_paths(&empty).unwrap_err().kind, Kind::FileNotFound);
}

#[test]
fn test_external_atomic_fingerprint_storage() {
    let run = scratch("client-external").join("run_external");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("host.xplane.pb"), b"trace_data").unwrap();
    let before = listing(&run);
    assert_eq!(Local::default().fetch("overview_page.json", &run.to_string_lossy(), &[]).unwrap(), None);
    assert_eq!(listing(&run), before);
}

#[test]
fn test_fetch_readonly_trace_dir() {
    use std::os::unix::fs::PermissionsExt;
    let run = scratch("client-readonly").join("run_readonly");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("host.xplane.pb"), b"trace_data").unwrap();
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o555)).unwrap();
    let fetched = Local::default().fetch("overview_page.json", &run.to_string_lossy(), &[]);
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(fetched.is_ok());
}
