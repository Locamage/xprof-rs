use super::xspace::XSpace;
use crate::megascale::{json, table};
use prost::Message;
use std::path::PathBuf;

const ALL_HOSTS: &str = "ALL_HOSTS";

struct SessionSnapshot {
    dir: PathBuf,
    path: PathBuf,
}

impl Drop for SessionSnapshot {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn create_session_snapshot(test_name: &str, has_dcn_collective_stats: bool) -> SessionSnapshot {
    let mut xspace = XSpace::default();
    let xplane = xspace.plane("/host:CPU");
    if has_dcn_collective_stats {
        xplane.event_metadata("MegaScale:");
    }
    let dir = std::env::temp_dir().join(format!("xprof-rs-dcn-{}-{test_name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("hostname.xplane.pb");
    std::fs::write(&path, xspace.encode_to_vec()).unwrap();
    SessionSnapshot { dir, path }
}

fn has_dcn_collective_stats_in_multi_x_space(session_snapshot: &SessionSnapshot) -> bool {
    crate::run_tools::tools(&std::fs::read(&session_snapshot.path).unwrap()).unwrap().contains(&"megascale_stats")
}

#[test]
fn has_all_hosts_dcn_collective_stats_cache_file() {
    let session_snapshot = create_session_snapshot("has_all_hosts", true);
    assert!(has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
}

#[test]
fn has_no_host_dcn_collective_stats_cache_file() {
    let session_snapshot = create_session_snapshot("has_no_host", false);
    assert!(!has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
}

#[test]
fn no_cache_file_but_trace_has_dcn_collective_stats() {
    let session_snapshot = create_session_snapshot("no_cache_has_stats", true);
    assert!(has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
}

#[test]
fn no_cache_file_no_dcn_collective_stats_present() {
    let session_snapshot = create_session_snapshot("no_cache_no_stats", false);
    assert!(!has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
}

#[test]
fn convert_x_space_to_dcn_collective_stats_when_stats_present() {
    let session_snapshot = create_session_snapshot("convert_present", true);
    let paths = [session_snapshot.path.clone()];
    assert!(has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
    assert!(json(&paths, ALL_HOSTS).is_some_and(|all_hosts| !all_hosts.is_empty()));
    assert!(json(&paths, "hostname").is_some_and(|host| !host.is_empty()));
}

#[test]
fn convert_x_space_to_dcn_collective_stats_when_stats_not_present() {
    let session_snapshot = create_session_snapshot("convert_absent", false);
    assert!(!has_dcn_collective_stats_in_multi_x_space(&session_snapshot));
    assert_eq!(json(std::slice::from_ref(&session_snapshot.path), "hostname"), Some(table(&[])));
}

#[test]
fn get_host_dcn_slack_analysis_when_stats_not_present() {
    let session_snapshot = create_session_snapshot("slack_absent", false);
    assert_eq!(json(std::slice::from_ref(&session_snapshot.path), "hostname"), Some(table(&[])));
}
