use super::cli_support::{parse, run, scratch};
use super::e2e_oracles::{DEMO, TRAINING_STEP_PS, session, training_trace};
use super::xspace::XSpace;
use crate::cli::client::{Client, Local};
use crate::cli::{COMMANDS, Kind};
use prost::Message;

const SPILL_EVENTS: i64 = 80_000;

#[test]
fn test_x01_four_input_forms() {
    let dir = scratch("x01");
    let path = session(&dir, "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let step = |target: &std::path::Path| parse(&run(&["get_overview", target.to_str().unwrap()]).1).at("performance_summary").at("steptime_ms_average").clone();
    assert_eq!(step(&path), step(path.parent().unwrap()));
    assert_eq!(step(&path).str(), Some("20.00"));
}

#[test]
fn test_x05_truncation_contract() {
    let dir = scratch("x05");
    let path = session(&dir, "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let result = parse(&run(&["list_xplane_events", path.to_str().unwrap(), "--max_events=5"]).1);
    for key in ["events", "returned", "total_matched", "truncated"] {
        assert!(result.has(key), "{key}");
    }
    assert!(result.at("returned").int().unwrap() <= 5);
    assert!(result.at("total_matched").int().unwrap() > 5);
    assert_eq!(result.at("truncated"), &crate::cli::json::J::Bool(true));
}

#[test]
fn test_x06_volume_guard_spill_to_file() {
    let dir = scratch("x06");
    let mut space = XSpace::default();
    let plane = space.plane("/host:CPU");
    for index in 0..SPILL_EVENTS {
        plane.event(0, "a_reasonably_long_event_name_for_volume", index * 10, 5, &[]);
    }
    let path = session(&dir, "host.xplane.pb", &space.encode_to_vec());
    let (code, out, _) = run(&["list_xplane_events", path.to_str().unwrap(), "--max_events=0"]);
    let result = parse(&out);
    assert_eq!(code, 0);
    assert_eq!(result.at("status").str(), Some("SAVED_TO_FILE"));
    let spilled = std::path::PathBuf::from(result.at("file_path").str().unwrap());
    assert!(spilled.exists());
    assert!(result.at("size_bytes").int().unwrap() > 10 * 1024 * 1024);
    assert_eq!(parse(&std::fs::read_to_string(&spilled).unwrap()).at("returned").int(), Some(i128::from(SPILL_EVENTS)));
    std::fs::remove_file(spilled).unwrap();
}

#[test]
fn test_x07_cache_freshness_in_place_swap() {
    let dir = scratch("x07");
    let path = session(&dir, "trace.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let first = run(&["get_overview", path.to_str().unwrap()]).1;
    std::fs::copy(DEMO, &path).unwrap();
    let second = run(&["get_overview", path.to_str().unwrap()]).1;
    assert_ne!(first, second);
}

#[test]
fn test_x08_cache_error_rejection() {
    let dir = scratch("x08");
    let run_dir = dir.join("plugins/profile/run");
    std::fs::create_dir_all(&run_dir).unwrap();
    assert_eq!(run(&["get_overview", run_dir.to_str().unwrap()]).0, 3);
    session(&dir, "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let (code, out, _) = run(&["get_overview", run_dir.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert!(!parse(&out).has("error"));
}

#[test]
fn test_x09_registry_integrity() {
    assert!(!COMMANDS.is_empty());
    for (name, _, _) in COMMANDS {
        let (code, out, err) = run(&[name]);
        assert_ne!(code, 0, "{name}: {out}");
        assert!(!err.contains("Could not consume arg"), "{name}: {err}");
    }
}

#[test]
fn test_x10_unknown_tool_raises_value_error() {
    let error = Local::default().fetch("nonexistent_tool_xyz", "dummy", &[]).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}
