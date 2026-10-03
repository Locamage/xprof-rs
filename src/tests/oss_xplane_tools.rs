use super::cli_support::{Fake, args, json, scratch};
use super::xspace::XSpace;
use crate::cli::Kind;
use crate::cli::json::J;
use crate::cli::xplane::list_xplane_events;
use prost::Message;

fn space(planes: &[&str]) -> Vec<u8> {
    let mut space = XSpace::default();
    for name in planes {
        let plane = space.plane(name);
        plane.named_line(0, "line1");
        plane.event(0, "event", 0, 10_000, &[]);
    }
    space.encode_to_vec()
}

#[test]
fn test_nonexistent_absolute_path_raises_filenotfounderror() {
    let error = list_xplane_events(&Fake::fixed(""), &args("/nonexistent/path/that/does/not/exist", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_empty_directory_raises_filenotfounderror() {
    let error = list_xplane_events(&Fake::fixed(""), &args(&scratch("xp-empty").to_string_lossy(), &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_directory_glob_finds_both_xplane_and_xspace_files() {
    let dir = scratch("xp-both");
    std::fs::write(dir.join("trace.xplane.pb"), space(&["plane1", "plane2"])).unwrap();
    std::fs::write(dir.join("trace.xspace.pb"), space(&["plane1", "plane2"])).unwrap();
    let result = json(list_xplane_events(&Fake::fixed(""), &args(&dir.to_string_lossy(), &[])));
    assert_eq!(result.at("total_matched"), &J::Int(4));
}

#[test]
fn test_existing_file_path_yields_planes() {
    let path = scratch("xp-file").join("trace.xplane.pb");
    std::fs::write(&path, space(&["plane1"])).unwrap();
    let result = json(list_xplane_events(&Fake::fixed(""), &args(&path.to_string_lossy(), &[])));
    let planes: Vec<&str> = result.at("events").items().iter().filter_map(|event| event.at("plane").str()).collect();
    assert_eq!(planes, ["plane1"]);
}

#[test]
fn test_nonexistent_relative_path_falls_through_to_server() {
    let dir = scratch("xp-relative");
    std::fs::write(dir.join("host.xplane.pb"), space(&["plane1"])).unwrap();
    let result = json(list_xplane_events(&Fake::fixed("").in_dir(&dir), &args("nonexistent_relative_session_id", &[])));
    assert_eq!(result.at("total_matched"), &J::Int(1));
}

#[test]
fn test_list_xplane_events_max_events_negative_one_returns_all() {
    let mut trace = XSpace::default();
    let plane = trace.plane("plane1");
    plane.named_line(0, "line1");
    plane.event(0, "event1", 0, 10_000, &[]);
    plane.event(0, "event2", 10_000, 10_000, &[]);
    let path = scratch("xp-limit").join("trace.xplane.pb");
    std::fs::write(&path, trace.encode_to_vec()).unwrap();
    let session = path.to_string_lossy().into_owned();
    let unlimited = json(list_xplane_events(&Fake::fixed(""), &args(&session, &[("max_events", J::Int(-1))]))).dumps();
    assert!(unlimited.contains("event1") && unlimited.contains("event2"));
    let one = json(list_xplane_events(&Fake::fixed(""), &args(&session, &[("max_events", J::Int(1))]))).dumps();
    assert!(one.contains("event1") && !one.contains("event2"));
}
