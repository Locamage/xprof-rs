use super::*;
use crate::tests::legacy::{bytes, entry as named, number};

fn plane(name: &str, rest: &[Vec<u8>]) -> Vec<u8> {
    bytes(1, &[bytes(2, name.as_bytes()), rest.concat()].concat())
}

fn counter_line(stat: u64) -> Vec<u8> {
    bytes(3, &bytes(4, &[number(1, 1), bytes(4, &[number(1, stat), number(4, 7)].concat())].concat()))
}

#[test]
fn tool_list_follows_planes_of_the_first_host() {
    let space = [
        plane("/host:metadata", &[named(5, 3, "Hlo Proto")]),
        plane("/host:CPU", &[named(4, 1, "MegaScale: Send")]),
        plane("/device:TPU:0", &[named(5, 9, "counter_value"), counter_line(9)]),
        plane("/device:GPU:0", &[]),
    ]
    .concat();
    assert_eq!(
        sorted(tools(&space).unwrap()),
        [
            "overview_page",
            "trace_viewer@",
            "graph_viewer",
            "op_profile",
            "input_pipeline_analyzer",
            "kernel_stats",
            "memory_profile",
            "memory_viewer",
            "roofline_model",
            "perf_counters",
            "framework_op_stats",
            "hlo_stats",
            "megascale_stats",
            "kernel_utilization",
            "utilization_viewer"
        ]
    );
}

#[test]
fn counters_count_only_on_tensor_core_events_carrying_the_stat() {
    let unused = [plane("/device:TPU:0", &[named(5, 9, "counter_value"), counter_line(8)]), plane("/host:CPU", &[named(5, 8, "counter_value"), counter_line(8)])].concat();
    assert_eq!(tools(&unused).unwrap(), BASE_TOOLS);
    let second_cpu = [plane("/host:CPU", &[]), plane("/host:CPU", &[named(4, 1, "MegaScale: Recv")])].concat();
    assert!(!tools(&second_cpu).unwrap().contains(&"megascale_stats"));
}

#[test]
fn malformed_or_non_utf8_spaces_are_rejected() {
    let truncated = plane("/host:CPU", &[]);
    assert!(tools(&truncated[..truncated.len() - 1]).is_none());
    assert!(tools(&bytes(1, &bytes(2, b"/host:\xff"))).is_none());
    assert!(tools(&[0x0f]).is_none());
    assert!(tools(&[bytes(9, b"\xff"), number(7, 1)].concat()).is_some());
}

#[test]
fn strings_escape_like_python_json() {
    assert_eq!(python_string("a\"\\\n\u{1}é😀\u{7f}"), "\"a\\\"\\\\\\n\\u0001\\u00e9\\ud83d\\ude00\\u007f\"");
}

#[test]
fn python_sort_puts_unknown_tools_last_alphabetically() {
    assert_eq!(sorted(["zeta", "trace_viewer", "trace_viewer@", "alpha", "hlo_stats"]), ["trace_viewer@", "hlo_stats", "alpha", "zeta"]);
}

#[test]
fn cache_round_trips_and_invalidates_on_file_changes() {
    let dir = crate::tests::temp_dir().join(format!("xprof-rs-run-tools-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("host.xplane.pb"), plane("/host:CPU", &[])).unwrap();
    assert_eq!(json(&dir), r#"["overview_page", "trace_viewer@", "op_profile", "input_pipeline_analyzer", "memory_profile", "roofline_model", "framework_op_stats", "hlo_stats"]"#);
    let written = std::fs::read_to_string(dir.join(CACHE_FILE)).unwrap();
    assert!(written.starts_with("{\n  \"files\": {\n    \"host.xplane.pb\": \"") && written.ends_with("\"\n  },\n  \"tools\": [\n    \"overview_page\",\n    \"trace_viewer@\",\n    \"op_profile\",\n    \"input_pipeline_analyzer\",\n    \"memory_profile\",\n    \"roofline_model\",\n    \"framework_op_stats\",\n    \"hlo_stats\"\n  ],\n  \"version\": 1\n}"), "{written}");
    std::fs::write(dir.join(CACHE_FILE), written.replace("\"hlo_stats\"\n", "\"other\"\n")).unwrap();
    assert!(json(&dir).ends_with("\"framework_op_stats\", \"other\"]"));
    std::fs::write(dir.join("host.xplane.pb"), b"\x0f").unwrap();
    assert_eq!(json(&dir), "[]");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn demo_sessions_cache_under_the_temp_directory() {
    assert_eq!(cache_path(Path::new("/logs/run/plugins/profile/s")), Path::new("/logs/run/plugins/profile/s/.cached_tools.json"));
    let demo = cache_path(Path::new("/logs/demo/plugins/profile/s"));
    assert!(demo.to_string_lossy().ends_with("_.cached_tools.json") && demo.file_name().unwrap().len() == "xprof_0123456789abcdef_.cached_tools.json".len());
    assert_eq!(cache_path(Path::new("/logs/mydemo/plugins/profile/s")), Path::new("/logs/mydemo/plugins/profile/s/.cached_tools.json"));
}

const TOOLS_1: [&str; 1] = ["tool1"];
const TOOLS_2: [&str; 2] = ["tool1", "tool2"];

fn tools_cache_dir(name: &str) -> PathBuf {
    crate::tests::scratch(&format!("tools-cache-{name}"))
}

fn saved(dir: &Path, tools: &[&str]) {
    save(&dir.join(CACHE_FILE), dir, &tools.iter().map(std::string::ToString::to_string).collect::<Vec<_>>());
}

fn loaded(dir: &Path) -> Option<Vec<Value>> {
    load(&dir.join(CACHE_FILE), dir)
}

fn listed(tools: &[&str]) -> Option<Vec<Value>> {
    Some(tools.iter().map(|tool| Value::String(tool.to_string())).collect())
}

#[test]
fn test_initialization() {
    let dir = tools_cache_dir("initialization");
    assert_eq!(cache_path(&dir), dir.join(".cached_tools.json"));
    assert!(!cache_path(&dir).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_save_and_load() {
    let dir = tools_cache_dir("save-and-load");
    std::fs::write(dir.join("host1.xplane.pb"), "").unwrap();
    std::fs::write(dir.join("host2.xplane.pb"), "").unwrap();
    saved(&dir, &TOOLS_2);
    assert!(dir.join(CACHE_FILE).exists());
    assert_eq!(loaded(&dir), listed(&TOOLS_2));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_no_cache_file() {
    let dir = tools_cache_dir("no-cache-file");
    assert_eq!(loaded(&dir), None);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_version_mismatch() {
    let dir = tools_cache_dir("version-mismatch");
    saved(&dir, &TOOLS_1);
    let text = std::fs::read_to_string(dir.join(CACHE_FILE)).unwrap();
    std::fs::write(dir.join(CACHE_FILE), text.replace("\"version\": 1", "\"version\": 0")).unwrap();
    assert_eq!(loaded(&dir), None);
    assert!(!dir.join(CACHE_FILE).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_file_mtime_changed() {
    let dir = tools_cache_dir("mtime-changed");
    let file = dir.join("host1.xplane.pb");
    std::fs::write(&file, "").unwrap();
    saved(&dir, &TOOLS_1);
    let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
    std::fs::File::options().write(true).open(&file).unwrap().set_modified(modified + std::time::Duration::from_secs(1)).unwrap();
    assert_eq!(loaded(&dir), None);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_file_added() {
    let dir = tools_cache_dir("file-added");
    std::fs::write(dir.join("host1.xplane.pb"), "").unwrap();
    saved(&dir, &TOOLS_1);
    std::fs::write(dir.join("host2.xplane.pb"), "").unwrap();
    assert_eq!(loaded(&dir), None);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_file_removed() {
    let dir = tools_cache_dir("file-removed");
    std::fs::write(dir.join("host1.xplane.pb"), "").unwrap();
    std::fs::write(dir.join("host2.xplane.pb"), "").unwrap();
    saved(&dir, &TOOLS_1);
    std::fs::remove_file(dir.join("host1.xplane.pb")).unwrap();
    assert_eq!(loaded(&dir), None);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_empty_directory() {
    let dir = tools_cache_dir("empty-directory");
    saved(&dir, &TOOLS_1);
    assert_eq!(loaded(&dir), listed(&TOOLS_1));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_corrupted_cache_file() {
    let dir = tools_cache_dir("corrupted");
    std::fs::write(dir.join(CACHE_FILE), "invalid json").unwrap();
    assert_eq!(loaded(&dir), None);
    assert!(!dir.join(CACHE_FILE).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_get_states_failed_on_save() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tools_cache_dir("states-failed");
    std::fs::write(dir.join("host1.xplane.pb"), "").unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o000)).unwrap();
    saved(&dir, &TOOLS_1);
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(!dir.join(CACHE_FILE).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_save_load_non_existent_directory() {
    let dir = tools_cache_dir("non-existent").join("non_existent");
    saved(&dir, &TOOLS_1);
    assert!(!dir.join(CACHE_FILE).exists());
    assert_eq!(loaded(&dir), None);
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}

#[test]
fn test_save_load_after_creating_directory() {
    let dir = tools_cache_dir("after-creating").join("was_non_existent");
    assert!(!dir.join(CACHE_FILE).exists());
    assert_eq!(loaded(&dir), None);
    std::fs::create_dir(&dir).unwrap();
    saved(&dir, &TOOLS_1);
    assert!(dir.join(CACHE_FILE).exists());
    assert_eq!(loaded(&dir), listed(&TOOLS_1));
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}

#[test]
fn test_get_xplane_file_states_success() {
    let dir = tools_cache_dir("file-states");
    for (name, content) in [("1.xplane.pb", "test"), ("2.txt", "test2"), ("3.xplane.riegeli", "test3")] {
        std::fs::write(dir.join(name), content).unwrap();
    }
    let states = file_states(&dir).unwrap();
    assert!(states.contains_key("1.xplane.pb") && states.contains_key("3.xplane.riegeli") && !states.contains_key("2.txt"));
    std::fs::remove_dir_all(&dir).unwrap();
}
