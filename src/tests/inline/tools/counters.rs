use super::*;
use crate::tests::legacy::{bytes as bytes_field, entry, number as number_field};
use crate::tools::counter_ids::V6E_IDS;

const CYCLES: &str = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES";
const BUSY: &str = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_MXU_BUSY_1";

fn stat(id: u64, value: Vec<u8>) -> Vec<u8> {
    [number_field(1, id), value].concat()
}

fn event(meta: u64, offset: u64, duration: u64, stats: &[Vec<u8>]) -> Vec<u8> {
    let mut body = [number_field(1, meta), number_field(2, offset), number_field(3, duration)].concat();
    for item in stats {
        body.extend(bytes_field(4, item));
    }
    bytes_field(4, &body)
}

fn line_bytes(id: u64, name: &str, events: &[Vec<u8>]) -> Vec<u8> {
    bytes_field(3, &[number_field(1, id), bytes_field(2, name.as_bytes()), events.concat()].concat())
}

fn space(group: bool) -> Vec<u8> {
    let stat_names = ["device_id", "device_type_string", "counter_value", "performance_counter_description", "performance_counter_sets", "global_chip_id", "performance_counter_id", "group_id"];
    let mut plane = bytes_field(2, b"/device:TPU:0");
    for (index, name) in stat_names.iter().enumerate().filter(|(index, _)| group || *index != 7) {
        plane.extend(entry(5, index as u64 + 1, name));
    }
    for (id, name) in [(10, "VF_Cycles"), (11, "Busy"), (20, "fusion.1"), (21, "fusion.0")] {
        plane.extend(entry(4, id, name));
    }
    plane.extend(bytes_field(6, &stat(1, number_field(4, 0))));
    plane.extend(bytes_field(6, &stat(2, bytes_field(5, b"TPU v6 Lite"))));
    plane.extend(bytes_field(6, &stat(6, number_field(3, 3))));
    plane.extend(line_bytes(7, "XLA Ops", &[event(20, 5, 10, &[]), event(21, 1, 3, &[])]));
    let counters = [
        event(10, 0, 2_000_000, &[stat(7, number_field(3, V6E_IDS[CYCLES])), stat(3, number_field(3, 1000)), stat(4, bytes_field(5, b"cycles")), stat(5, bytes_field(5, b"set\"a"))]),
        event(11, 0, 0, &[stat(7, number_field(3, V6E_IDS[BUSY])), stat(3, number_field(3, 500))]),
    ];
    plane.extend(line_bytes(11, "counters_0", &counters));
    let mut out = bytes_field(1, &plane);
    if !group {
        out.extend(bytes_field(1, &bytes_field(2, b"/host:CPU")));
    }
    out
}

#[test]
fn perf_counters_rows() {
    let map = space(false);
    assert!(valid_space(&map));
    let out = perf_counters(&[("host".to_string(), &map[..])]);
    assert_eq!(
        out,
        PERF_COLUMNS.to_string()
            + r#"{"c":[{"v":"host"},{"v":3.0},{"v":"counters_0"},{"v":11.0},{"v":"vf_cycles"},{"f":"0x3e8","v":1000.0},{"v":"cycles"},{"v":"set\"a"}]},{"c":[{"v":"host"},{"v":3.0},{"v":"counters_0"},{"v":11.0},{"v":"busy"},{"f":"0x1f4","v":500.0},{"v":""},{"v":""}]}]}"#
    );
    assert_eq!(perf_counters(&[]), PERF_COLUMNS.to_string() + "]}");
}

#[test]
fn utilization_viewer_rows() {
    let out = utilization_viewer(&space(false));
    let rows: Vec<&str> = out.split("{\"c\":").skip(1).collect();
    assert_eq!(rows.len(), 21 + 1 + 2 + 32);
    assert_eq!(rows[0], r#"[{"v":0.0},{"v":0.0},{"v":11.0},{"v":0.0},{"v":"Scalar Unit"},{"v":0.0},{"v":2000.0},{"v":"instructions"}]},"#);
    assert!(out.contains(r#"{"v":"1 MXU Busy"},{"v":500.0},{"v":1000.0},{"v":"cycles"}"#));
    assert!(utilization_viewer(&[]).ends_with(r#"{"id":"unit","label":"Unit","type":"string"}],"rows":[]}"#));
}

#[test]
fn kernel_utilization_report() {
    let out = kernel_utilization(&space(false), &Default::default());
    assert!(out.starts_with("{\n  \"devices\": [\n    {\n      \"device_id\": 0,\n      \"device_type\": \"TPU v6 Lite\",\n      \"kernels\": [\n        {\n          \"duration_us\": 2.0,\n          \"kernel_name\": \"fusion.0\",\n          \"mxu_cycles_breakdown\": {\n            \"BF16\": 0.0,"));
    assert!(out.contains("\"mxu_is_anomaly\": false,\n          \"mxu_utilization\": 25.0,\n          \"other_metrics\": {\n            \"1 MXU Busy\": 50.0,"));
    assert!(out.ends_with("\n          }\n        }\n      ]\n    }\n  ],\n  \"status\": \"SUCCESS\"\n}"));
    assert!(kernel_utilization(&space(true), &Default::default()).contains("\"kernel_name\": \"fusion.1\""));
    assert_eq!(kernel_utilization(&[], &Default::default()), "{\n  \"devices\": [],\n  \"status\": \"SUCCESS\"\n}");
}

#[test]
fn kernel_helpers() {
    assert_eq!(kernel_name("counters_7", "f"), "7");
    assert_eq!(kernel_name("counters_0", "f"), "f");
    assert_eq!(kernel_name("_counters_", ""), DEFAULT_KERNEL);
    assert_eq!(kernel_name("other", "f"), DEFAULT_KERNEL);
    assert_eq!([f64::NAN, -1.0, 1e30, 2.9, 0.0].map(|value| counter_value(&Value::Double(value))), [0, 0, u64::MAX, 2, 0]);
    let metric = |name: &str, achieved, peak| Metric { node: 0, name: name.into(), achieved, peak, unit: "cycles" };
    let kernel = summarize("k", 1.0, &[metric("Avg MXU Busy", 1.0, 3.0), metric("MXU BF16", 3.0, 0.0), metric("MXU I8", 1.0, 0.0), metric("HBM Rd+Wr - core 0", 1.0, 8.0), metric("Idle", 1.0, 0.0)]);
    assert_eq!(kernel["mxu_utilization"], 33.33);
    assert_eq!(kernel["mxu_cycles_breakdown"], json!({"BF16": 75.0, "FP8": 0.0, "Int4": 0.0, "Int8": 25.0}));
    assert_eq!(kernel["other_metrics"], json!({"HBM Bandwidth Utilization": 12.5}));
}

#[test]
fn validates_wire_format() {
    let map = space(false);
    assert!(!valid_space(&map[..map.len() - 1]));
    assert!(!valid_space(&bytes_field(1, &bytes_field(2, &[0xff]))));
    assert!(valid_space(&bytes_field(9, &[0xff])));
    assert!(!valid_space(&[0x0b]));
    assert!(valid_space(&[]));
    assert_eq!(chunks(&map).concat(), map);
}

#[test]
fn chunks_of_a_truncated_line() {
    let mut line = [0x08, 0x00].repeat(FIELDS_PER_CHUNK - 1);
    line.extend([0x09, 0x01, 0x02]);
    assert_eq!(chunks(&line).concat(), line);
    let long = [0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
    assert_eq!(chunks(&long).concat(), long);
}

#[test]
fn grouping_needs_group_id_on_every_plane() {
    assert!(sorted_samples(&planes(&space(false), |_| true)));
    assert!(!sorted_samples(&planes(&space(true), |_| true)));
    let mut map = space(true);
    map.extend(bytes_field(1, &bytes_field(2, b"/host:metadata")));
    assert!(sorted_samples(&planes(&map, |_| true)));
}
