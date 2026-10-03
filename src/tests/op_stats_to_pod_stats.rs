use super::op_stats_to_pod_viewer::{check_record, create_op_stats};
use crate::pod_viewer::render;
use serde_json::Value;

#[test]
fn gpu_pod_stats() {
    let pod_viewer_db: Value = serde_json::from_str(&render(&create_op_stats())).unwrap();
    let records: Vec<&Value> = pod_viewer_db["podStatsSequence"]["podStatsMap"].as_array().unwrap().iter().flat_map(|map| map["podStatsPerCore"].as_object().unwrap().values()).collect();
    assert_eq!(records.len(), 1);
    check_record(records[0]);
}
