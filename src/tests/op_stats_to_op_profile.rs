use super::opstats_adapter::op_stats;
use serde_json::Value;

const PERF_ENV: &str = "perf_env { peak_tera_flops_per_second: 10.0 peak_bws_giga_bytes_per_second: [100.0, 100.0, 100.0] }";

fn create_op_metrics(name: &str, time: u64, category: &str, deduplicated_name: &str) -> String {
    format!(r#"metrics_db {{ name: "{name}" time_ps: {time} self_time_ps: {time} category: "{category}" occurrences: 1 deduplicated_name: "{deduplicated_name}" }}"#)
}

fn by_category(text: &str) -> Value {
    let profile: Value = serde_json::from_str(&crate::tools::op_profile::json(&op_stats(text), Some("category"))).unwrap();
    profile["byCategory"].clone()
}

fn names(node: &Value) -> Vec<&str> {
    node["children"].as_array().unwrap().iter().map(|child| child["name"].as_str().unwrap()).collect()
}

fn find<'a>(node: &'a Value, name: &str) -> &'a Value {
    node["children"].as_array().unwrap().iter().find(|child| child["name"] == name).unwrap()
}

#[test]
fn simple_profile_by_category() {
    let op = create_op_metrics("op1", 500, "convolution", "");
    let by_cat = by_category(&format!("device_op_metrics_db {{ total_time_ps: 1000 total_op_time_ps: 800 {op} }} {PERF_ENV}"));
    assert_eq!(names(&by_cat), ["convolution"]);
}

#[test]
fn deduplication_grouping_with_and_without_duplicates() {
    let ops = [
        create_op_metrics("conv_op_1", 500, "convolution", "conv_dedup"),
        create_op_metrics("conv_op_2", 400, "convolution", "conv_dedup"),
        create_op_metrics("conv_single_op", 300, "convolution", ""),
        create_op_metrics("fusion_op_1", 600, "fusion", "fusion_dedup"),
        create_op_metrics("fusion_op_2", 200, "fusion", "fusion_dedup"),
        create_op_metrics("fusion_single_op", 700, "fusion", ""),
        create_op_metrics("dense_single_op", 800, "dense", ""),
    ]
    .join(" ");
    let by_cat = by_category(&format!("device_op_metrics_db {{ total_time_ps: 10000 total_op_time_ps: 8000 {ops} }} {PERF_ENV}"));
    assert_eq!(by_cat["children"].as_array().unwrap().len(), 3);
    let (conv_cat, fusion_cat, dense_cat) = (find(&by_cat, "convolution"), find(&by_cat, "fusion"), find(&by_cat, "dense"));

    assert_eq!(conv_cat["children"].as_array().unwrap().len(), 2);
    let conv_dedup_node = find(conv_cat, "conv_op_1 and its duplicate(s)");
    assert_eq!(names(conv_dedup_node), ["conv_op_1", "conv_op_2"]);
    assert_eq!(names(find(conv_cat, "conv_single_op")).len(), 0);

    assert_eq!(fusion_cat["children"].as_array().unwrap().len(), 2);
    let fusion_dedup_node = find(fusion_cat, "fusion_op_1 and its duplicate(s)");
    assert_eq!(names(fusion_dedup_node), ["fusion_op_1", "fusion_op_2"]);
    assert_eq!(names(find(fusion_cat, "fusion_single_op")).len(), 0);

    assert_eq!(names(dense_cat), ["dense_single_op"]);
    assert_eq!(names(find(dense_cat, "dense_single_op")).len(), 0);
}
