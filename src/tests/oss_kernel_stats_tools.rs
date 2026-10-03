use super::cli_support::{Fake, json, scratch};
use super::xspace::XSpace;
use crate::cli::Args;
use crate::cli::json::J;
use crate::cli::xplane::{get_avg_step_time, get_kernel_stats, union_ns};
use prost::Message;

const TPU: &str = "/device:TPU:0";
const MS: i64 = 1_000_000_000;

fn source(name: &str, space: &XSpace, flags: &[(&str, J)]) -> Args {
    let dir = scratch(name);
    std::fs::write(dir.join("host.xplane.pb"), space.encode_to_vec()).unwrap();
    let mut values = vec![("source".to_string(), J::from(dir.to_string_lossy().as_ref()))];
    values.extend(flags.iter().map(|(key, value)| (key.to_string(), value.clone())));
    Args { values }
}

fn ops(events: &[(&str, i64, i64)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.plane(TPU);
    plane.named_line(0, "XLA Ops");
    for (name, offset, duration) in events {
        plane.event(0, name, *offset, *duration, &[]);
    }
    space
}

#[test]
fn test_get_kernel_stats_tpu_filter() {
    let mut space = ops(&[("matmul_fwd", 0, 1400 * MS / 1000)]);
    let plane = space.plane(TPU);
    plane.named_line(1, "Auxiliary Core Sync Flag");
    plane.event(1, "sync_barrier", 0, 15 * MS, &[]);
    let records = json(get_kernel_stats(&Fake::fixed(""), &source("ks-filter", &space, &[])));
    assert_eq!(records.items().len(), 1);
    assert_eq!(records.items()[0].at("kernel_name").str(), Some("matmul_fwd"));
    assert_eq!(records.items()[0].at("total_duration_us"), &J::Float(1400.0));
    assert_eq!(records.items()[0].at("execution_count"), &J::Int(1));
}

#[test]
fn test_get_avg_step_time() {
    let mut space = XSpace::default();
    let plane = space.plane(TPU);
    plane.named_line(0, "XLA Modules");
    plane.event(0, "jit_train_step", 0, 15 * MS, &[]);
    plane.event(0, "jit_train_step", 20 * MS, 17 * MS, &[]);
    let result = json(get_avg_step_time(&Fake::fixed(""), &source("ks-steps", &space, &[("func_name", J::from("train_step"))])));
    assert_eq!(result.at("step_count"), &J::Int(2));
    assert_eq!(result.at("avg_step_time_ms"), &J::Float(16.0));
}

#[test]
fn test_compute_disjoint_interval_union_ns() {
    assert_eq!(union_ns(vec![(0, 100), (50, 150)]), 150);
    assert_eq!(union_ns(vec![(0, 100), (200, 300)]), 200);
    assert_eq!(union_ns(vec![(0, 200), (50, 100)]), 200);
    assert_eq!(union_ns(Vec::new()), 0);
}

#[test]
fn test_get_kernel_stats_with_include_summary() {
    let space = ops(&[("matmul_fwd", 0, 1400 * MS / 1000), ("dot", 700 * MS / 1000, 800 * MS / 1000)]);
    let result = json(get_kernel_stats(&Fake::fixed(""), &source("ks-summary", &space, &[("output_format", J::from("dict")), ("include_summary", J::Bool(true))])));
    for key in ["total_device_duration_ns", "total_device_duration_us", "total_device_duration_ms", "kernel_records", "step_durations_us", "stats"] {
        assert!(result.has(key), "{key}");
    }
    assert_eq!(result.at("total_device_duration_ns"), &J::Int(1500000));
    assert_eq!(result.at("total_device_duration_us"), &J::Float(1500.0));
    assert_eq!(result.at("kernel_records").items().len(), 2);
}

#[test]
fn test_get_kernel_stats_in_memory_profile_data() {
    let space = ops(&[("matmul_fwd", 0, 1400 * MS / 1000)]);
    let result = json(get_kernel_stats(&Fake::fixed(""), &source("ks-memory", &space, &[("output_format", J::from("dict"))])));
    assert_eq!(result.items().len(), 1);
    assert_eq!(result.items()[0].at("kernel_name").str(), Some("matmul_fwd"));
}

#[test]
fn test_get_kernel_stats_trace_matchers() {
    let space = ops(&[("matmul_fwd", 0, 1400 * MS / 1000), ("conv2d", 1400 * MS / 1000, 800 * MS / 1000)]);
    let flags = [("output_format", J::from("dict")), ("trace_matchers", J::List(vec![J::from("matmul")]))];
    let result = json(get_kernel_stats(&Fake::fixed(""), &source("ks-matchers", &space, &flags)));
    assert_eq!(result.items().len(), 1);
    assert_eq!(result.items()[0].at("kernel_name").str(), Some("matmul_fwd"));
}
