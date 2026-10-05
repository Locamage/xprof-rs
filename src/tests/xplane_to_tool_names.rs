use super::xspace::XSpace;
use crate::server::run_tools::tools;
use prost::Message;

const BASE_TOOLS: [&str; 8] = ["trace_viewer@", "overview_page", "input_pipeline_analyzer", "framework_op_stats", "memory_profile", "op_profile", "hlo_stats", "roofline_model"];

fn tools_list(plane_name: &str, has_hlo_module: bool, has_dcn_collective_stats: bool, has_perf_counters: bool, expected: &[&str]) {
    let mut space = XSpace::default();
    space.plane(plane_name);
    if has_hlo_module {
        space.add_plane().name = "/host:metadata".into();
        space.planes.last_mut().unwrap().stat_metadata("Hlo Proto");
    }
    if has_dcn_collective_stats {
        let host = space.add_plane();
        host.name = "/host:CPU".into();
        host.event_metadata("MegaScale:Megascale Event");
    }
    if has_perf_counters {
        let device = space.add_plane();
        device.name = "/device:TPU:".into();
        device.event(0, "dummy_event", 0, 0, &[("counter_value", 123.into())]);
    }
    let mut found = tools(&space.encode_to_vec()).unwrap();
    let mut wanted: Vec<&str> = BASE_TOOLS.iter().chain(expected).copied().collect();
    found.sort_unstable();
    wanted.sort_unstable();
    assert_eq!(found, wanted);
}

#[test]
fn tools_for_tpu_without_hlo_module() {
    tools_list("/device:TPU:", false, false, false, &[]);
}

#[test]
fn tools_for_tpu_with_hlo_module() {
    tools_list("/device:TPU:", true, false, false, &["graph_viewer", "memory_viewer"]);
}

#[test]
fn tools_for_gpu_without_hlo_module() {
    tools_list("/device:GPU:", false, false, false, &["kernel_stats"]);
}

#[test]
fn tools_for_gpu_with_hlo_module() {
    tools_list("/device:GPU:", true, false, false, &["kernel_stats", "graph_viewer", "memory_viewer"]);
}

#[test]
fn tools_for_tpu_with_dcn_collective_stats() {
    tools_list("/device:TPU:", false, true, false, &["megascale_stats"]);
}

#[test]
fn tools_for_tpu_with_perf_counters() {
    tools_list("/device:TPU:", false, false, true, &["perf_counters", "utilization_viewer", "kernel_utilization"]);
}
